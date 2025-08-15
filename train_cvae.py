import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb
from torchvision.utils import make_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D CVAE on paired lung scan data.")
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="./processed_data")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader.")
    return parser.parse_args()

def log_images_to_wandb(model, dataloader, device, epoch):
    """Generates and logs a grid of original, reconstructed, and interpolated images for a cVAE."""
    model.eval()
    
    # Get a fixed batch of (exhale, inhale) pairs for consistent visualization
    exhale_originals, inhale_conditions = next(iter(dataloader))
    exhale_originals = exhale_originals.to(device)
    inhale_conditions = inhale_conditions.to(device)
    
    with torch.no_grad():
        # 1. Reconstructions
        reconstructions, _, _ = model(exhale_originals, inhale_conditions)

        # 2. Conditional Interpolation
        # Fix the condition (first inhale scan) and interpolate between two exhale latent spaces
        fixed_condition = inhale_conditions[0].unsqueeze(0)
        x1 = exhale_originals[0].unsqueeze(0)
        x2 = exhale_originals[1].unsqueeze(0)
        
        # Encode the two different exhale images with the same inhale condition
        _, mu, _ = model(torch.cat([x1, x2]), torch.cat([fixed_condition, fixed_condition]))
        z1, z2 = mu[0], mu[1]

        # Create 10 interpolation steps and decode using the fixed condition
        interp_steps = torch.linspace(0, 1, 10, device=device)
        interpolated_z = torch.stack([z1 * (1-step) + z2 * step for step in interp_steps])
        
        # To decode, we need to provide the latent vector AND the encoded condition
        # We need to get the features of the fixed condition
        c_features = model.module.condition_encoder(fixed_condition).repeat(10, 1)
        z_conditioned = torch.cat([interpolated_z, c_features], dim=1)
        
        dec_input = model.module.decoder_input(z_conditioned).view(-1, 128, 8, 8, 8)
        interpolated_images = model.module.decoder(dec_input)

    # Extract the central slice from each 3D volume for visualization
    slice_idx = exhale_originals.shape[2] // 2
    originals_2d = exhale_originals[:5, :, slice_idx, :, :].squeeze(2)
    recons_2d = reconstructions[:5, :, slice_idx, :, :].squeeze(2)
    interps_2d = interpolated_images[:, :, slice_idx, :, :].squeeze(2)

    # Create the grid using placeholders for alignment
    grid_originals = make_grid(originals_2d, nrow=5, padding=2, normalize=True)
    grid_recons = make_grid(recons_2d, nrow=5, padding=2, normalize=True)
    grid_interps = make_grid(interps_2d, nrow=10, padding=2, normalize=True)
    
    placeholder = torch.zeros_like(grid_originals)
    final_grid = torch.cat((grid_originals, placeholder[:, :, :0], grid_recons, placeholder[:, :, :0], grid_interps), dim=1)


    wandb.log({
        "visualizations": wandb.Image(final_grid, caption=f"Epoch {epoch+1}: Top:Originals, Mid:Recons, Btm:Conditional Interps")
    })

class CVAE3D(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8 * 8, latent_dim)

        self.condition_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        self.decoder_input = nn.Linear(latent_dim + (128 * 8 * 8 * 8), 128 * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        combined = torch.cat([x, c], dim=1)
        h = self.encoder(combined)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        c_features = self.condition_encoder(c)
        z_conditioned = torch.cat([z, c_features], dim=1)
        
        dec_input = self.decoder_input(z_conditioned).view(-1, 128, 8, 8, 8)
        return self.decoder(dec_input), mu, logvar

class PairedLungDataset(Dataset):
    def __init__(self, data_dir, num_samples=None):
        self.inhale_files = sorted(list(Path(data_dir) / "inhale").glob("*.npy"))
        self.exhale_files = sorted(list(Path(data_dir) / "exhale").glob("*.npy"))
        if num_samples: # Use a subset for consistent logging
            self.inhale_files = self.inhale_files[:num_samples]
            self.exhale_files = self.exhale_files[:num_samples]

    def __len__(self):
        return len(self.inhale_files)
        
    def __getitem__(self, idx):
        inhale = torch.from_numpy(np.load(self.inhale_files[idx])).unsqueeze(0)
        exhale = torch.from_numpy(np.load(self.exhale_files[idx])).unsqueeze(0)
        return exhale, inhale

def cvae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def main(args):
    wandb.init(project=args.project_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    full_dataset = PairedLungDataset(Path(args.data_dir))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    log_dataset = PairedLungDataset(Path(args.data_dir), num_samples=5)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    log_loader = DataLoader(log_dataset, batch_size=5, shuffle=False)
    
    model = CVAE3D(latent_dim=args.latent_dim).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for exhale_batch, inhale_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            exhale_batch, inhale_batch = exhale_batch.to(device), inhale_batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(exhale_batch, inhale_batch)
            loss = cvae_loss_function(recon_batch, exhale_batch, mu, logvar)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for exhale_batch, inhale_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                exhale_batch, inhale_batch = exhale_batch.to(device), inhale_batch.to(device)
                recon_batch, mu, logvar = model(exhale_batch, inhale_batch)
                loss = cvae_loss_function(recon_batch, exhale_batch, mu, logvar)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        # Log image grid for cVAE
        log_images_to_wandb(model, log_loader, device, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = Path(args.model_save_dir) / "best_cvae_model.pth"
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, model_path)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)