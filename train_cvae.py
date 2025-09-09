import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb

# --- PyTorch Dataset ---
class PairedLungDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.inhale_dir = data_dir / "inhale"
        self.exhale_dir = data_dir / "exhale"
        self.patient_ids = sorted([f.stem for f in self.inhale_dir.glob("*.npy")])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        inhale_path = self.inhale_dir / f"{patient_id}.npy"
        exhale_path = self.exhale_dir / f"{patient_id}.npy"
        
        inhale_scan = np.load(inhale_path).astype(np.float32)
        exhale_scan = np.load(exhale_path).astype(np.float32)
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        inhale_scan = np.expand_dims(inhale_scan, axis=0)
        exhale_scan = np.expand_dims(exhale_scan, axis=0)
        
        return torch.from_numpy(exhale_scan), torch.from_numpy(inhale_scan)

# --- CVAE Model with Improved Conditioning ---
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, condition_size=128):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_size = condition_size

        # --- Encoder ---
        # Input channels = 2 (exhale scan + inhale scan)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
        )

        # --- Latent Space Layers ---
        self.fc_mu = nn.Linear(4 * 4 * 4 * 256, latent_dim)
        self.fc_var = nn.Linear(4 * 4 * 4 * 256, latent_dim)

        # --- NEW: Conditioning Network ---
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(32, self.condition_size)
        )

        # --- Modified Decoder Input ---
        self.decoder_input = nn.Linear(latent_dim + self.condition_size, 4 * 4 * 4 * 256)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Sigmoid for [0, 1] output range
        )

    def encode(self, x, y):
        combined = torch.cat([x, y], dim=1)
        result = self.encoder(combined)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        processed_condition = self.condition_encoder(y)
        combined_latent = torch.cat([z, processed_condition], dim=1)
        result = self.decoder_input(combined_latent)
        result = result.view(-1, 256, 4, 4, 4)
        result = self.decoder(result)
        return result

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var

# --- Loss Function with Beta and selectable Recon Loss ---
def loss_function(recon_x, x, mu, log_var, beta, recon_loss_fn):
    reconstruction_loss = recon_loss_fn(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kld_loss

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D Conditional VAE.")
    parser.add_argument('--project_name', type=str, required=True, help="W&B project name.")
    parser.add_argument('--model_save_dir', type=str, required=True, help="Directory to save models.")
    parser.add_argument('--data_dir', type=str, default="../data/processed", help="Path to processed data.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--condition_size', type=int, default=128, help="Size of processed condition vector.")
    parser.add_argument('--num_workers', type=int, default=4)
    # New arguments
    parser.add_argument('--beta', type=float, default=1.0, help="Weight for the KL divergence term (β-VAE).")
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'l1'], help="Reconstruction loss.")
    return parser.parse_args()

# --- Main Training Loop ---
def main(args):
    wandb.init(project=args.project_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Data ---
    full_dataset = PairedLungDataset(Path(args.data_dir))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Model, Optimizer, and Loss ---
    model = ConditionalVAE(latent_dim=args.latent_dim, condition_size=args.condition_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.loss_fn == 'mse':
        recon_loss_fn = F.mse_loss
    elif args.loss_fn == 'l1':
        recon_loss_fn = F.l1_loss
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")

    best_val_loss = float('inf')

    # --- Training ---
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for real_exhale, real_inhale in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            real_exhale, real_inhale = real_exhale.to(device), real_inhale.to(device)
            
            optimizer.zero_grad()
            recon_exhale, mu, log_var = model(real_exhale, real_inhale)
            loss = loss_function(recon_exhale, real_exhale, mu, log_var, args.beta, recon_loss_fn)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for real_exhale, real_inhale in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                real_exhale, real_inhale = real_exhale.to(device), real_inhale.to(device)
                recon_exhale, mu, log_var = model(real_exhale, real_inhale)
                loss = loss_function(recon_exhale, real_exhale, mu, log_var, args.beta, recon_loss_fn)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = Path(args.model_save_dir) / "best_cvae_model.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(str(model_path))

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)