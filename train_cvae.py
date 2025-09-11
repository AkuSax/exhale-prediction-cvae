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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

# Dataset class
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

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, condition_size=128):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_size = condition_size

        # Encoder
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

        # Latent space layers
        self.fc_mu = nn.Linear(8 * 8 * 8 * 256, latent_dim)
        self.fc_var = nn.Linear(8 * 8 * 8 * 256, latent_dim)

        # Conditioning network 
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(32, self.condition_size)
        )

        self.decoder_input = nn.Linear(latent_dim + self.condition_size, 8 * 8 * 8 * 256)

        # Decoder architecture
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
        result = result.view(-1, 256, 8, 8, 8)
        result = self.decoder(result)
        return result

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var

def loss_function(recon_x, x, mu, log_var, beta, recon_loss_fn):
    reconstruction_loss = recon_loss_fn(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kld_loss

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
    parser.add_argument('--beta', type=float, default=1.0, help="Weight for the KL divergence term (β-VAE).")
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'l1'], help="Reconstruction loss.")
    return parser.parse_args()

# Main training loop
def main(args):
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if local_rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
        Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    full_dataset = PairedLungDataset(Path(args.data_dir))
    train_sampler = DistributedSampler(full_dataset)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

    if local_rank == 0:
        val_dataset = PairedLungDataset(Path(args.data_dir))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConditionalVAE(latent_dim=args.latent_dim, condition_size=args.condition_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.loss_fn == 'mse':
        recon_loss_fn = F.mse_loss
    elif args.loss_fn == 'l1':
        recon_loss_fn = F.l1_loss
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_fn}")

    best_val_loss = float('inf')

    # Training
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = 0
        
        train_iterator = tqdm(train_loader) if local_rank == 0 else train_loader
        
        for real_exhale, real_inhale in train_iterator:
            real_exhale, real_inhale = real_exhale.to(local_rank), real_inhale.to(local_rank)
            
            optimizer.zero_grad()
            recon_exhale, mu, log_var = model(real_exhale, real_inhale)
            loss = loss_function(recon_exhale, real_exhale, mu, log_var, args.beta, recon_loss_fn)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        if local_rank == 0:
            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for real_exhale, real_inhale in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    real_exhale, real_inhale = real_exhale.to(local_rank), real_inhale.to(local_rank)
                    recon_exhale, mu, log_var = model(real_exhale, real_inhale)
                    loss = loss_function(recon_exhale, real_exhale, mu, log_var, args.beta, recon_loss_fn)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = Path(args.model_save_dir) / "best_cvae_model.pth"
                torch.save(model.module.state_dict(), model_path)
                # wandb.save(str(model_path))
    
    if local_rank == 0:
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    args = parse_args()
    main(args)