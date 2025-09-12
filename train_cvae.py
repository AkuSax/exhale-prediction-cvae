import argparse
import os
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --- FIX: Use torch.amp instead of torch.cuda.amp ---
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

# (DDP Setup and PairedLungDataset classes remain the same)
# --- DDP Setup ---
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

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
        
        inhale_scan = np.expand_dims(inhale_scan, axis=0)
        exhale_scan = np.expand_dims(exhale_scan, axis=0)
        
        return torch.from_numpy(exhale_scan), torch.from_numpy(inhale_scan)

# --- CVAE Model ---
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=256, condition_size=128):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_size = condition_size

        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(8 * 8 * 8 * 256, latent_dim)
        self.fc_var = nn.Linear(8 * 8 * 8 * 256, latent_dim)

        self.condition_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(32, self.condition_size)
        )
        self.decoder_input = nn.Linear(latent_dim + self.condition_size, 8 * 8 * 8 * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x, y):
        combined = torch.cat([x, y], dim=1)
        result = self.encoder(combined)
        result = torch.flatten(result, start_dim=1)
        return self.fc_mu(result), self.fc_var(result)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        processed_condition = self.condition_encoder(y)
        combined_latent = torch.cat([z, processed_condition], dim=1)
        result = self.decoder_input(combined_latent)
        result = result.view(-1, 256, 8, 8, 8)
        return self.decoder(result)

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var

# --- Loss Function ---
def loss_function(recon_x, x, mu, log_var, beta, recon_loss_fn):
    reconstruction_loss = recon_loss_fn(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = reconstruction_loss + beta * kld_loss
    return reconstruction_loss, kld_loss, total_loss

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D Conditional VAE.")
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="../data/processed")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--condition_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['mse', 'l1'])
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--log_images_every', type=int, default=10)
    parser.add_argument('--kl_anneal_epochs', type=int, default=40, help="Epochs for KL annealing.")
    return parser.parse_args()

# --- Main Training Loop ---
def main(args):
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if local_rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
        Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Data ---
    full_dataset = PairedLungDataset(Path(args.data_dir))
    train_sampler = DistributedSampler(full_dataset, drop_last=True)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    
    val_sampler = DistributedSampler(full_dataset, shuffle=False, drop_last=True)
    val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    
    # --- Model, Optimizer, Loss ---
    model = ConditionalVAE(latent_dim=args.latent_dim, condition_size=args.condition_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    recon_loss_fn = F.mse_loss if args.loss_fn == 'mse' else F.l1_loss
    scaler = GradScaler(device_type='cuda')
    best_val_loss = float('inf')

    # --- Training ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(local_rank!=0))
        
        for i, (real_exhale, real_inhale) in enumerate(train_iterator):
            real_exhale, real_inhale = real_exhale.to(local_rank), real_inhale.to(local_rank)
            
            if args.kl_anneal_epochs > 0:
                current_beta = args.beta * min(1.0, epoch / args.kl_anneal_epochs)
            else:
                current_beta = args.beta

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                recon_exhale, mu, log_var = model(real_exhale, real_inhale)
                recon_loss, kld_loss, loss = loss_function(recon_exhale, real_exhale, mu, log_var, current_beta, recon_loss_fn)
            
            scaler.scale(loss).backward()
            
            if local_rank == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                global_step = epoch * len(train_loader) + i
                wandb.log({
                    "train/step_loss": loss.item() / args.batch_size,
                    "train/step_recon_loss": recon_loss.item() / args.batch_size,
                    "train/step_kld_loss": kld_loss.item() / args.batch_size,
                    "train/grad_norm": grad_norm.item(),
                    "hyper/beta": current_beta,
                    "hyper/learning_rate": optimizer.param_groups[0]['lr'],
                    "latent/mu_mean": mu.mean().item(),
                    "latent/mu_std": mu.std().item(),
                    "latent/log_var_mean": log_var.mean().item(),
                    "latent/log_var_std": log_var.std().item(),
                }, step=global_step)
            
            scaler.step(optimizer)
            scaler.update()

        # --- Validation ---
        if (epoch + 1) % args.validate_every == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(local_rank!=0))
                for real_exhale, real_inhale in val_iterator:
                    real_exhale, real_inhale = real_exhale.to(local_rank), real_inhale.to(local_rank)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        recon_exhale, mu, log_var = model(real_exhale, real_inhale)
                        _, _, loss = loss_function(recon_exhale, real_exhale, mu, log_var, args.beta, recon_loss_fn)
                    total_val_loss += loss.item()
            
            if local_rank == 0:
                avg_val_loss = total_val_loss / len(val_loader.dataset)
                wandb.log({"val/epoch_loss": avg_val_loss, "epoch": epoch + 1}, step=global_step)
                
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_path = Path(args.model_save_dir) / "best_cvae_model.pth"
                    torch.save(model.module.state_dict(), model_path)

        # --- Image Logging ---
        if (epoch + 1) % args.log_images_every == 0 and local_rank == 0:
            model.eval()
            with torch.no_grad():
                fixed_exhale, fixed_inhale = next(iter(val_loader))
                fixed_exhale, fixed_inhale = fixed_exhale.to(local_rank), fixed_inhale.to(local_rank)
                with autocast(device_type='cuda', dtype=torch.float16):
                    recon_exhale, _, _ = model(fixed_exhale, fixed_inhale)

                slice_idx = fixed_exhale.shape[2] // 2
                real_slice = fixed_exhale[0, 0, slice_idx].cpu()
                recon_slice = recon_exhale[0, 0, slice_idx].cpu()
                condition_slice = fixed_inhale[0, 0, slice_idx].cpu()
                
                images = torch.stack([condition_slice, real_slice, recon_slice])
                grid = make_grid(images, nrow=3, normalize=True, pad_value=1)
                
                wandb.log({"val_images": wandb.Image(grid, caption=f"Epoch {epoch+1} | Inhale (Cond), Real Exhale, Recon Exhale")}, step=global_step)

    if local_rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    args = parse_args()
    main(args)