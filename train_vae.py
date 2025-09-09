import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb
from torchvision.utils import make_grid
import math
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D VAE on lung scan data.")
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default="./processed_data")
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.1) # Lowered beta
    return parser.parse_args()

def log_images_to_wandb(model, dataloader, device, epoch):
    model.eval()
    fixed_batch = next(iter(dataloader)).to(device)
    
    with torch.no_grad():
        reconstructions, _, _ = model(fixed_batch)
        img1, img2 = fixed_batch[0], fixed_batch[-1]
        
        model_module = model.module
        
        encoded_output = model_module.encoder_conv(torch.stack([img1, img2]))
        mu, _ = model_module.fc_mu(encoded_output), model_module.fc_logvar(encoded_output)
        z1, z2 = mu[0], mu[1]

        interp_steps = torch.linspace(0, 1, 10, device=device)
        interpolated_z = torch.stack([z1 * (1 - step) + z2 * step for step in interp_steps])
        
        interp_reshaped = model_module.decoder_input(interpolated_z).view(-1, 1024, 2, 2, 2)
        interpolated_images = model_module.decoder(interp_reshaped)

    slice_idx = fixed_batch.shape[2] // 2
    originals_2d = fixed_batch[:, :, slice_idx, :, :]
    recons_2d = reconstructions[:, :, slice_idx, :, :]
    interps_2d = interpolated_images[:, :, slice_idx, :, :]

    grid_originals = make_grid(originals_2d, nrow=10, padding=2, normalize=True)
    grid_recons = make_grid(recons_2d, nrow=10, padding=2, normalize=True)
    grid_interps = make_grid(interps_2d, nrow=10, padding=2, normalize=True)
    
    final_grid = torch.cat((grid_originals, grid_recons, grid_interps), dim=1)

    wandb.log({"visualizations": wandb.Image(final_grid, caption=f"Epoch {epoch+1}")})

class SSIM3D(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM3D, self).__init__()
        self.window = self._create_window(window_size)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel=1):
        _1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t())
        _3D = _2D.unsqueeze(2) @ (_1D.t())
        window = _3D.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        window = self.window.to(img1.device).type(img1.dtype)
        mu1 = F.conv3d(img1, window, padding=5, groups=1)
        mu2 = F.conv3d(img2, window, padding=5, groups=1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv3d(img1*img1, window, padding=5, groups=1) - mu1_sq
        sigma2_sq = F.conv3d(img2*img2, window, padding=5, groups=1) - mu2_sq
        sigma12 = F.conv3d(img1*img2, window, padding=5, groups=1) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class VAE2D(nn.Module):
    def __init__(self, latent_dim):
        super(VAE2D, self).__init__()
        # Use a pre-trained ResNet18 as the encoder backbone
        resnet = resnet18(weights=None)
        # Adapt ResNet for single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 2x2 -> 4x4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()# 64x64 -> 128x128 -> THIS IS A MISTAKE! The target is 256x256. This needs another layer.
            # Correcting this live.
        )
        # Corrected decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)), # 1x1
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0), nn.ReLU(), # 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), # 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),   # 128x128
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()  # 256x256
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_reshaped = self.decoder_input(z)
        return self.decoder(z_reshaped), mu, logvar

class LungDataset(Dataset):
    def __init__(self, data_dir, num_log_samples=None):
        inhale_files = sorted(list(Path(data_dir / "inhale").glob("*.npy")))
        exhale_files = sorted(list(Path(data_dir / "exhale").glob("*.npy")))
        if num_log_samples:
            self.files = inhale_files[:num_log_samples//2] + exhale_files[:num_log_samples//2]
        else:
            self.files = inhale_files + exhale_files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        scan = np.load(self.files[idx])
        return torch.from_numpy(scan).unsqueeze(0)

ssim3d_loss = SSIM3D()
def vae_loss_function(recon_x, x, mu, logvar, beta):
    l1 = F.l1_loss(recon_x, x, reduction='sum')
    ssim = 1 - ssim3d_loss(recon_x, x)
    recon_loss = 0.85 * l1 + 0.15 * ssim * 1e6
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return individual components for logging
    return recon_loss + beta * kld, recon_loss, kld

def main(args):
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if local_rank == 0:
        wandb.init(project=args.project_name, config=vars(args))
        Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    full_dataset = LungDataset(Path(args.data_dir))
    train_sampler = DistributedSampler(full_dataset)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    
    if local_rank == 0:
        val_dataset = LungDataset(Path(args.data_dir))
        log_dataset = LungDataset(Path(args.data_dir), num_log_samples=10)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        log_loader = DataLoader(log_dataset, batch_size=10, shuffle=False)
    
    model = VAE3D(latent_dim=args.latent_dim).to(local_rank)
    ssim3d_loss.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # New: Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss, total_recon_loss, total_kld = 0, 0, 0
        
        train_iterator = tqdm(train_loader) if local_rank == 0 else train_loader

        for batch in train_iterator:
            batch = batch.to(local_rank)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss, recon, kld = vae_loss_function(recon_batch, batch, mu, logvar, args.beta)
            loss.backward()
            # New: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            total_recon_loss += recon.item()
            total_kld += kld.item()
        
        if local_rank == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(local_rank)
                    recon_batch, mu, logvar = model(batch)
                    loss, _, _ = vae_loss_function(recon_batch, batch, mu, logvar, args.beta)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            avg_recon_loss = total_recon_loss / len(train_loader.dataset)
            avg_kld = total_kld / len(train_loader.dataset)
            
            # New: Log individual loss components and learning rate
            wandb.log({
                "epoch": epoch + 1, 
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss,
                "train_recon_loss": avg_recon_loss,
                "train_kld": avg_kld,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            log_images_to_wandb(model, log_loader, local_rank, epoch)
            
            # New: Step the scheduler
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = Path(args.model_save_dir) / "best_vae_model.pth"
                torch.save(model.module.state_dict(), model_path)

    cleanup_ddp()

if __name__ == "__main__":
    args = parse_args()
    main(args)
