import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from datasets import PairedLungDataset 

# --- VoxelMorph U-Net Architecture Components ---

class ConvBlock3D(nn.Module):
    """
    A convolutional block with two Conv3D layers, each followed by LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer Network.
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")

# --- VoxelMorph Model ---

class VoxelMorphNet(nn.Module):
    def __init__(self, in_channels=2, img_size=(128, 128, 128)):
        super().__init__()
        
        c = [16, 32, 32, 32]
        
        self.encoder1 = ConvBlock3D(in_channels, c[0])
        self.encoder2 = ConvBlock3D(c[0], c[1])
        self.encoder3 = ConvBlock3D(c[1], c[2])
        self.encoder4 = ConvBlock3D(c[2], c[3])

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock3D(c[3], c[3])

        self.upconv4 = nn.ConvTranspose3d(c[3], c[3], kernel_size=2, stride=2)
        self.decoder4 = ConvBlock3D(c[3] * 2, c[3])
        
        self.upconv3 = nn.ConvTranspose3d(c[3], c[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock3D(c[2] * 2, c[2])

        self.upconv2 = nn.ConvTranspose3d(c[2], c[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(c[1] * 2, c[1])

        self.upconv1 = nn.ConvTranspose3d(c[1], c[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(c[0] * 2, c[0])

        self.conv_last = nn.Conv3d(c[0], 3, kernel_size=3, stride=1, padding=1)
        
        self.spatial_transformer = SpatialTransformer(img_size)

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Output DVF
        dvf = self.conv_last(d1)
        
        # Warp the moving image
        warped_moving = self.spatial_transformer(moving, dvf)

        return warped_moving, dvf

# --- Loss Functions ---

class NCCLoss(nn.Module):
    """
    Normalized Cross-Correlation Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        eps = 1e-5
        # Reshape to 1D vectors
        y_true = y_true.view(y_true.shape[0], -1)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        
        # Compute means
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        
        # Compute standard deviations
        y_true_std = torch.std(y_true, dim=1, keepdim=True)
        y_pred_std = torch.std(y_pred, dim=1, keepdim=True)

        # Center the vectors
        y_true_centered = y_true - y_true_mean
        y_pred_centered = y_pred - y_pred_mean
        
        # Compute NCC
        corr = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true_std * y_pred_std * y_true.shape[1] + eps)
        
        # Return 1 - NCC as the loss (to be minimized)
        return 1 - torch.mean(corr)

class GradSmoothnessLoss(nn.Module):
    """
    Gradient smoothness loss for the DVF.
    """
    def __init__(self):
        super().__init__()

    def forward(self, dvf):
        # Calculate gradients in each dimension
        dx = torch.abs(dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :])
        dy = torch.abs(dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :])
        dz = torch.abs(dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1])

        return torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

# --- Argument Parser ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D VoxelMorph model.")
    parser.add_argument('--project_name', type=str, required=True, help="W&B project name.")
    parser.add_argument('--model_save_dir', type=str, required=True, help="Directory to save models.")
    parser.add_argument('--data_dir', type=str, default="../data/processed", help="Path to processed data.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (VoxelMorph is memory intensive).")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Adam optimizer learning rate.")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers.")
    parser.add_argument('--alpha', type=float, default=0.02, help="Regularization weight for DVF smoothness.")
    return parser.parse_args()

# --- Main Training Loop ---

def main(args):
    wandb.init(project=args.project_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Data Loading ---
    full_dataset = PairedLungDataset(Path(args.data_dir))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Model and Optimizer ---
    model = VoxelMorphNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    similarity_loss_fn = NCCLoss()
    smoothness_loss_fn = GradSmoothnessLoss()

    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        for exhale_batch, inhale_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            # VoxelMorph predicts exhale (fixed) from inhale (moving)
            moving_img = inhale_batch.to(device)
            fixed_img = exhale_batch.to(device)
            
            optimizer.zero_grad()
            
            warped_img, dvf = model(moving_img, fixed_img)
            
            similarity_loss = similarity_loss_fn(fixed_img, warped_img)
            smoothness_loss = smoothness_loss_fn(dvf)
            
            loss = similarity_loss + args.alpha * smoothness_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for exhale_batch, inhale_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                moving_img = inhale_batch.to(device)
                fixed_img = exhale_batch.to(device)
                
                warped_img, dvf = model(moving_img, fixed_img)
                
                similarity_loss = similarity_loss_fn(fixed_img, warped_img)
                smoothness_loss = smoothness_loss_fn(dvf)
                
                loss = similarity_loss + args.alpha * smoothness_loss
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = Path(args.model_save_dir) / "best_voxelmorph_model.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(str(model_path))

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)