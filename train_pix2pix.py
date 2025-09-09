import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

# (Assuming datasets.py contains the PairedLungDataset)
from datasets import PairedLungDataset

# --- 3D U-Net Generator ---

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = UNetDownBlock(in_channels, 64, normalize=False)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256)
        self.down4 = UNetDownBlock(256, 512, dropout=0.5)
        self.down5 = UNetDownBlock(512, 512, dropout=0.5)
        self.down6 = UNetDownBlock(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUpBlock(512, 512, dropout=0.5)
        self.up2 = UNetUpBlock(1024, 512, dropout=0.5)
        self.up3 = UNetUpBlock(1024, 256)
        self.up4 = UNetUpBlock(512, 128)
        self.up5 = UNetUpBlock(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)

# --- 3D PatchGAN Discriminator ---

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# --- Argument Parser ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D Pix2Pix (Vox2Vox) model.")
    parser.add_argument('--project_name', type=str, required=True, help="W&B project name.")
    parser.add_argument('--model_save_dir', type=str, required=True, help="Directory to save models.")
    parser.add_argument('--data_dir', type=str, default="../data/processed", help="Path to processed data.")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Adam optimizer learning rate.")
    parser.add_argument('--b1', type=float, default=0.5, help="Adam: beta1.")
    parser.add_argument('--b2', type=float, default=0.999, help="Adam: beta2.")
    parser.add_argument('--lambda_l1', type=float, default=100.0, help="Weight for L1 loss.")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers.")
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
    
    # --- Models, Losses, and Optimizers ---
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    
    best_val_l1 = float('inf')
    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        for real_exhale, real_inhale in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            real_inhale = real_inhale.to(device) # Condition
            real_exhale = real_exhale.to(device) # Target
            
            # Adversarial ground truths
            valid = torch.ones(real_exhale.size(0), 1, 4, 4, 4, device=device)
            fake = torch.zeros(real_exhale.size(0), 1, 4, 4, 4, device=device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            
            fake_exhale = generator(real_inhale)
            pred_fake = discriminator(fake_exhale, real_inhale)
            
            loss_GAN = adversarial_loss_fn(pred_fake, valid)
            loss_L1 = l1_loss_fn(fake_exhale, real_exhale)
            
            loss_G = loss_GAN + args.lambda_l1 * loss_L1
            
            loss_G.backward()
            optimizer_G.step()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            pred_real = discriminator(real_exhale, real_inhale)
            loss_real = adversarial_loss_fn(pred_real, valid)
            
            pred_fake = discriminator(fake_exhale.detach(), real_inhale)
            loss_fake = adversarial_loss_fn(pred_fake, fake)
            
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()

        # --- Validation ---
        generator.eval()
        total_val_l1 = 0
        with torch.no_grad():
            for real_exhale, real_inhale in val_loader:
                real_inhale = real_inhale.to(device)
                real_exhale = real_exhale.to(device)
                
                fake_exhale = generator(real_inhale)
                total_val_l1 += l1_loss_fn(fake_exhale, real_exhale).item()
        
        avg_val_l1 = total_val_l1 / len(val_loader)
        
        wandb.log({
            "epoch": epoch + 1,
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "val_L1": avg_val_l1,
        })

        if avg_val_l1 < best_val_l1:
            best_val_l1 = avg_val_l1
            torch.save(generator.state_dict(), Path(args.model_save_dir) / "best_pix2pix_generator.pth")
            
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)