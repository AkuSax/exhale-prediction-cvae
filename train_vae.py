# ~/Fibrosis/Akul/exhale_pred/train_vae.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.optim as optim

# --- Configuration ---
PROCESSED_DATA_DIR = Path("./processed_data")
BATCH_SIZE = 4 # Adjust based on VRAM usage
LATENT_DIM = 256
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- 3D VAE Model ---
class VAE3D(nn.Module):
    def __init__(self):
        super(VAE3D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16x64x64x64
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x32x32x32
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64x16x16x16
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),# -> 128x8x8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 * 8, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 8 * 8 * 8, LATENT_DIM)
        
        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, 128 * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # To output values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_reshaped = self.decoder_input(z).view(-1, 128, 8, 8, 8)
        return self.decoder(z_reshaped), mu, logvar

# --- Dataset ---
class LungDataset(Dataset):
    def __init__(self, data_dir):
        # We combine inhale and exhale scans for VAE training
        self.inhale_files = sorted(list(Path(data_dir / "inhale").glob("*.npy")))
        self.exhale_files = sorted(list(Path(data_dir / "exhale").glob("*.npy")))
        self.all_files = self.inhale_files + self.exhale_files
        print(f"Found {len(self.all_files)} total scans for VAE training.")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        scan_path = self.all_files[idx]
        scan = np.load(scan_path)
        return torch.from_numpy(scan).unsqueeze(0) # Add channel dimension

# --- Loss Function ---
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = LungDataset(PROCESSED_DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = VAE3D().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model) # Leverage both A6000s
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("💪 Starting VAE training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch+1} Average Loss: {train_loss / len(train_loader.dataset):.4f}")

    torch.save(model.state_dict(), "vae_model.pth")
    print("✅ VAE training complete. Model saved to vae_model.pth.")
    print("Next step: Use this model to encode all images and train a regressor on the latents.")

if __name__ == "__main__":
    main()