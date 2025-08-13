# ~/Fibrosis/Akul/exhale_pred/train_cvae.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.optim as optim

# --- Configuration ---
PROCESSED_DATA_DIR = Path("./processed_data")
BATCH_SIZE = 4 # Adjust based on VRAM
LATENT_DIM = 256
EPOCHS = 75 # Might need a bit more training than VAE
LEARNING_RATE = 1e-4

# --- 3D CVAE Model ---
class CVAE3D(nn.Module):
    def __init__(self):
        super(CVAE3D, self).__init__()
        # Encoder: Takes target (exhale) and condition (inhale)
        # We concatenate them on the channel dimension (1+1=2)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1), # -> 16x64x64x64
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),# -> 32x32x32x32
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),# -> 64x16x16x16
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),# -> 128x8x8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 * 8, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 8 * 8 * 8, LATENT_DIM)
        
        # Decoder: Takes latent z and condition (inhale)
        # We need to process the condition and merge it with z
        # Simple condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 64
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 32
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 16
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 8
            nn.Flatten()
        )
        self.decoder_input = nn.Linear(LATENT_DIM + (128 * 8 * 8 * 8), 128 * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c): # x is exhale (target), c is inhale (condition)
        # Encode
        combined_input = torch.cat([x, c], dim=1) # Concatenate along channel dim
        h = self.encoder(combined_input)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        c_features = self.condition_encoder(c)
        z_conditioned = torch.cat([z, c_features], dim=1)
        
        dec_input = self.decoder_input(z_conditioned).view(-1, 128, 8, 8, 8)
        recon_x = self.decoder(dec_input)
        
        return recon_x, mu, logvar

# --- Dataset for Paired Data ---
class PairedLungDataset(Dataset):
    def __init__(self, data_dir):
        self.inhale_files = sorted(list(Path(data_dir / "inhale").glob("*.npy")))
        self.exhale_files = sorted(list(Path(data_dir / "exhale").glob("*.npy")))
        print(f"Found {len(self.inhale_files)} inhale/exhale pairs.")

    def __len__(self):
        return len(self.inhale_files)

    def __getitem__(self, idx):
        inhale_scan = np.load(self.inhale_files[idx])
        exhale_scan = np.load(self.exhale_files[idx])
        # Add channel dimension
        return torch.from_numpy(exhale_scan).unsqueeze(0), torch.from_numpy(inhale_scan).unsqueeze(0)

def cvae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = PairedLungDataset(PROCESSED_DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = CVAE3D().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("💪 Starting CVAE training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for exhale_batch, inhale_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            exhale_batch = exhale_batch.to(device) # This is x
            inhale_batch = inhale_batch.to(device) # This is c
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(exhale_batch, inhale_batch)
            loss = cvae_loss_function(recon_batch, exhale_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch+1} Average Loss: {train_loss / len(train_loader.dataset):.4f}")

    torch.save(model.state_dict(), "cvae_model.pth")
    print("✅ CVAE training complete. Model saved to cvae_model.pth.")

if __name__ == "__main__":
    main()