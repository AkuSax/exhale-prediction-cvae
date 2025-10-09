import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# DDP-specific imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import wandb

# Assuming models.py and losses.py are in the same directory
from models import CycleTransMorph, SpatialTransformer
from losses import NCCLoss, GradientSmoothingLoss, CycleConsistencyLoss

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

# --- Dataset Class (remains the same) ---
class NumpyDataset(Dataset):
    def __init__(self, inhale_paths, exhale_paths, inhale_mask_paths, exhale_mask_paths):
        self.inhale_paths = inhale_paths
        self.exhale_paths = exhale_paths
        self.inhale_mask_paths = inhale_mask_paths
        self.exhale_mask_paths = exhale_mask_paths

    def __len__(self):
        return len(self.inhale_paths)

    def __getitem__(self, idx):
        inhale_img = np.load(self.inhale_paths[idx]).astype(np.float32)
        exhale_img = np.load(self.exhale_paths[idx]).astype(np.float32)
        inhale_mask = np.load(self.inhale_mask_paths[idx]).astype(np.float32)
        exhale_mask = np.load(self.exhale_mask_paths[idx]).astype(np.float32)
        inhale_img = np.expand_dims(inhale_img, axis=0)
        exhale_img = np.expand_dims(exhale_img, axis=0)
        inhale_mask = np.expand_dims(inhale_mask, axis=0)
        exhale_mask = np.expand_dims(exhale_mask, axis=0)
        return (torch.from_numpy(inhale_img), torch.from_numpy(exhale_img),
                torch.from_numpy(inhale_mask), torch.from_numpy(exhale_mask))

# --- Main Training Function ---
def train(args):
    setup_ddp()
    # DDP: Each process gets a unique rank. The local rank is the GPU ID.
    local_rank = int(os.environ['LOCAL_RANK'])
    device = local_rank

    # Data Loading
    # ... (robust data loading logic) ...
    train_inhale_paths, train_exhale_paths, train_inhale_mask_paths, train_exhale_mask_paths = [], [], [], []
    if local_rank == 0:
        print("--- Scanning for complete data quadruplets ---")
    inhale_dir, exhale_dir = os.path.join(args.data_dir, 'inhale'), os.path.join(args.data_dir, 'exhale')
    inhale_masks_dir = os.path.join(args.data_dir, 'masks', 'inhale')
    exhale_masks_dir = os.path.join(args.data_dir, 'masks', 'exhale')
    for f_name in sorted(os.listdir(inhale_dir)):
        base_name = f_name.replace('.npy', '')
        inhale_path, exhale_path = os.path.join(inhale_dir, f_name), os.path.join(exhale_dir, f_name)
        inhale_mask_path = os.path.join(inhale_masks_dir, f"{base_name}_INSP_mask.npy")
        exhale_mask_path = os.path.join(exhale_masks_dir, f"{base_name}_EXP_mask.npy")
        if all(os.path.exists(p) for p in [exhale_path, inhale_mask_path, exhale_mask_path]):
            train_inhale_paths.append(inhale_path)
            train_exhale_paths.append(exhale_path)
            train_inhale_mask_paths.append(inhale_mask_path)
            train_exhale_mask_paths.append(exhale_mask_path)
    if local_rank == 0:
        print(f"--- Found {len(train_inhale_paths)} complete data quadruplets. ---")

    # Debug mode slicing
    if args.debug:
        train_inhale_paths = train_inhale_paths[:8] # Use a few more for DDP testing
        train_exhale_paths = train_exhale_paths[:8]
        train_inhale_mask_paths = train_inhale_mask_paths[:8]
        train_exhale_mask_paths = train_exhale_mask_paths[:8]
        args.epochs = 2
        
    train_dataset = NumpyDataset(train_inhale_paths, train_exhale_paths, train_inhale_mask_paths, train_exhale_mask_paths)
    # DDP: Use DistributedSampler to ensure each GPU gets a unique part of the data
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)

    # Model Initialization
    model = CycleTransMorph(img_size=(args.img_size, args.img_size, args.img_size)).to(device)
    # DDP: Wrap the model
    model = DDP(model, device_ids=[local_rank])

    # WandB: Initialize only on the main process (rank 0)
    if local_rank == 0:
        wandb.init(project="exhale_prediction_cycletransmorph-ddp", config=args, mode="disabled" if args.debug else "online")

    stn = SpatialTransformer(size=(args.img_size, args.img_size, args.img_size)).to(device)
    sim_loss_fn = NCCLoss().to(device)
    reg_loss_fn = GradientSmoothingLoss().to(device)
    cycle_loss_fn = CycleConsistencyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"--- Starting Training on Rank {local_rank} ---")
    for epoch in range(args.epochs):
        model.train()
        # DDP: Set epoch on sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(local_rank != 0))
        for i, (inhale, exhale, inhale_mask, exhale_mask) in enumerate(pbar):
            inhale, exhale = inhale.to(device), exhale.to(device)
            inhale_mask, exhale_mask = inhale_mask.to(device), exhale_mask.to(device)
            
            optimizer.zero_grad()
            
            # The model forward pass is identical
            warped_inhale, dvf_i_to_e, svf_i_to_e = model(inhale, exhale)
            warped_exhale, dvf_e_to_i, svf_e_to_i = model(exhale, inhale)

            # Loss calculation is identical
            loss_sim = sim_loss_fn(warped_inhale, exhale, exhale_mask) + sim_loss_fn(warped_exhale, inhale, inhale_mask)
            loss_reg = reg_loss_fn(svf_i_to_e) + reg_loss_fn(svf_e_to_i)
            reconstructed_inhale = stn(warped_exhale, dvf_i_to_e)
            reconstructed_exhale = stn(warped_inhale, dvf_e_to_i)
            loss_cycle = cycle_loss_fn(inhale, reconstructed_inhale) + cycle_loss_fn(exhale, reconstructed_exhale)
            total_loss = loss_sim + args.lambda_reg * loss_reg + args.lambda_cycle * loss_cycle

            # DDP handles gradient averaging automatically during backward()
            total_loss.backward()
            optimizer.step()

            # Logging: Only from the main process
            if local_rank == 0:
                pbar.set_postfix({"Total": f"{total_loss.item():.4f}", "GPU": local_rank})
                wandb.log({"step_total_loss": total_loss.item()})
        
        # Saving: Only from the main process
        if local_rank == 0 and (epoch + 1) % args.save_interval == 0 and not args.debug:
            # DDP saves the underlying model's state_dict
            state_dict = model.module.state_dict()
            save_path = os.path.join(args.save_dir, f"cycletransmorph_ddp_epoch_{epoch+1}.pth")
            torch.save(state_dict, save_path)
            print(f"Model saved to {save_path}")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size PER GPU.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_cycle", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--debug", action='store_true', help="Run in debug mode")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
