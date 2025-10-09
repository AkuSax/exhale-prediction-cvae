# train_ctm.py (Corrected for DDP)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# Assuming models.py and losses.py are in the same directory
from models import CycleTransMorph, SpatialTransformer
from losses import NCCLoss, GradientSmoothingLoss, CycleConsistencyLoss

# --- DDP Helper Functions ---
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


# --- Updated Dataset Class for separate inhale/exhale masks ---
class NumpyDataset(Dataset):
    def __init__(self, inhale_paths, exhale_paths, inhale_mask_paths, exhale_mask_paths):
        self.inhale_paths = inhale_paths
        self.exhale_paths = exhale_paths
        self.inhale_mask_paths = inhale_mask_paths
        self.exhale_mask_paths = exhale_mask_paths

    def __len__(self):
        # All lists are guaranteed to be the same length by the data loading logic
        return len(self.inhale_paths)

    def __getitem__(self, idx):
        # Load the .npy arrays for images and both masks
        inhale_img = np.load(self.inhale_paths[idx]).astype(np.float32)
        exhale_img = np.load(self.exhale_paths[idx]).astype(np.float32)
        inhale_mask = np.load(self.inhale_mask_paths[idx]).astype(np.float32)
        exhale_mask = np.load(self.exhale_mask_paths[idx]).astype(np.float32)

        # Add channel dimension
        inhale_img = np.expand_dims(inhale_img, axis=0)
        exhale_img = np.expand_dims(exhale_img, axis=0)
        inhale_mask = np.expand_dims(inhale_mask, axis=0)
        exhale_mask = np.expand_dims(exhale_mask, axis=0)

        return (
            torch.from_numpy(inhale_img),
            torch.from_numpy(exhale_img),
            torch.from_numpy(inhale_mask),
            torch.from_numpy(exhale_mask),
        )

# --- Main Training Function ---
def train(args):
    # --- DDP Setup ---
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])

    # --- WandB Setup (only on rank 0) ---
    if local_rank == 0:
        if args.debug:
            print("--- RUNNING IN DEBUG MODE (WandB disabled) ---")
            wandb.init(mode="disabled")
        else:
            wandb.init(project="exhale_prediction_cycletransmorph", config=args)

    # --- Robust Data Loading ---
    if local_rank == 0:
        print("--- Scanning for complete data quadruplets (inhale, exhale, inhale_mask, exhale_mask) ---")
    
    inhale_dir = os.path.join(args.data_dir, 'inhale')
    exhale_dir = os.path.join(args.data_dir, 'exhale')
    inhale_masks_dir = os.path.join(args.data_dir, 'masks', 'inhale')
    exhale_masks_dir = os.path.join(args.data_dir, 'masks', 'exhale')

    train_inhale_paths, train_exhale_paths = [], []
    train_inhale_mask_paths, train_exhale_mask_paths = [], []

    # Use the inhale scans as the reference list
    for f_name in sorted(os.listdir(inhale_dir)):
        base_name = f_name.replace('.npy', '')
        inhale_path = os.path.join(inhale_dir, f_name)
        exhale_path = os.path.join(exhale_dir, f_name)
        inhale_mask_path = os.path.join(inhale_masks_dir, f"{base_name}_INSP_mask.npy")
        exhale_mask_path = os.path.join(exhale_masks_dir, f"{base_name}_EXP_mask.npy")

        if os.path.exists(exhale_path) and os.path.exists(inhale_mask_path) and os.path.exists(exhale_mask_path):
            train_inhale_paths.append(inhale_path)
            train_exhale_paths.append(exhale_path)
            train_inhale_mask_paths.append(inhale_mask_path)
            train_exhale_mask_paths.append(exhale_mask_path)
        elif local_rank == 0:
            print(f"--> Warning: Skipping incomplete quadruplet for base file: {f_name}")

    if local_rank == 0:
        print(f"--- Found {len(train_inhale_paths)} complete data quadruplets. ---")
    if len(train_inhale_paths) == 0:
        raise FileNotFoundError("No complete data quadruplets found! Check data directories and filenames.")

    # Apply debug slicing if enabled
    if args.debug:
        train_inhale_paths = train_inhale_paths[:4]
        train_exhale_paths = train_exhale_paths[:4]
        train_inhale_mask_paths = train_inhale_mask_paths[:4]
        train_exhale_mask_paths = train_exhale_mask_paths[:4]
        args.epochs = 2
        if local_rank == 0:
            print(f"--- Debug mode active. Using {len(train_inhale_paths)} scans for {args.epochs} epochs. ---")

    train_dataset = NumpyDataset(
        train_inhale_paths, train_exhale_paths, train_inhale_mask_paths, train_exhale_mask_paths
    )
    
    # --- DDP: Use DistributedSampler ---
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    # --- Model, Loss, and Optimizer ---
    model = CycleTransMorph(img_size=(args.img_size, args.img_size, args.img_size)).to(local_rank)
    
    # --- DDP: Wrap the model ---
    model = DDP(model, device_ids=[local_rank])
    
    stn = SpatialTransformer(size=(args.img_size, args.img_size, args.img_size)).to(local_rank)
    sim_loss_fn = NCCLoss().to(local_rank)
    reg_loss_fn = GradientSmoothingLoss().to(local_rank)
    cycle_loss_fn = CycleConsistencyLoss().to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    if local_rank == 0:
        print("--- Starting Training ---")
        
    for epoch in range(args.epochs):
        model.train()
        
        # --- DDP: Set the epoch for the sampler ---
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(local_rank != 0))

        for i, (inhale, exhale, inhale_mask, exhale_mask) in enumerate(pbar):
            inhale, exhale = inhale.to(local_rank), exhale.to(local_rank)
            inhale_mask, exhale_mask = inhale_mask.to(local_rank), exhale_mask.to(local_rank)

            optimizer.zero_grad()

            warped_inhale, dvf_i_to_e, svf_i_to_e = model(inhale, exhale)
            warped_exhale, dvf_e_to_i, svf_e_to_i = model(exhale, inhale)

            loss_sim_i_to_e = sim_loss_fn(warped_inhale, exhale, exhale_mask)
            loss_sim_e_to_i = sim_loss_fn(warped_exhale, inhale, inhale_mask)
            loss_sim = loss_sim_i_to_e + loss_sim_e_to_i

            loss_reg = reg_loss_fn(svf_i_to_e) + reg_loss_fn(svf_e_to_i)

            reconstructed_inhale = stn(warped_exhale, dvf_i_to_e)
            reconstructed_exhale = stn(warped_inhale, dvf_e_to_i)
            loss_cycle = cycle_loss_fn(inhale, reconstructed_inhale) + cycle_loss_fn(exhale, reconstructed_exhale)

            total_loss = loss_sim + args.lambda_reg * loss_reg + args.lambda_cycle * loss_cycle
            total_loss.backward()
            optimizer.step()
            
            if local_rank == 0:
                pbar.set_postfix({
                    "Total": f"{total_loss.item():.4f}", "Sim": f"{loss_sim.item():.4f}",
                    "Reg": f"{loss_reg.item():.4f}", "Cycle": f"{loss_cycle.item():.4f}"
                })
                wandb.log({
                    "step_total_loss": total_loss.item(), "step_sim_loss": loss_sim.item(),
                    "step_reg_loss": loss_reg.item(), "step_cycle_loss": loss_cycle.item()
                })

        if local_rank == 0 and (epoch + 1) % args.save_interval == 0 and not args.debug:
            save_path = os.path.join(args.save_dir, f"cycletransmorph_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing processed .npy files")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=128, help="Size of the input images (assumed to be cubes)")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the regularization loss")
    parser.add_argument("--lambda_cycle", type=float, default=1.0, help="Weight for the cycle consistency loss")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode with a small dataset subset")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)