import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import glob
import random
import argparse

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from models import CycleTransMorph, NiftiDataset, PairedNiftiDataset
from losses import NCC, Grad, DiceLoss

# --- DEBUG: Function to print with rank ---
def print_rank(msg):
    rank = dist.get_rank() if dist.is_initialized() else 'N/A'
    print(f"--- [Rank {rank}] {msg}", flush=True)

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    print_rank(f"Initializing DDP process group for world size {world_size}...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print_rank("DDP process group initialized and CUDA device set.")

def cleanup_ddp():
    """Cleans up the distributed process group."""
    print_rank("Cleaning up DDP process group.")
    dist.destroy_process_group()

def get_file_list(data_dir, dataset_fraction):
    """
    Gets the list of files for training, ensuring all necessary files exist.
    This function should only be run on the main process (rank 0).
    """
    print_rank("Starting initial data scan...")
    inhale_files = sorted(glob.glob(os.path.join(data_dir, 'inhale', '*.nii.gz')))
    
    print_rank(f"Found {len(inhale_files)} potential inhale files. Now verifying pairs...")
    
    verified_files = []
    for inhale_file in tqdm(inhale_files, desc="[Rank 0] Verifying file sets"):
        base_name = os.path.basename(inhale_file)
        exhale_file = os.path.join(data_dir, 'exhale', base_name)
        inhale_mask_file = os.path.join(data_dir, 'inhale_mask', base_name)
        exhale_mask_file = os.path.join(data_dir, 'exhale_mask', base_name)

        if all(os.path.exists(f) for f in [exhale_file, inhale_mask_file, exhale_mask_file]):
            verified_files.append((inhale_file, exhale_file, inhale_mask_file, exhale_mask_file))

    print_rank(f"Found {len(verified_files)} complete data quadruplets.")

    if dataset_fraction < 1.0:
        num_files_to_use = int(len(verified_files) * dataset_fraction)
        verified_files = random.sample(verified_files, num_files_to_use)
        print_rank(f"Using a {dataset_fraction*100:.1f}% subset of data ({len(verified_files)} scans).")
        
    return verified_files

def train(args):
    """
    Main training function.
    """
    # DDP Setup
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_ddp(rank, world_size)

    # Logging and setup only on rank 0
    if rank == 0:
        print("--- [Rank 0] Initializing wandb...")
        wandb.init(project="exhale_pred", name="ctm_run")
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        print("--- [Rank 0] wandb and checkpoints directory initialized.")

    # Get file list on rank 0
    file_list = None
    if rank == 0:
        file_list = get_file_list(args.data_dir, args.dataset_fraction)
        # Convert file_list to a tensor or some object that can be broadcasted
        # For simplicity, we can just broadcast the list of lists.
        file_list_obj = [file_list]
    else:
        file_list_obj = [None]
        
    # Broadcast the file list from rank 0 to all other processes
    print_rank("Broadcasting file list from rank 0 to all other ranks...")
    dist.broadcast_object_list(file_list_obj, src=0)
    print_rank("File list broadcast complete.")
    
    # All processes now have the same file list
    file_list = file_list_obj[0]
    print_rank(f"Received {len(file_list)} files for training.")

    # Create dataset and dataloader
    print_rank("Creating dataset and dataloader...")
    dataset = PairedNiftiDataset(file_list)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, sampler=sampler, pin_memory=True)
    print_rank("Dataset and dataloader created.")

    # Model and optimizer
    print_rank("Initializing model and optimizer...")
    model = CycleTransMorph(img_size=(128, 128, 128)).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    print_rank("Model and optimizer initialized.")

    # Losses
    ncc_loss = NCC()
    grad_loss = Grad()
    dice_loss = DiceLoss()
    scaler = GradScaler()
    
    # Training loop
    print_rank("--- Starting Training Loop ---")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch) # Important for shuffling in DDP
        epoch_loss = 0
        
        # Wrap dataloader with tqdm only on rank 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if rank == 0 else dataloader

        for i, (inhale, exhale, inhale_mask, exhale_mask) in enumerate(pbar):
            print_rank(f"Batch {i+1}/{len(dataloader)} loaded.")
            
            inhale, exhale, inhale_mask, exhale_mask = inhale.to(rank), exhale.to(rank), inhale_mask.to(rank), exhale_mask.to(rank)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                warped_inhale, dvf_i_to_e, svf_i_to_e = ddp_model(inhale, exhale)
                warped_exhale, dvf_e_to_i, svf_e_to_i = ddp_model(exhale, inhale)
                
                # Reconstructions for cycle consistency
                reconstructed_inhale = ddp_model.module.spatial_transformer(warped_exhale, dvf_i_to_e)
                reconstructed_exhale = ddp_model.module.spatial_transformer(warped_inhale, dvf_e_to_i)

                # Loss calculation
                loss_recon_i = ncc_loss(reconstructed_inhale, inhale)
                loss_recon_e = ncc_loss(reconstructed_exhale, exhale)
                loss_sim_i = ncc_loss(warped_inhale, exhale)
                loss_sim_e = ncc_loss(warped_exhale, inhale)
                loss_reg = grad_loss(dvf_i_to_e) + grad_loss(dvf_e_to_i)
                
                total_loss = loss_sim_i + loss_sim_e + loss_recon_i + loss_recon_e + args.alpha * loss_reg

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            
            if rank == 0:
                pbar.set_postfix({"Loss": total_loss.item()})
                wandb.log({"loss": total_loss.item()})
        
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
            
            if (epoch + 1) % args.val_interval == 0:
                torch.save(ddp_model.module.state_dict(), f'checkpoints/ctm_epoch_{epoch+1}.pth')

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0, help="Regularization weight")
    parser.add_argument('--dataset_fraction', type=float, default=1.0)
    parser.add_argument('--latent_dim', type=int, default=16)
    args = parser.parse_args()
    
    train(args)