import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from models import CycleTransMorph
from losses import NCCLoss, GradientSmoothingLoss, CycleConsistencyLoss
from torch.cuda.amp import GradScaler, autocast
import wandb

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # --- CHANGE HERE: Using 'gloo' backend for robust debugging ---
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method='env://')

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

class NpyDataset(Dataset):
    """
    Custom PyTorch Dataset that loads inhale/exhale pairs from a directory structure.
    """
    def __init__(self, data_dir, file_names):
        self.data_dir = data_dir
        self.file_names = file_names
        self.inhale_dir = os.path.join(data_dir, 'inhale')
        self.exhale_dir = os.path.join(data_dir, 'exhale')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        
        inhale_path = os.path.join(self.inhale_dir, filename)
        exhale_path = os.path.join(self.exhale_dir, filename)
        
        inhale = torch.from_numpy(np.load(inhale_path)).float().unsqueeze(0)
        exhale = torch.from_numpy(np.load(exhale_path)).float().unsqueeze(0)
        
        return inhale, exhale

def train(args):
    """
    Main training function.
    """
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0 and not args.debug:
        wandb.init(project="exhale_pred", name=args.run_name)
    
    # --- DDP-Aware Data Scanning ---
    if rank == 0: 
        print("--- [Rank 0] Starting data scan... ---", flush=True)

    inhale_dir = os.path.join(args.data_dir, 'inhale')
    exhale_dir = os.path.join(args.data_dir, 'exhale')
    inhale_mask_dir = os.path.join(args.data_dir, 'masks', 'inhale')
    exhale_mask_dir = os.path.join(args.data_dir, 'masks', 'exhale')

    complete_files = []
    # Only the main process (rank 0) performs the scan
    if rank == 0:
        try:
            print("--- [Rank 0] Getting initial file list from 'inhale' directory... ---", flush=True)
            reference_files = os.listdir(inhale_dir)
            print(f"--- [Rank 0] Found {len(reference_files)} total files. Now verifying sets... ---", flush=True)
        except FileNotFoundError:
            print(f"Error: Could not find the 'inhale' directory at {inhale_dir}", flush=True)
            reference_files = [] # Ensure the list is empty to signal an error
        
        # This loop will now have a progress bar
        for filename in tqdm(reference_files, desc="[Rank 0] Verifying file sets"):
            base_id = filename.replace('.npy', '')
            inhale_mask_name = f"{base_id}_INSP_mask.npy"
            exhale_mask_name = f"{base_id}_EXP_mask.npy"

            if os.path.exists(os.path.join(exhale_dir, filename)) and \
               os.path.exists(os.path.join(inhale_mask_dir, inhale_mask_name)) and \
               os.path.exists(os.path.join(exhale_mask_dir, exhale_mask_name)):
                complete_files.append(filename)
    
    # Synchronize the file list across all GPUs
    if world_size > 1:
        dist.barrier()
        
        file_list_size = torch.tensor(len(complete_files) if rank == 0 else 0, dtype=torch.long, device=rank)
        dist.broadcast(file_list_size, src=0)

        if rank != 0:
            complete_files = [None] * file_list_size.item()
        
        for i in range(file_list_size.item()):
            if rank == 0:
                filename_bytes = complete_files[i].encode('utf-8')
                filename_len = torch.tensor(len(filename_bytes), dtype=torch.long, device=rank)
            else:
                filename_len = torch.empty(1, dtype=torch.long, device=rank)
            dist.broadcast(filename_len, src=0)

            if rank == 0:
                filename_tensor = torch.tensor(list(filename_bytes), dtype=torch.uint8, device=rank)
            else:
                filename_tensor = torch.empty(filename_len.item(), dtype=torch.uint8, device=rank)
            dist.broadcast(filename_tensor, src=0)

            if rank != 0:
                complete_files[i] = filename_tensor.cpu().numpy().tobytes().decode('utf-8')
        
        dist.barrier()

    if rank == 0: print(f"--- Found {len(complete_files)} complete data quadruplets. ---", flush=True)
    
    if args.dataset_fraction < 1.0:
        num_samples = int(len(complete_files) * args.dataset_fraction)
        random.shuffle(complete_files)
        complete_files = complete_files[:num_samples]
        if rank == 0: print(f"--- Using a {args.dataset_fraction*100:.1f}% subset of data ({len(complete_files)} scans) ---", flush=True)

    if args.debug:
        complete_files = complete_files[:4]
        args.epochs = 2
        if rank == 0: print(f"--- Debug mode active. Using {len(complete_files)} scans for {args.epochs} epochs. ---", flush=True)

    train_dataset = NpyDataset(data_dir=args.data_dir, file_names=complete_files)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler)
    
    model = CycleTransMorph().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    sim_loss = NCCLoss().to(rank)
    reg_loss = GradientSmoothingLoss().to(rank)
    cycle_loss = CycleConsistencyLoss().to(rank)
    scaler = GradScaler()
    
    if rank == 0: print("--- Starting Training ---", flush=True)
    
    fixed_viz_inhale, fixed_viz_exhale = next(iter(train_loader))
    fixed_viz_inhale = fixed_viz_inhale.to(rank)
    fixed_viz_exhale = fixed_viz_exhale.to(rank)

    for epoch in range(args.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank!=0))
        
        for i, (inhale, exhale) in enumerate(progress_bar):
            inhale, exhale = inhale.to(rank), exhale.to(rank)
            optimizer.zero_grad()
            
            with autocast():
                warped_inhale, dvf_i_to_e, svf_i_to_e = ddp_model(inhale, exhale)
                warped_exhale, dvf_e_to_i, svf_e_to_i = ddp_model(exhale, inhale)
                
                reconstructed_inhale = model.module.spatial_transformer(warped_exhale, dvf_i_to_e)
                reconstructed_exhale = model.module.spatial_transformer(warped_inhale, dvf_e_to_i)
                
                loss_sim = sim_loss(warped_inhale, exhale)
                loss_reg = reg_loss(svf_i_to_e) + reg_loss(svf_e_to_i)
                loss_cycle = cycle_loss(inhale, reconstructed_inhale) + cycle_loss(exhale, reconstructed_exhale)
                total_loss = loss_sim + args.reg_lambda * loss_reg + args.cycle_lambda * loss_cycle

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0 and not args.debug:
                step = epoch * len(train_loader) + i
                wandb.log({"step_total_loss": total_loss.item(), "step_sim_loss": loss_sim.item(), 
                           "step_reg_loss": loss_reg.item(), "step_cycle_loss": loss_cycle.item()}, step=step)

        if rank == 0 and not args.debug:
            if (epoch + 1) % args.viz_interval == 0:
                ddp_model.eval()
                with torch.no_grad(), autocast():
                    slice_idx = fixed_viz_inhale.shape[2] // 2
                    sample_inhale = fixed_viz_inhale[0:1]
                    sample_exhale = fixed_viz_exhale[0:1]
                    
                    warped_inhale_viz, dvf_i_to_e_viz, _ = ddp_model(sample_inhale, sample_exhale)
                    
                    inhale_slice = sample_inhale.squeeze().cpu().numpy()[slice_idx, :, :]
                    exhale_slice = sample_exhale.squeeze().cpu().numpy()[slice_idx, :, :]
                    warped_inhale_slice = warped_inhale_viz.squeeze().cpu().numpy()[slice_idx, :, :]
                    dvf_mag_slice = torch.norm(dvf_i_to_e_viz.squeeze().cpu(), dim=0).numpy()[slice_idx, :, :]
                    
                    images = [
                        wandb.Image(inhale_slice, caption="Original Inhale"),
                        wandb.Image(exhale_slice, caption="Target Exhale"),
                        wandb.Image(warped_inhale_slice, caption="Warped Inhale"),
                        wandb.Image(dvf_mag_slice, caption="DVF Magnitude")
                    ]
                    wandb.log({"Visualizations": images}, step=(epoch + 1) * len(train_loader))
                ddp_model.train()

            if (epoch + 1) % args.val_interval == 0:
                torch.save(ddp_model.module.state_dict(), f"ctm_epoch_{epoch+1}.pth")
                
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CycleTransMorph model for inhale-to-exhale CT prediction.")
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the BASE data directory containing inhale/exhale subfolders')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size PER GPU')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
    parser.add_argument('--reg_lambda', type=float, default=0.02, help='Weight for the regularization (gradient smoothing) loss')
    parser.add_argument('--cycle_lambda', type=float, default=0.1, help='Weight for the cycle consistency loss')
    parser.add_argument('--val_interval', type=int, default=5, help='Epoch interval for saving model checkpoints')
    parser.add_argument('--viz_interval', type=int, default=1, help='Epoch interval for logging image visualizations')
    parser.add_argument('--run_name', type=str, default='ctm_run', help='A name for this training run on Weights & Biases')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a tiny dataset and no WandB logging')
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help='Fraction of the dataset to use for training (e.g., 0.2 for 20%%)')

    args = parser.parse_args()
    train(args)