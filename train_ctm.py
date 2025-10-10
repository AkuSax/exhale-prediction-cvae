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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

class NpyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_file = self.file_paths[idx]
        inhale = torch.from_numpy(np.load(base_file.replace('.npy', '_inhale.npy'))).float().unsqueeze(0)
        exhale = torch.from_numpy(np.load(base_file.replace('.npy', '_exhale.npy'))).float().unsqueeze(0)
        return inhale, exhale

def train(args):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        if not args.debug:
            wandb.init(project="exhale_pred", name=args.run_name)
    
    # --- Data Loading (NOW FAST!) ---
    if rank == 0: print(f"--- Loading file list from: {args.file_list} ---")
    with open(args.file_list, 'r') as f:
        complete_files = [line.strip() for line in f.readlines()]
    if rank == 0: print(f"--- Found {len(complete_files)} complete data quadruplets. ---")
    
    # --- Dataset Subsampling ---
    if args.dataset_fraction < 1.0:
        num_samples = int(len(complete_files) * args.dataset_fraction)
        random.shuffle(complete_files)
        complete_files = complete_files[:num_samples]
        if rank == 0: print(f"--- Using a {args.dataset_fraction*100:.1f}% subset of data ({len(complete_files)} scans) ---")

    if args.debug:
        complete_files = complete_files[:4]
        args.epochs = 2
        if rank == 0: print(f"--- Debug mode active. Using {len(complete_files)} scans for {args.epochs} epochs. ---")

    train_dataset = NpyDataset(complete_files)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler)
    
    model = CycleTransMorph().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    sim_loss = NCCLoss().to(rank)
    reg_loss = GradientSmoothingLoss().to(rank)
    cycle_loss = CycleConsistencyLoss().to(rank)
    scaler = GradScaler()
    
    if rank == 0: print("--- Starting Training ---")
    
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
            # --- Visual Inspection ---
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, default='training_files.txt', help='Path to the text file containing training file paths')
    # All other arguments remain the same
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--reg_lambda', type=float, default=0.02)
    parser.add_argument('--cycle_lambda', type=float, default=0.1)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--viz_interval', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='ctm_training_run')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_fraction', type=float, default=1.0)
    args = parser.parse_args()
    train(args)