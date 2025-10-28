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
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import DiceMetric

from models import CycleTransMorph
from losses import NCCLoss, GradientSmoothingLoss, InverseConsistencyLoss
class PairedNiftiDataset(Dataset):
    def __init__(self, file_quadruplets):
        self.file_quadruplets = file_quadruplets

    def __len__(self):
        return len(self.file_quadruplets)

    def __getitem__(self, idx):
        inhale_path, exhale_path, inhale_mask_path, exhale_mask_path = self.file_quadruplets[idx]
        
        inhale_img = np.load(inhale_path).astype(np.float32)
        exhale_img = np.load(exhale_path).astype(np.float32)
        inhale_mask = np.load(inhale_mask_path).astype(np.float32)
        exhale_mask = np.load(exhale_mask_path).astype(np.float32)
        
        inhale_img = np.expand_dims(inhale_img, axis=0)
        exhale_img = np.expand_dims(exhale_img, axis=0)
        inhale_mask = np.expand_dims(inhale_mask, axis=0)
        exhale_mask = np.expand_dims(exhale_mask, axis=0)
        
        return (
            torch.from_numpy(inhale_img),
            torch.from_numpy(exhale_img),
            torch.from_numpy(inhale_mask),
            torch.from_numpy(exhale_mask)
        )

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def get_file_list(data_dir, dataset_fraction):
    inhale_files = sorted(glob.glob(os.path.join(data_dir, 'inhale', '*.npy')))
    verified_files = []
    
    file_iterator = inhale_files
    if int(os.environ.get('RANK', 0)) == 0:
        file_iterator = tqdm(inhale_files, desc="[Rank 0] Verifying file sets")

    for inhale_file in file_iterator:
        base_name_with_ext = os.path.basename(inhale_file)
        base_name = base_name_with_ext.replace('.npy', '')

        exhale_file = os.path.join(data_dir, 'exhale', base_name_with_ext)
        inhale_mask_file = os.path.join(data_dir, 'masks', 'inhale', f"{base_name}_INSP_mask.npy")
        exhale_mask_file = os.path.join(data_dir, 'masks', 'exhale', f"{base_name}_EXP_mask.npy")
        
        if all(os.path.exists(f) for f in [exhale_file, inhale_mask_file, exhale_mask_file]):
            verified_files.append((inhale_file, exhale_file, inhale_mask_file, exhale_mask_file))

    if dataset_fraction < 1.0:
        num_files_to_use = int(len(verified_files) * dataset_fraction)
        random.seed(42)
        verified_files = random.sample(verified_files, num_files_to_use)
    return verified_files

def save_and_log_images(save_dir, epoch, inhale, exhale, warped_inhale):
    moving = inhale.cpu().numpy().squeeze()
    warped = warped_inhale.cpu().numpy().squeeze()
    fixed = exhale.cpu().numpy().squeeze()

    local_img_dir = os.path.join(save_dir, 'images')
    os.makedirs(local_img_dir, exist_ok=True)
    np.save(os.path.join(local_img_dir, f'epoch_{epoch+1}_moving.npy'), moving)
    np.save(os.path.join(local_img_dir, f'epoch_{epoch+1}_warped.npy'), warped)
    np.save(os.path.join(local_img_dir, f'epoch_{epoch+1}_fixed_TARGET.npy'), fixed)

    def normalize_slice(sl):
        sl_min, sl_max = sl.min(), sl.max()
        return (sl - sl_min) / (sl_max - sl_min) if sl_max > sl_min else sl

    sl_x, sl_y, sl_z = [s // 2 for s in moving.shape]

    axial_row = np.concatenate([normalize_slice(moving[sl_z, :, :]), normalize_slice(warped[sl_z, :, :]), normalize_slice(fixed[sl_z, :, :])], axis=1)
    sagittal_row = np.concatenate([normalize_slice(moving[:, :, sl_x]), normalize_slice(warped[:, :, sl_x]), normalize_slice(fixed[:, :, sl_x])], axis=1)
    coronal_row = np.concatenate([normalize_slice(moving[:, sl_y, :]), normalize_slice(warped[:, sl_y, :]), normalize_slice(fixed[:, sl_y, :])], axis=1)
    
    max_width = max(axial_row.shape[1], sagittal_row.shape[1], coronal_row.shape[1])
    
    def pad_row(row, width):
        padding = width - row.shape[1]
        pad_left, pad_right = padding // 2, padding - (padding // 2)
        return np.pad(row, ((0, 0), (pad_left, pad_right)), 'constant')

    grid = np.concatenate([pad_row(axial_row, max_width), pad_row(sagittal_row, max_width), pad_row(coronal_row, max_width)], axis=0)

    wandb.log({"    _images": wandb.Image(grid, caption=f"Epoch {epoch+1}: Views | Cols: Moving, Warped, Fixed")})

def get_jacobian_stats(dvf_batch):
    """
    Calculates Jacobian determinant statistics for a batch of DVFs.
    Expects DVF batch of shape (B, 3, D, H, W)
    """
    # Use .detach() to ensure no gradients are computed
    dvf = dvf_batch.detach().cpu().numpy()
    all_jacs = []
    
    for i in range(dvf.shape[0]):
        # Get single DVF (3, D, H, W)
        dvf_single = dvf[i]
        
        # Get gradients for each component (disp_z, disp_y, disp_x)
        grad_uz = np.gradient(dvf_single[0]) 
        grad_uy = np.gradient(dvf_single[1])
        grad_ux = np.gradient(dvf_single[2])

        # J = I + grad(DVF)
        J_11 = 1 + grad_ux[2]
        J_12 = grad_ux[1]  
        J_13 = grad_ux[0]  
        
        J_21 = grad_uy[2]  
        J_22 = 1 + grad_uy[1]
        J_23 = grad_uy[0]  
        
        J_31 = grad_uz[2]  
        J_32 = grad_uz[1]  
        J_33 = 1 + grad_uz[0]
        
        det = J_11 * (J_22 * J_33 - J_23 * J_32) \
            - J_12 * (J_21 * J_33 - J_23 * J_31) \
            + J_13 * (J_21 * J_32 - J_22 * J_31)
            
        all_jacs.append(det)
    
    all_jacs = np.stack(all_jacs)
    
    stats = {
        "val_jac_mean": np.mean(all_jacs),
        "val_jac_min": np.min(all_jacs),
        "val_jac_pct_neg": np.mean(all_jacs <= 0) * 100,
        "val_jac_pct_compression": np.mean(all_jacs < 1) * 100
    }
    return stats

def train(args):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup_ddp(rank, world_size)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb.init(
            project="exhale_pred", 
            name=os.path.basename(args.save_dir)
        )

    # Let rank 0 prepare and split the file lists
    file_lists = [None, None]
    if rank == 0:
        file_list = get_file_list(args.data_dir, args.dataset_fraction)
        if file_list:
            val_split = int(len(file_list) * 0.2)
            train_files = file_list[val_split:]
            val_files = file_list[:val_split]
            file_lists = [train_files, val_files]
    
    # Broadcast the split lists to all processes
    dist.broadcast_object_list(file_lists, src=0)
    train_files, val_files = file_lists
    
    if not train_files or not val_files:
        if rank == 0: print(f"ERROR: No complete data sets found in {args.data_dir}.")
        cleanup_ddp(); return

    # Add print statements for debugging data distribution
    dist.barrier()
    print(
        f"[Rank {rank}] Post-split file counts: "
        f"Train={len(train_files)}, Val={len(val_files)}"
    )
    dist.barrier()

    train_dataset = PairedNiftiDataset(train_files)
    val_dataset = PairedNiftiDataset(val_files)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, sampler=train_sampler, pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, sampler=val_sampler, pin_memory=True)

    model = CycleTransMorph(img_size=(128, 128, 128)).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    ncc_loss = NCCLoss().to(rank)
    grad_loss = GradientSmoothingLoss().to(rank)
    inv_cons_loss = InverseConsistencyLoss(size=(128, 128, 128)).to(rank)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        ddp_model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]") if rank == 0 else train_dataloader

        for i, (inhale, exhale, inhale_mask, exhale_mask) in enumerate(pbar):
            inhale, exhale = inhale.to(rank), exhale.to(rank)
            inhale_mask, exhale_mask = inhale_mask.to(rank), exhale_mask.to(rank)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                warped_inhale, dvf_i_to_e, svf_i_to_e = ddp_model(inhale, exhale)
                warped_exhale, dvf_e_to_i, svf_e_to_i = ddp_model(exhale, inhale)

                loss_sim_i = ncc_loss(warped_inhale, exhale, mask=exhale_mask)
                loss_sim_e = ncc_loss(warped_exhale, inhale, mask=inhale_mask)
                loss_reg = grad_loss(dvf_i_to_e) + grad_loss(dvf_e_to_i)
                loss_cycle = inv_cons_loss(dvf_i_to_e, dvf_e_to_i)
                total_loss = (loss_sim_i + loss_sim_e) + \
                            args.lambda_cycle * loss_cycle + \
                            args.alpha * loss_reg

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            
            if rank == 0:
                wandb.log({
                    "step_loss": total_loss.item(),
                    "loss_cycle_field": loss_cycle.item(),
                    "loss_similarity_fwd": loss_sim_i.item(),
                    "loss_similarity_bwd": loss_sim_e.item(),
                    "loss_regularization": loss_reg.item(),
                })

        # --- VALIDATION ---
        if (epoch + 1) % args.val_interval == 0:
            dist.barrier()
            ddp_model.eval()
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]") if rank == 0 else val_dataloader
            
            with torch.no_grad():
                for i, (inhale, exhale, inhale_mask, exhale_mask) in enumerate(val_pbar):
                    inhale, exhale = inhale.to(rank), exhale.to(rank)
                    inhale_mask, exhale_mask = inhale_mask.to(rank), exhale_mask.to(rank)

                    with torch.amp.autocast('cuda'):
                        warped_inhale, dvf_i_to_e, _ = ddp_model(inhale, exhale)
                        warped_inhale_mask = ddp_model.module.spatial_transformer(inhale_mask, dvf_i_to_e)

                    # All ranks must participate in metric calculation
                    warped_mask_binary = (warped_inhale_mask > 0.5).float()
                    exhale_mask_binary = (exhale_mask > 0.5).float()
                    dice_metric(y_pred=warped_mask_binary, y=exhale_mask_binary)

                    if rank == 0:
                        if i == 0: # Log images and jacobian for the first validation batch only
                            save_and_log_images(args.save_dir, epoch, inhale[0], exhale[0], warped_inhale[0])
                            jacobian_stats = get_jacobian_stats(dvf_i_to_e)
                            wandb.log(jacobian_stats)

            # All ranks must participate in aggregation and reset.
            dist.barrier()
            val_dice_tensor = dice_metric.aggregate()
            dice_metric.reset()

            if rank == 0:
                val_dice = val_dice_tensor.item()
                wandb.log({"val_dice_score": val_dice})

        if rank == 0 and len(train_dataloader) > 0:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss}")
            wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"--- New best model found at epoch {epoch+1} with loss {best_loss:.4f}. Saving... ---")
                torch.save(ddp_model.module.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))

            if (epoch + 1) % args.val_interval == 0:
                torch.save(ddp_model.module.state_dict(), os.path.join(args.save_dir, f'ctm_epoch_{epoch+1}.pth'))
        
        # Synchronize all processes before starting the next epoch
        dist.barrier()

        scheduler.step()

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lambda_cycle', type=float, default=1.0)
    parser.add_argument('--dataset_fraction', type=float, default=1.0)
    parser.add_argument('--latent_dim', type=int, default=16)
    args = parser.parse_args()
    
    train(args)
