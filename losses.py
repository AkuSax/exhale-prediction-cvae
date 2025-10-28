import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=None):
        super(NCCLoss, self).__init__()
        self.win = [9, 9, 9] if win is None else win

    def forward(self, y_pred, y_true, mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            # y_true = Fixed Image (I); y_pred = Warped Image (J)
            I = y_true.float()
            J = y_pred.float()

            # I, J = [B, C, D, H, W]
            ndims = len(list(I.size())) - 2
            assert ndims == 3

            win = self.win
            sum_filt = torch.ones([1, 1, *win], device=I.device, dtype=I.dtype)
            pad_no = math.floor(win[0] / 2)
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
            conv_fn = F.conv3d

            I2 = I * I
            J2 = J * J
            IJ = I * J

            I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
            
            cc_map = cross * cross / (I_var * J_var + 1e-5)
            loss_map = -cc_map

            if mask is not None:
                masked_loss_map = loss_map * mask
                mean_loss = torch.sum(masked_loss_map) / (torch.sum(mask) + 1e-5)
            else:
                mean_loss = torch.mean(loss_map)

            return mean_loss

class GradientSmoothingLoss(nn.Module):
    """
    L2 loss on the spatial gradients of a vector field.
    """
    def __init__(self):
        super().__init__()
    def forward(self, vf):
        dy = torch.abs(vf[:, :, 1:, :, :] - vf[:, :, :-1, :, :])
        dx = torch.abs(vf[:, :, :, 1:, :] - vf[:, :, :, :-1, :])
        dz = torch.abs(vf[:, :, :, :, 1:] - vf[:, :, :, :, :-1])
        return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

class InverseConsistencyLoss(nn.Module):
    """
    Calculates the L2 norm of the cycle-consistency error in DVF space.
    Forces dvf_fwd and dvf_bwd to be mathematical inverses.
    """
    def __init__(self, size=(128, 128, 128), mode='bilinear'):
        super().__init__()
        self.mode = mode
        # Create a base identity grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).float() # Shape (3, D, H, W)
        self.register_buffer('grid', grid, persistent=False)
        self.l2_loss = nn.MSELoss()

    def _warp_dvf(self, dvf_to_warp, dvf_to_apply):
        """Warps dvf_to_warp with dvf_to_apply"""
        sampling_grid = self.grid.unsqueeze(0) + dvf_to_apply
        
        shape = dvf_to_apply.shape[2:]
        for i in range(len(shape)):
            sampling_grid[:, i, ...] = 2 * (sampling_grid[:, i, ...] / (shape[i] - 1) - 0.5)

        sampling_grid = sampling_grid.permute(0, 2, 3, 4, 1)
        
        warped_dvf = F.grid_sample(
            dvf_to_warp, 
            sampling_grid, 
            mode=self.mode, 
            padding_mode='border', 
            align_corners=True
        )
        return warped_dvf

    def forward(self, dvf_fwd, dvf_bwd):
        dvf_bwd_warped = self._warp_dvf(dvf_bwd, dvf_fwd)
        cycle_error_fwd = dvf_fwd + dvf_bwd_warped
        
        dvf_fwd_warped = self._warp_dvf(dvf_fwd, dvf_bwd)
        cycle_error_bwd = dvf_bwd + dvf_fwd_warped

        return self.l2_loss(cycle_error_fwd, torch.zeros_like(cycle_error_fwd)) + \
               self.l2_loss(cycle_error_bwd, torch.zeros_like(cycle_error_bwd))