import os
import torch
import torch.nn as nn

class NCCLoss(nn.Module):
    """
    Normalized Cross-Correlation Loss
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, I, J, mask=None):
        # I = warped_inhale, J = exhale (target)
        if I.dim() == 4: I = I.unsqueeze(0)
        if J.dim() == 4: J = J.unsqueeze(0)
        
        if mask is not None:
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            mask = mask.expand_as(I)
            
            num_voxels = torch.sum(mask, dim=[1, 2, 3, 4], keepdim=True) + self.eps
            
            I_mean = torch.sum(I * mask, dim=[1, 2, 3, 4], keepdim=True) / num_voxels
            J_mean = torch.sum(J * mask, dim=[1, 2, 3, 4], keepdim=True) / num_voxels
            
            I_std = torch.sqrt(torch.sum(torch.pow(I - I_mean, 2) * mask, dim=[1, 2, 3, 4], keepdim=True) / num_voxels)
            J_std = torch.sqrt(torch.sum(torch.pow(J - J_mean, 2) * mask, dim=[1, 2, 3, 4], keepdim=True) / num_voxels)
            
            I_minus_mean = (I - I_mean) * mask
            J_minus_mean = (J - J_mean) * mask
            
            numerator = torch.sum(I_minus_mean * J_minus_mean, dim=[1, 2, 3, 4]) / num_voxels.squeeze()
            denom = I_std * J_std

        else:
            I_mean = torch.mean(I, dim=[1, 2, 3, 4], keepdim=True)
            J_mean = torch.mean(J, dim=[1, 2, 3, 4], keepdim=True)
            I_std = torch.std(I, dim=[1, 2, 3, 4], keepdim=True)
            J_std = torch.std(J, dim=[1, 2, 3, 4], keepdim=True)
            
            I_minus_mean = I - I_mean
            J_minus_mean = J - J_mean
            
            numerator = torch.mean(I_minus_mean * J_minus_mean, dim=[1, 2, 3, 4])
            denom = I_std * J_std
        
        ncc = numerator / (denom.squeeze() + self.eps)
        return 1 - torch.mean(ncc)
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

class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss using L1 norm.
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    def forward(self, original_image, cycle_reconstructed_image):
        return self.l1_loss(original_image, cycle_reconstructed_image)