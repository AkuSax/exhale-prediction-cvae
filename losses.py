# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Import os for RANK check

class NCCLoss(nn.Module):
    """
    Normalized Cross-Correlation Loss
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, I, J, mask=None):
        # Ensure tensors are 5D
        if I.dim() == 4: I = I.unsqueeze(0)
        if J.dim() == 4: J = J.unsqueeze(0)
        
        I_mean = torch.mean(I, dim=[1, 2, 3, 4], keepdim=True)
        J_mean = torch.mean(J, dim=[1, 2, 3, 4], keepdim=True)

        I_std = torch.std(I, dim=[1, 2, 3, 4], keepdim=True)
        J_std = torch.std(J, dim=[1, 2, 3, 4], keepdim=True)

        I_minus_mean = I - I_mean
        J_minus_mean = J - J_mean

        if mask is not None:
            if mask.dim() == 4: mask = mask.unsqueeze(0)
            I_minus_mean = I_minus_mean * mask
            J_minus_mean = J_minus_mean * mask

        numerator = torch.mean(I_minus_mean * J_minus_mean, dim=[1, 2, 3, 4])
        denominator = I_std * J_std

        # --- DEBUGGING PRINTS for NCC Loss ---
        if 'RANK' in os.environ and os.environ['RANK'] == '0':
            print(f"\n[NCC Debug] I_std min/max: {I_std.min().item():.6f}/{I_std.max().item():.6f}")
            print(f"[NCC Debug] J_std min/max: {J_std.min().item():.6f}/{J_std.max().item():.6f}")
            print(f"[NCC Debug] denominator (I_std * J_std) min/max: {(I_std * J_std).min().item():.6f}/{(I_std * J_std).max().item():.6f}")
            print(f"[NCC Debug] denominator + eps min/max: {(denominator.squeeze() + self.eps).min().item():.6f}/{(denominator.squeeze() + self.eps).max().item():.6f}")
            print(f"[NCC Debug] numerator min/max: {numerator.min().item():.6f}/{numerator.max().item():.6f}")
            print(f"[NCC Debug] NCC term (numerator / (denominator + eps)) min/max: {(numerator / (denominator.squeeze() + self.eps)).min().item():.6f}/{(numerator / (denominator.squeeze() + self.eps)).max().item():.6f}")
            
        return 1 - torch.mean(numerator / (denominator.squeeze() + self.eps))


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
        
        # Sum of squared gradients
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