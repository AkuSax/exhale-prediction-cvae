# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NCCLoss(nn.Module):
    """
    Normalized Cross-Correlation Loss
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, I, J, mask=None):
        I_mean = torch.mean(I, dim=[1, 2, 3, 4], keepdim=True)
        J_mean = torch.mean(J, dim=[1, 2, 3, 4], keepdim=True)

        I_std = torch.std(I, dim=[1, 2, 3, 4], keepdim=True)
        J_std = torch.std(J, dim=[1, 2, 3, 4], keepdim=True)

        I_minus_mean = I - I_mean
        J_minus_mean = J - J_mean

        if mask is not None:
            I_minus_mean = I_minus_mean * mask
            J_minus_mean = J_minus_mean * mask

        numerator = torch.mean(I_minus_mean * J_minus_mean, dim=[1, 2, 3, 4])
        denominator = I_std * J_std

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