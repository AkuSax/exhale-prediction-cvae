import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
import os

class ScalingAndSquaring(nn.Module):
    """
    Layer to compute the exponential of a dense vector field via scaling and squaring.
    This version is robust to batch sizes for DDP training.
    """
    def __init__(self, size, scaling_steps=7):
        super().__init__()
        self.scaling_steps = scaling_steps
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, v):
        v = v / (2**self.scaling_steps)
        for _ in range(self.scaling_steps):
            v = v + self.compose(v, v)
        return v

    def compose(self, v1, v2):
        grid = self.grid.unsqueeze(0).repeat(v1.shape[0], 1, 1, 1, 1).to(v1.device)
        sampling_grid = grid + v1
        size = v1.shape[2:]
        for i, s in enumerate(size):
            sampling_grid[:, i, ...] = 2 * (sampling_grid[:, i, ...] / (s - 1) - 0.5)
        
        sampling_grid = sampling_grid.permute(0, 2, 3, 4, 1)
        v2_warped = F.grid_sample(
            v2, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        return v2_warped
    
class CycleTransMorph(nn.Module):
    """
    CycleTransMorph model for diffeomorphic image registration.
    """
    def __init__(self, img_size=(128, 128, 128), in_channels=2, out_channels=3, feature_size=48):
        super().__init__()
        self.img_size = img_size
        self.transformer_backbone = SwinUNETR(
            spatial_dims=3, in_channels=in_channels, out_channels=16,
            feature_size=feature_size, use_checkpoint=True,
        )
        self.svf_head = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)
        self.svf_head.weight = nn.Parameter(torch.randn(self.svf_head.weight.shape) * 1e-5)
        self.svf_head.bias = nn.Parameter(torch.zeros(self.svf_head.bias.shape))
        self.diffeomorphic_layer = ScalingAndSquaring(size=self.img_size, scaling_steps=7)
        self.spatial_transformer = SpatialTransformer(size=self.img_size)

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        transformer_output = self.transformer_backbone(x)

        if isinstance(transformer_output, (list, tuple)):
            transformer_features = transformer_output[0]
        else:
            transformer_features = transformer_output

        # FIX: Permute the backbone output to correct swapped (Width, Depth) dimensions.
        transformer_features = transformer_features.permute(0, 1, 2, 4, 3)

        svf = self.svf_head(transformer_features)
        dvf = self.diffeomorphic_layer(svf)
        warped_image = self.spatial_transformer(moving, dvf)
        return warped_image, dvf, svf
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer, robust to DDP.
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        grid = self.grid.unsqueeze(0).repeat(flow.shape[0], 1, 1, 1, 1).to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)