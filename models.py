# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

class ScalingAndSquaring(nn.Module):
    """
    Layer to compute the exponential of a dense vector field via scaling and squaring.
    The number of squaring steps is determined by the maximum of the absolute values of
    the vector field.
    """
    def __init__(self, scaling_steps=7):
        super().__init__()
        self.scaling_steps = scaling_steps

    def forward(self, v):
        """
        v: Stationary velocity field of shape (B, 3, D, H, W)
        """
        # Scale the velocity field
        v = v / (2**self.scaling_steps)
        
        # Repeatedly square the displacement field
        for _ in range(self.scaling_steps):
            v = v + self.compose(v, v)
            
        return v

    @staticmethod
    def compose(v1, v2):
        """
        Compose two vector fields.
        Warps the vector field v2 according to the displacement field v1.
        v_new(x) = v2(x + v1(x))
        """
        size = v1.shape[2:]

        # 1. Create an identity grid (gives pixel coordinates)
        vectors = [torch.arange(0, s, device=v1.device, dtype=v1.dtype) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # Shape: (3, D, H, W)
        grid = grid.unsqueeze(0)   # Shape: (1, 3, D, H, W)

        # 2. Add the displacement field to the identity grid to get the sampling locations
        # The identity grid will be broadcast to match the batch size of v1
        sampling_grid = grid + v1

        # 3. Normalize the sampling grid to be in the required [-1, 1] range
        for i, s in enumerate(size):
            sampling_grid[:, i, ...] = 2 * (sampling_grid[:, i, ...] / (s - 1) - 0.5)

        # 4. Permute the grid to the format expected by F.grid_sample
        # (B, 3, D, H, W) -> (B, D, H, W, 3)
        sampling_grid = sampling_grid.permute(0, 2, 3, 4, 1)

        # 5. Sample (warp) v2 at the new grid locations
        v2_warped = F.grid_sample(
            v2,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False  # Must be False for displacement fields
        )
        
        return v2_warped

class CycleTransMorph(nn.Module):
    """
    CycleTransMorph model for diffeomorphic image registration.

    Combines a Swin Transformer (SwinUNETR) backbone with a scaling and squaring
    layer to produce a diffeomorphic transformation. The model is designed to be
    used in a Siamese fashion to enforce cycle consistency.
    """
    def __init__(self, img_size=(128, 128, 128), in_channels=2, out_channels=3, feature_size=48):
        super().__init__()
        
        self.img_size = img_size

        # 1. Swin Transformer Backbone (from MONAI)
        self.transformer_backbone = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=16,
            feature_size=feature_size,
            use_checkpoint=True,
        )

        # 2. Velocity Field Prediction Head
        self.svf_head = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)

        # Initialize weights of the prediction head to be small
        self.svf_head.weight = nn.Parameter(torch.randn(self.svf_head.weight.shape) * 1e-5)
        self.svf_head.bias = nn.Parameter(torch.zeros(self.svf_head.bias.shape))

        # 3. Diffeomorphic Integration Layer
        self.diffeomorphic_layer = ScalingAndSquaring(scaling_steps=7)

        # 4. Spatial Transformer for Warping
        self.spatial_transformer = SpatialTransformer(size=self.img_size)

    def forward(self, moving, fixed):
        # Concatenate moving and fixed images along the channel dimension
        x = torch.cat([moving, fixed], dim=1)
        
        # Get hierarchical features from the transformer
        transformer_features = self.transformer_backbone(x)[0]
        
        # Predict the stationary velocity field (SVF)
        svf = self.svf_head(transformer_features)
        
        # Compute the dense deformation field (DVF)
        dvf = self.diffeomorphic_layer(svf)
        
        # Warp the moving image
        warped_image = self.spatial_transformer(moving, dvf)
        
        return warped_image, dvf, svf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)