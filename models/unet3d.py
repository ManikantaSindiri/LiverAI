"""
3D UNet Architecture for Multi-Class Liver & Lesion Segmentation
================================================================
Defines deep neural network models for 3D volumetric medical segmentation
using MONAI and PyTorch.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet, SegResNet


def build_liver_segmentation_model(
    model_type="unet",
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    norm="INSTANCE",
    dropout=0.1,
) -> nn.Module:
    """
    Builds a 3D medical segmentation network.

    Parameters
    ----------
    model_type : str
        'unet' for MONAI 3D UNet or 'segresnet' for SegResNet.
    spatial_dims : int
        3 for 3D volumetric CT scans.
    in_channels : int
        1 (Grayscale preprocessed CT volume).
    out_channels : int
        3 classes (0: Background, 1: Liver, 2: Lesion/Tumor).
    channels : tuple
        Feature map channel progression at each resolution stage.
    strides : tuple
        Downsampling stride for each layer.
    num_res_units : int
        Number of residual units per block.
    norm : str
        Normalization type ('INSTANCE' or 'BATCH').
    dropout : float
        Dropout probability.

    Returns
    -------
    nn.Module
        Initialized PyTorch / MONAI 3D model.
    """
    if model_type.lower() == "segresnet":
        model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=dropout,
        )
    else:
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            dropout=dropout,
        )

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_liver_segmentation_model(model_type="unet").to(device)
    print(f"[OK] Model built successfully on {device}")
    
    # Test forward pass with dummy 3D batch: (Batch, Channel, D, H, W)
    dummy_input = torch.randn(1, 1, 32, 64, 64, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"[OK] Forward pass test: Input {dummy_input.shape} -> Output {output.shape}")
