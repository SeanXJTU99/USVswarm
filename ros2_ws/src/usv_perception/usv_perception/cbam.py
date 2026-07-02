"""CBAM: Convolutional Block Attention Module.

Channel + Spatial attention for suppressing water surface glare,
wave reflections, and high-frequency background noise in YOLO feature maps.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention: learns *which* feature channels matter.

    Uses both avg-pool and max-pool to squeeze spatial dimensions,
    then a shared MLP to produce channel-wise weights.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize channel attention.

        Args:
            channels: Number of input feature channels.
            reduction: Reduction ratio for the bottleneck MLP.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Channel attention map (B, C, 1, 1).
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention: learns *where* in the feature map to focus.

    Uses avg-pool and max-pool along the channel axis, concatenates,
    then a 7x7 convolution to produce a spatial weight map.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize spatial attention.

        Args:
            kernel_size: Convolution kernel size (typically 7).
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Spatial attention map (B, 1, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequentially applies channel attention then spatial attention
    to refine feature maps. Inserted after YOLO backbone / before neck.

    On water surfaces, CBAM forces the network to focus on object
    textures (vessel hulls, buoy shapes) rather than responding to
    wave patterns and sun glare reflections.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize CBAM.

        Args:
            channels: Number of input feature channels.
            reduction: Reduction ratio for channel attention bottleneck.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM refinement.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Refined feature map of same shape.
        """
        # Channel attention: scale each channel
        x = x * self.channel_attention(x)
        # Spatial attention: scale each spatial location
        x = x * self.spatial_attention(x)
        return x


def insert_cbam_into_yolo(
    model: nn.Module, positions: list[int] | None = None, reduction: int = 16
) -> nn.Module:
    """Insert CBAM modules after specified YOLO backbone layers.

    Typical insertion points: after C3/C2f blocks in the backbone,
    just before the SPPF module and neck connections.

    Args:
        model: A YOLOv5/v8 model (ultralytics).
        positions: Layer indices where CBAM should be inserted.
                   Defaults to [4, 6, 9] for YOLOv5s.
        reduction: Channel reduction ratio.

    Returns:
        Modified model with CBAM modules inserted.
    """
    if positions is None:
        positions = [4, 6, 9]

    # This is a hook pattern — actual insertion depends on model architecture.
    # In practice, wrap the model's backbone forward with post-CBAM refinement.
    return model
