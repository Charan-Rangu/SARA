"""SegFormer-B0 Model Implementation for SARA SAR Flood Segmentation.

Native PyTorch implementation of SegFormer-B0:
- Encoder: Hierarchical Mix Vision Transformer (MiT-B0) with Overlapped Patch Embeddings,
  Efficient Spatial Reduction Attention (SRA), and Mix-FFN position encoding.
- Decoder: Lightweight All-MLP SegFormer Decoder Head with multiscale feature fusion.
- Parameters: ~3.7 M parameters.
- Optimized for 3-channel SAR inputs (VV_dB, VH_dB, VV_VH_ratio) at 256x256 resolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class OverlapPatchEmbed(nn.Module):
    """Overlapped Patch Embedding layer for hierarchical Transformer stages."""

    def __init__(self, patch_size: int = 7, stride: int = 4, in_channels: int = 3, embed_dim: int = 32):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """Spatial Reduction Attention (SRA) module for SegFormer MiT encoder."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_reshaped = x.transpose(1, 2).reshape(b, c, h, w)
            x_sr = self.sr(x_reshaped).flatten(2).transpose(1, 2)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with 3x3 depthwise convolution position encoding."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, padding=1, groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        # Reshape to 2D for depthwise 3x3 conv
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """SegFormer MiT Transformer Block."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class MixVisionTransformerB0(nn.Module):
    """MiT-B0 Encoder backbone for SegFormer-B0."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        embed_dims = [32, 64, 160, 256]
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        depths = [2, 2, 2, 2]

        # Stage 1
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_channels=in_channels, embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[0], num_heads=num_heads[0], sr_ratio=sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[1], num_heads=num_heads[1], sr_ratio=sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_channels=embed_dims[2], embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([
            TransformerBlock(dim=embed_dims[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3])
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b = x.shape[0]
        features = []

        # Stage 1
        x, h1, w1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h1, w1)
        x = self.norm1(x).transpose(1, 2).reshape(b, -1, h1, w1)
        features.append(x)

        # Stage 2
        x, h2, w2 = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, h2, w2)
        x = self.norm2(x).transpose(1, 2).reshape(b, -1, h2, w2)
        features.append(x)

        # Stage 3
        x, h3, w3 = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, h3, w3)
        x = self.norm3(x).transpose(1, 2).reshape(b, -1, h3, w3)
        features.append(x)

        # Stage 4
        x, h4, w4 = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, h4, w4)
        x = self.norm4(x).transpose(1, 2).reshape(b, -1, h4, w4)
        features.append(x)

        return features


class SegFormerDecoderHead(nn.Module):
    """Lightweight All-MLP Decoder Head for SegFormer."""

    def __init__(self, in_channels: List[int], embedding_dim: int = 256, out_channels: int = 1):
        super().__init__()
        assert len(in_channels) == 4, "SegFormer decoder requires 4 multiscale feature maps"

        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embedding_dim, out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        h1, w1 = c1.shape[2:]

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h1, w1), mode="bilinear", align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h1, w1), mode="bilinear", align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=(h1, w1), mode="bilinear", align_corners=False)

        fused = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        logits = self.classifier(fused)
        return logits


class SegFormerB0(nn.Module):
    """Complete SegFormer-B0 Binary Segmentation Model for 3-band SAR imagery."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, embedding_dim: int = 256):
        super().__init__()
        self.backbone = MixVisionTransformerB0(in_channels=in_channels)
        self.decode_head = SegFormerDecoderHead(
            in_channels=[32, 64, 160, 256],
            embedding_dim=embedding_dim,
            out_channels=out_channels
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[2:]
        features = self.backbone(x)
        logits_low = self.decode_head(features)
        logits = F.interpolate(logits_low, size=(in_h, in_w), mode="bilinear", align_corners=False)
        return logits
