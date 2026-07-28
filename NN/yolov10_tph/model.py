#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv10-TPH image-level classifier.

The revised TPH is not a plain Transformer encoder appended to the final CNN
feature map. It explicitly combines:

1. an 80 x 80 detail feature from the penultimate YOLOv10n stage;
2. a 40 x 40 semantic feature from the final backbone stage;
3. high-frequency residual enhancement on the detail feature;
4. global token interaction on the semantic feature; and
5. learnable cross-scale gated fusion before classification.

This keeps the change lightweight while preserving finer spatial evidence that
would otherwise be available only before the final downsampling operation.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class HighFrequencyDetailBranch(nn.Module):
    """Extract and downsample local high-frequency evidence from the 80 x 80 map."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.detail_filter = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # A parameter-free local low-pass estimate makes the residual explicitly
        # sensitive to fine cracks, edges, and reflection boundaries.
        low_frequency = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_frequency = x - low_frequency
        enhanced_detail = x + self.detail_filter(high_frequency)
        return self.downsample(enhanced_detail)


class DetailPreservingTPH(nn.Module):
    """Dual-scale Transformer Prediction Head with gated local-global fusion."""

    def __init__(
        self,
        detail_channels,
        semantic_channels,
        embed_dim=None,
        num_heads=4,
        num_layers=1,
        use_ffn=True,
        dropout=0.1,
    ):
        super().__init__()
        embed_dim = embed_dim or semantic_channels
        self.embed_dim = embed_dim
        self.use_ffn = use_ffn

        self.detail_branch = HighFrequencyDetailBranch(
            detail_channels, embed_dim
        )
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        if use_ffn:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.global_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        else:
            self.global_attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.global_norm = nn.LayerNorm(embed_dim)

        gate_channels = max(embed_dim // 4, 16)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, gate_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.local_refinement = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.output_activation = nn.SiLU(inplace=True)

    def forward(self, detail_feature, semantic_feature):
        detail_map = self.detail_branch(detail_feature)
        semantic_map = self.semantic_projection(semantic_feature)

        batch, channels, height, width = semantic_map.shape
        tokens = semantic_map.flatten(2).transpose(1, 2)

        if self.use_ffn:
            global_tokens = self.global_encoder(tokens)
        else:
            attended, _ = self.global_attention(tokens, tokens, tokens)
            global_tokens = self.global_norm(tokens + attended)

        global_map = global_tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        if detail_map.shape[-2:] != global_map.shape[-2:]:
            detail_map = F.interpolate(
                detail_map,
                size=global_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gate = self.fusion_gate(torch.cat([detail_map, global_map], dim=1))
        fused = global_map + gate * detail_map
        return self.output_activation(fused + self.local_refinement(fused))


class YOLOv10TPHClassifier(nn.Module):
    """YOLOv10n classifier with a detail-preserving Transformer head."""

    DETAIL_LAYER_INDEX = 6
    SEMANTIC_LAYER_INDEX = 8

    def __init__(
        self,
        model_weight="yolov10n.pt",
        num_classes=2,
        num_heads=4,
        num_layers=1,
        use_ffn=True,
    ):
        super().__init__()

        if str(model_weight).endswith(".pt") and not os.path.exists(model_weight):
            model_weight = os.path.basename(model_weight)

        try:
            full_model = YOLO(model_weight)
            self.features = nn.ModuleList(
                list(full_model.model.model.children())[
                    : self.SEMANTIC_LAYER_INDEX + 1
                ]
            )
        except Exception as exc:
            print(f"Failed to load the YOLOv10 backbone: {exc}")
            sys.exit(1)

        with torch.no_grad():
            detail_feature, semantic_feature = self._forward_backbone(
                torch.zeros(1, 3, 640, 640)
            )

        detail_channels = detail_feature.shape[1]
        semantic_channels = semantic_feature.shape[1]

        self.tph = DetailPreservingTPH(
            detail_channels=detail_channels,
            semantic_channels=semantic_channels,
            embed_dim=semantic_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            use_ffn=use_ffn,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(semantic_channels, num_classes)
        self._init_new_weights()

    def _forward_backbone(self, x):
        detail_feature = None
        semantic_feature = None

        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == self.DETAIL_LAYER_INDEX:
                detail_feature = x
            if index == self.SEMANTIC_LAYER_INDEX:
                semantic_feature = x

        if detail_feature is None or semantic_feature is None:
            raise RuntimeError(
                "Could not extract the configured detail and semantic features "
                "from the YOLOv10 backbone."
            )
        return detail_feature, semantic_feature

    def _init_new_weights(self):
        modules = [
            self.tph.detail_branch,
            self.tph.semantic_projection,
            self.tph.fusion_gate,
            self.tph.local_refinement,
            self.classifier,
        ]
        for root in modules:
            for module in root.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        detail_feature, semantic_feature = self._forward_backbone(x)
        x = self.tph(detail_feature, semantic_feature)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
