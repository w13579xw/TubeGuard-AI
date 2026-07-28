"""Comparable image-level classifiers built from YOLO and DETR backbones."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


YOLO_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolov10n": "yolov10n.pt",
    "yolo11n": "yolo11n.pt",
}
DETR_WEIGHTS = {
    "detr_r50": "facebook/detr-resnet-50",
    "deformable_detr_r50": "SenseTime/deformable-detr",
}


class YOLOBackboneClassifier(nn.Module):
    """Use the official YOLO backbone with GAP and a common classification head."""

    def __init__(self, weights: str, num_classes: int = 2):
        super().__init__()
        detector = YOLO(weights).model
        self.layers = detector.model
        self.save = detector.save
        self.backbone_end = self._find_backbone_end()
        with torch.no_grad():
            feature = self._forward_backbone(torch.zeros(1, 3, 640, 640))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature.shape[1], num_classes)

    def _find_backbone_end(self) -> int:
        # The first layer with a multi-source route marks the feature pyramid neck.
        for index, layer in enumerate(self.layers):
            source = getattr(layer, "f", -1)
            if index >= 8 and isinstance(source, (list, tuple)):
                return index - 1
        return min(9, len(self.layers) - 1)

    def _forward_backbone(self, x):
        outputs = []
        for index, layer in enumerate(self.layers):
            if index > self.backbone_end:
                break
            source = getattr(layer, "f", -1)
            if source != -1:
                if isinstance(source, int):
                    x = outputs[source]
                else:
                    x = [x if item == -1 else outputs[item] for item in source]
            x = layer(x)
            layer_index = getattr(layer, "i", index)
            outputs.append(x if layer_index in self.save else None)
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise RuntimeError("YOLO backbone did not produce a BCHW feature tensor")
        return x

    def forward(self, images):
        features = self._forward_backbone(images)
        return self.classifier(self.pool(features).flatten(1))


class DETRBackboneClassifier(nn.Module):
    """Fine-tune a pretrained DETR encoder for image-level classification."""

    def __init__(self, checkpoint: str, num_classes: int = 2):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DETR comparisons require `pip install transformers`."
            ) from exc
        self.backbone = AutoModel.from_pretrained(checkpoint)
        hidden_size = getattr(self.backbone.config, "d_model", 256)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        tokens = getattr(outputs, "encoder_last_hidden_state", None)
        if tokens is None:
            tokens = outputs.last_hidden_state
        return self.classifier(tokens.mean(dim=1))


def build_model(name: str, num_classes: int = 2, weights: str | None = None):
    if name in YOLO_WEIGHTS:
        return YOLOBackboneClassifier(weights or YOLO_WEIGHTS[name], num_classes)
    if name == "yolov10_tph":
        from NN.yolov10_tph.model import YOLOv10TPHClassifier

        return YOLOv10TPHClassifier(
            model_weight=weights or YOLO_WEIGHTS["yolov10n"],
            num_classes=num_classes,
        )
    if name in DETR_WEIGHTS:
        return DETRBackboneClassifier(weights or DETR_WEIGHTS[name], num_classes)
    choices = sorted([*YOLO_WEIGHTS, "yolov10_tph", *DETR_WEIGHTS])
    raise ValueError(f"Unknown model {name!r}. Choices: {', '.join(choices)}")

