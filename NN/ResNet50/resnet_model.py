#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models


class ResNetBaseline(nn.Module):
    """
    [论文基线模型] ResNet50 Classifier
    使用 ImageNet 预训练权重，修改最后一层全连接层以适配二分类。
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        print(f"Initializing ResNet50 (Pretrained={pretrained})...")

        # 1. 加载 ResNet50
        # 'DEFAULT' 会自动下载最新的 ImageNet 权重
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # 2. 修改全连接层 (FC Layer)
        # ResNet50 的 fc 输入特征数通常是 2048
        num_ftrs = self.backbone.fc.in_features

        # 替换为适应当前任务的分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 以此增加一点抗过拟合能力
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # ResNet 的 forward 已经包含了池化和展平，直接调用即可
        return self.backbone(x)


if __name__ == "__main__":
    # 测试代码
    try:
        model = ResNetBaseline(num_classes=2)
        # 模拟一张 1280 的大图 (注意显存)
        dummy = torch.randn(2, 3, 1280, 1280)
        output = model(dummy)
        print(f"Input: {dummy.shape} -> Output: {output.shape}")
        print("✅ ResNet50 模型构建成功！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")