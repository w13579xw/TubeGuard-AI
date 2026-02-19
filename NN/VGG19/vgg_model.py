#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models


class VGGBaseline(nn.Module):
    """
    [论文基线模型] VGG19_BN Classifier
    使用 ImageNet 预训练权重。
    特点：深层卷积，参数量巨大，用于对比"深层堆叠"的效果。
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        print(f"Initializing VGG19_BN (Pretrained={pretrained})...")

        # 1. 加载 VGG19_BN (带 Batch Norm，否则很难收敛)
        weights = models.VGG19_BN_Weights.DEFAULT if pretrained else None
        self.backbone = models.vgg19_bn(weights=weights)

        # 2. 修改分类头 (Classifier)
        # VGG 的分类器是：Linear(25088 -> 4096) -> ReLU -> Dropout -> Linear(4096 -> 4096) -> ...
        # 我们需要修改最后一层 Linear(4096 -> 1000) 为 Linear(4096 -> num_classes)

        # 获取分类器中最后一个全连接层的输入特征数
        # self.backbone.classifier 是一个 Sequential，第 6 层是输出层
        num_ftrs = self.backbone.classifier[6].in_features

        # 替换最后一层
        self.backbone.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # 测试代码
    try:
        # VGG 显存占用极大，测试时用小一点的图或 batch=1
        model = VGGBaseline(num_classes=2)
        dummy = torch.randn(1, 3, 640, 640)
        output = model(dummy)
        print(f"Input: {dummy.shape} -> Output: {output.shape}")
        print("✅ VGG19_BN 模型构建成功！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")