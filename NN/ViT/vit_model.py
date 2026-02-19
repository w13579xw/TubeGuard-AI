# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models


class ViTBaseline(nn.Module):
    """
    [论文基线模型] Vision Transformer (ViT-B/16)
    使用 ImageNet 预训练权重。
    注意：ViT 对输入分辨率非常敏感，标准输入为 224x224。
    如果在 1280x1280 下运行，Attention 矩阵会极其巨大导致 OOM。
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        print(f"Initializing ViT-B/16 (Pretrained={pretrained})...")

        # 1. 加载 ViT-B/16
        # weights='DEFAULT' 对应 ImageNet-1K 预训练
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # 2. 修改分类头 (Heads)
        # torchvision 的 ViT 输出头封装在 self.backbone.heads 中
        # 原始结构: Sequential(Linear(768 -> 1000))
        in_features = self.backbone.heads.head.in_features

        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # 测试代码
    try:
        model = ViTBaseline(num_classes=2)
        # ViT 通常测试 224 或 384
        dummy = torch.randn(2, 3, 224, 224)
        output = model(dummy)
        print(f"Input: {dummy.shape} -> Output: {output.shape}")
        print("✅ ViT-B/16 模型构建成功！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")