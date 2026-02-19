#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchvision import models


class SwinV2Baseline(nn.Module):
    """
    [论文基线模型] Swin Transformer V2 (Tiny)
    使用 ImageNet 预训练权重。
    Swin V2 相比 V1，更适合高分辨率图像迁移，且训练更稳定。
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        print(f"Initializing Swin Transformer V2 Tiny (Pretrained={pretrained})...")

        # 1. 尝试加载 Swin V2 Tiny
        try:
            # Swin V2 需要 torchvision >= 0.15
            weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_v2_t(weights=weights)
        except AttributeError:
            print("⚠️ 警告: 你的 torchvision 版本过低，不支持 Swin V2。")
            print("->正在降级使用 Swin V1 (swin_t)...")
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_t(weights=weights)
        except Exception as e:
            print(f"❌ 模型加载错误: {e}")
            raise e

        # 2. 修改分类头 (Head)
        # Swin 的分类头通常也是一个 Linear 层
        # 获取输入特征维度 (Tiny 版本通常是 768)
        num_ftrs = self.backbone.head.in_features

        # 替换为二分类头
        self.backbone.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # 测试代码
    try:
        model = SwinV2Baseline(num_classes=2)
        # Swin V2 标准输入通常是 256x256
        dummy = torch.randn(2, 3, 256, 256)
        output = model(dummy)
        print(f"Input: {dummy.shape} -> Output: {output.shape}")
        print("✅ Swin V2 Tiny 模型构建成功！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")