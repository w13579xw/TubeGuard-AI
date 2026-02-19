#!/usr/bin/env python
# -*- coding:utf-8 -*-


import sys
import os
import torch
import torch.nn as nn
from ultralytics import YOLO


# =========================================================================
# 1. 核心模块: TransformerBlock
# =========================================================================
class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads=4, num_layers=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.linear_emb = nn.Linear(c2, c2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=c2, nhead=num_heads,
                                                   dim_feedforward=c2 * 2,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x


# =========================================================================
# 2. 模型定义: YOLOv10TPHClassifier (严格版)
# =========================================================================
class YOLOv10TPHClassifier(nn.Module):
    def __init__(self, model_weight='yolov10n.pt', num_classes=2):
        super().__init__()

        # --- [改进1] 严格检查权重文件 ---
        # 如果不是 .pt 结尾（可能是想从头训练），或者是 .pt 但文件不存在
        if str(model_weight).endswith('.pt') and not os.path.exists(model_weight):
            print(f"\n❌ [致命错误] 找不到 YOLOv10 预训练权重: {model_weight}")
            print("请下载权重文件 (yolov10n.pt) 并放到项目根目录下。")
            print("程序已强制退出，避免使用随机权重进行无效训练。")
            sys.exit(1)  # 直接退出程序

        print(f"Loading YOLOv10 Backbone from {model_weight}...")

        # 加载官方模型
        try:
            full_model = YOLO(model_weight)
            # 截取 Backbone (前9层)
            self.features = nn.Sequential(*list(full_model.model.model.children())[:9])
        except Exception as e:
            print(f"\n❌ [加载失败] YOLO 模型加载出错: {e}")
            sys.exit(1)

        # --- [改进2] 在初始化时直接计算通道数 (Static Init) ---
        # 我们在这里内部跑一次，外部就不用跑了
        with torch.no_grad():
            # 创建一个微型假数据 (尺寸 640 即可，通道数必须是 3)
            # 这一步非常快，几乎不耗时
            dummy = torch.zeros(1, 3, 640, 640)
            features_out = self.features(dummy)
            c_out = features_out.shape[1]  # 获取输出通道数 (例如 256)
            print(f"-> Auto-detected Backbone Output Channels: {c_out}")

        # --- [改进3] 立即定义层，不再设置为 None ---
        self.tph = TransformerBlock(c1=c_out, c2=c_out, num_heads=4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 定义分类头
        self.classifier = nn.Linear(c_out, num_classes)

        # 初始化权重 (Transformer 和 Linear 最好做个初始化)
        self._init_weights()

    def _init_weights(self):
        """对新增的层进行 Xavier 初始化，加快收敛"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Backbone
        x = self.features(x)

        # 2. TPH
        x = self.tph(x)

        # 3. Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out