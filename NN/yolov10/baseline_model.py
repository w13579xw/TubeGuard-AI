# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOv10BaselineClassifier(nn.Module):
    """
    [论文基线模型] Vanilla YOLOv10 Classifier (修复版)
    结构：YOLOv10n Backbone (Layers 0-9) -> GAP -> Linear Head
    修复了直接调用 nn.Sequential 导致的 Concat 错误。
    """

    def __init__(self, model_weight='yolov10n.pt', num_classes=2):
        super().__init__()

        # 1. 加载官方 YOLOv10n 模型
        print(f"Loading official YOLOv10 weights from {model_weight}...")
        try:
            # 加载完整的检测模型对象 (DetectionModel)
            yolo = YOLO(model_weight)
            self.full_model = yolo.model
        except Exception as e:
            print(f"Error loading YOLOv10: {e}")
            # 备用方案
            self.full_model = YOLO('yolov8n.pt').model

        # 2. 提取层列表 (nn.Sequential)
        # 注意：每一层都附带了 .f (from), .i (index) 等属性，我们需要用到它们
        self.layers = self.full_model.model
        self.save = self.full_model.save  # 需要保存输出的层索引列表

        # 3. 自动推断骨干网络输出通道数
        # 我们需要运行一次修复后的前向传播来获取通道数
        dummy_input = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            features = self._forward_backbone(dummy_input)
            self.in_features = features.shape[1]
            print(f"-> Detected Backbone Output Channels: {self.in_features}")

        # 4. 定义分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_features, num_classes)

    def _forward_backbone(self, x):
        """
        手动实现 YOLO 的前向传播逻辑 (仅运行骨干部分)
        """
        y = []  # 用于缓存中间层的输出 (用于跨层连接)

        # 遍历所有层
        for i, m in enumerate(self.layers):
            # [关键] 我们只运行前 10 层 (0-9)
            # YOLOv8/v10 的 Backbone 通常在第 9 层 (SPPF) 结束
            if i > 9:
                break

            # --- YOLO 路由逻辑 ---
            if m.f != -1:  # 如果输入不是来自上一层
                if isinstance(m.f, int):
                    x = y[m.f]  # 取指定层的输出
                else:
                    # 取多个层的输出 (用于 Concat)
                    x = [x if j == -1 else y[j] for j in m.f]

            # 执行当前层
            x = m(x)

            # 缓存输出 (如果后续层需要用到)
            y.append(x if m.i in self.save else None)

        return x

    def forward(self, x):
        # 1. 提取骨干特征 (使用修复后的逻辑)
        x = self._forward_backbone(x)

        # 2. 分类头
        x = self.avgpool(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.fc(x)  # [B, Num_Classes]

        return x


if __name__ == "__main__":
    # 测试代码
    try:
        model = YOLOv10BaselineClassifier(num_classes=2)
        dummy = torch.randn(2, 3, 1280, 1280)
        output = model(dummy)
        print(f"Input: {dummy.shape} -> Output: {output.shape}")
        print("✅ 基线模型修复成功！现在可以正常训练了。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")