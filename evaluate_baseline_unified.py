#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
快速验证 YOLOv10 原版基线在增强数据集上的表现
对 data/unified_dataset/test.csv 进行单模型推理测试。
"""

import os
import sys
import csv
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 导入 baseline 模型
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

try:
    from NN.yolov10.baseline_model import YOLOv10BaselineClassifier
except ImportError:
    print(f"❌ 导入失败，请检查项目目录下是否存在 NN/yolov10/baseline_model.py")
    sys.exit(1)

def main():
    test_csv = Path("data/unified_dataset/test.csv")
    img_dir = Path("data/unified_dataset/images")
    
    if not test_csv.exists():
        print(f"❌ 找不到测试集 CSV: {test_csv}")
        return

    # 1. 读入统一增强大测试集
    test_data = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
             if len(row) >= 2:
                 gt = 0 if ("有缺陷" in row[1] or "Defective" in row[1]) else 1
                 test_data.append((row[0], gt))
                 
    print(f"🔍 从 amplified dataset 加载了 {len(test_data)} 条测试样本")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 硬件设备: {device}")

    # 2. 初始化 YOLOv10 Baseline 模型
    model_weight = "TubeGuard_GFC_System/weights/yolov10n.pt"
    if not os.path.exists(model_weight):
         model_weight = "NN/yolov10_tph/yolov10n.pt"
         
    model = YOLOv10BaselineClassifier(model_weight=model_weight, num_classes=2)
    model.to(device)
    
    # 尝试加载训练好的最佳基线权重（必须存在）
    baseline_weights_path = Path("NN/yolov10/processed/baseline_best.pth")
    if not baseline_weights_path.exists():
        baseline_weights_path = Path("NN/yolov10/yolov10_baseline_best.pth") # 另一个可能的位置
        if not baseline_weights_path.exists():
            print(f"⚠️ 找不到预训练好的基线分类头权重！请确认您已经用原版 YOLOv10 跑过这批数据。将用随机权重进行前传（无意义）。")
    else:
        try:
            model.load_state_dict(torch.load(baseline_weights_path, map_location=device))
            print(f"✅ 成功加载基线模型权重：{baseline_weights_path}")
        except Exception as e:
            print(f"❌ 加载基线权重失败: {e}")
            
    model.eval()

    # 3. 准备变换
    img_size = 640  # 基线使用640
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cm = [[0, 0], [0, 0]]
    
    # 4. 前向推理评价
    with torch.no_grad():
        for img_name, gt_label in tqdm(test_data, desc=f"Eval Baseline"):
            img_path = img_dir / img_name
            try:
                raw_img = Image.open(img_path).convert('RGB')
                input_tensor = transform(raw_img).unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                cm[gt_label][pred_idx] += 1
            except:
                continue

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    total = sum(map(sum, cm))
    
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("🏆 无 TPH 的原生 YOLOv10 综合扩充测试集表现")
    print("="*50)
    print(f"Accuracy : {accuracy:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall   : {recall:.4%}")
    print(f"F1-Score : {f1:.4f}")
    print("="*50)
    
    # 作为对比参考
    print("\n[对比] 之前加入 TPH 和强化后模型的参考指标：")
    print("Accuracy : 98.13% \nPrecision: 98.66% \nRecall   : 99.38% \nF1-Score : 0.9902")

if __name__ == "__main__":
    main()
