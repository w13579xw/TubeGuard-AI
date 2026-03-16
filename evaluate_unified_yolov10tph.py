#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
使用已训练好的 YOLOv10-TPH 网络在统一数据集的测试集上进行识别并评估
输出指标：Accuracy, Precision, Recall, F1-score 以及 Confusion Matrix
"""

import os
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# 添加 YOLOv10-TPH 代码目录到 Python 路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
yolo_tph_dir = os.path.join(curr_dir, "NN", "yolov10_tph")
sys.path.insert(0, yolo_tph_dir)

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import YOLOv10TPHClassifier

def main():
    # ==== 配置路径 ====
    test_csv = Path("data/unified_dataset/test.csv")
    img_dir = Path("data/unified_dataset/images")
    
    weights_path = Path("TubeGuard_GFC_System/weights/yolov10_tph_best.pth")
    if not weights_path.exists():
        weights_path = Path("NN/yolov10_tph/processed/yolov10_tph_best.pth")
        
    if not weights_path.exists():
        print(f"❌ 找不到权重文件！尝试路径: {weights_path}")
        return
        
    if not test_csv.exists():
        print(f"❌ 找不到测试集 CSV: {test_csv}。请先执行 build_unified_dataset.py。")
        return

    # 1. 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 1280

    print(f"🚀 初始化 YOLOv10-TPH 分类评估 (设备: {device})...")
    model = YOLOv10TPHClassifier(model_weight='TubeGuard_GFC_System/weights/yolov10n.pt', num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 读取测试集数据
    test_data = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                # ground truth label: 0 -> Defective, 1 -> Good
                # 注意 CSV 里是 "[有缺陷]" / "[无缺陷]" 或者是类似于 "Defective"
                label_str = row[1].strip()
                if "有缺陷" in label_str or "Defective" in label_str:
                    gt = 0
                else:
                    gt = 1
                test_data.append((row[0], gt))
                
    print(f"🔍 从 test.csv 加载了 {len(test_data)} 条测试样本")

    # 3. 推理评估
    # 混淆矩阵: cm[true_label][pred_label]
    cm = [[0, 0], [0, 0]]
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for img_name, gt_label in tqdm(test_data, desc="Evaluating"):
            img_path = img_dir / img_name
            try:
                raw_img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"⚠️ 跳过无法读取的文件 {img_path}: {e}")
                continue
            
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            
            cm[gt_label][pred_idx] += 1
            all_preds.append(pred_idx)
            all_targets.append(gt_label)

    # 4. 计算指标
    # label0 = 异常样本(Defective/Positive), label1 = 正常样本(Good/Negative)
    # TP: 真异常 (gt=0, pred=0)
    # FN: 漏报 (gt=0, pred=1) 应该报异常但没报
    # FP: 误报 (gt=1, pred=0) 正常被当成异常
    # TN: 真正常 (gt=1, pred=1)
    
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
    print("📊 YOLOv10-TPH 模型在统一 Test 集合上的评估结果")
    print("="*50)
    print(f"🎯 总测试样本数 : {total}")
    print(f"✅ Accuracy (准确率) : {accuracy:.4%}")
    print(f"🎯 Precision (精确率) : {precision:.4%} (检测出的缺陷有多大概率是对的)")
    print(f"🔍 Recall (召回率)    : {recall:.4%} (实际的缺陷被找出来了多少)")
    print(f"🥇 F1-Score        : {f1:.4f}")
    
    print("\n[混淆矩阵 Confusion Matrix]")
    print("          | Pred Defective(0) | Pred Good(1)")
    print(f"--------------------------------------------")
    print(f"True Def(0)|      {TP:<12} | {FN:<10}  (FN=漏报/假相)")
    print(f"True Good(1)|      {FP:<12} | {TN:<10}  (FP=误报/过杀)")
    print("==================================================")
    
    # 将评估结果保存至文本文件
    report_path = Path("data/unified_dataset/evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("YOLOv10-TPH Evaluation Report\n")
        f.write("==============================\n")
        f.write(f"Total: {total}\n")
        f.write(f"Accuracy: {accuracy:.4%}\n")
        f.write(f"Precision: {precision:.4%}\n")
        f.write(f"Recall: {recall:.4%}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(f"TP: {TP}  FN: {FN}\n")
        f.write(f"FP: {FP}  TN: {TN}\n")
        
    print(f"📝 报告已保存至 {report_path}")

if __name__ == "__main__":
    main()
