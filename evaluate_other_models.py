#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
使用其他基线模型对 unified_dataset 的 test 集进行评估，并汇总结果。
包含的模型: ResNet50, VGG19, ViT-B/16, Swin V2, YOLOv10 Baseline
"""

import os
import sys
import csv
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# 将各类模型所在路径加入 sys.path
sys.path.insert(0, os.path.abspath("NN/ResNet50"))
sys.path.insert(0, os.path.abspath("NN/VGG19"))
sys.path.insert(0, os.path.abspath("NN/ViT"))
sys.path.insert(0, os.path.abspath("NN/Swin Transformer V2"))
sys.path.insert(0, os.path.abspath("NN/yolov10"))

# 导入各个模型类
from resnet_model import ResNetBaseline
try:
    from vgg_model import VGG19Baseline
except:
    VGG19Baseline = None

from vit_model import ViTBaseline
from swinv2_model import SwinV2Baseline
from baseline_model import YOLOv10BaselineClassifier


def evaluate_model(model_name, model, weights_path, test_data, img_dir, device, img_size=1280):
    print(f"\n{'='*50}")
    print(f"🚀 开始评估: {model_name} (分辨率: {img_size}x{img_size})")
    
    if not os.path.exists(weights_path):
        print(f"❌ 找不到权重文件: {weights_path}")
        return None
        
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return None

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cm = [[0, 0], [0, 0]]
    
    with torch.no_grad():
        for img_name, gt_label in tqdm(test_data, desc=f"Eval {model_name}"):
            img_path = img_dir / img_name
            try:
                raw_img = Image.open(img_path).convert('RGB')
            except:
                continue
                
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            
            cm[gt_label][pred_idx] += 1

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    total = sum(map(sum, cm))
    
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 {model_name} 结果: Acc={accuracy:.4%} | Prec={precision:.4%} | Rec={recall:.4%} | F1={f1:.4f}")
    
    return {
        "Model": model_name,
        "Accuracy": f"{accuracy:.4%}",
        "Precision": f"{precision:.4%}",
        "Recall": f"{recall:.4%}",
        "F1-Score": f"{f1:.4f}"
    }

def main():
    test_csv = Path("data/unified_dataset/test.csv")
    img_dir = Path("data/unified_dataset/images")
    report_path = Path("data/unified_dataset/other_models_evaluation.txt")
    
    if not test_csv.exists():
        print(f"❌ 找不到测试集 CSV: {test_csv}")
        return

    # 1. 读入测试集
    test_data = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                gt = 0 if ("有缺陷" in row[1] or "Defective" in row[1]) else 1
                test_data.append((row[0], gt))
                
    print(f"🔍 从 test.csv 加载了 {len(test_data)} 条测试样本")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")

    # 2. 准备模型列表
    models_to_eval = [
        {
            "name": "ResNet50",
            "model_cls": ResNetBaseline,
            "weights": "NN/ResNet50/resnet_best.pth",
            "img_size": 224  # ResNet 通常使用 224 以节约显存和时间，或用 1280
        },
        {
            "name": "ViT-B/16",
            "model_cls": ViTBaseline,
            "weights": "NN/ViT/vit_best.pth",
            "img_size": 224  # ViT 必须 224 或面临 OOM
        },
        {
            "name": "Swin Transformer V2",
            "model_cls": SwinV2Baseline,
            "weights": "NN/Swin Transformer V2/swinv2_best.pth",
            "img_size": 256  # Swin 标准 256
        },
        {
            "name": "YOLOv10 Baseline",
            "model_cls": lambda **kwargs: YOLOv10BaselineClassifier(model_weight="TubeGuard_GFC_System/weights/yolov10n.pt", **kwargs),
            "weights": "NN/yolov10/raw/baseline_best.pth",
            "img_size": 640  # YOLO 通用评估尺寸 640 或 1280
        }
    ]
    
    if VGG19Baseline:
        models_to_eval.append({
            "name": "VGG19",
            "model_cls": VGG19Baseline,
            "weights": "NN/VGG19/vgg_best.pth",
            "img_size": 224
        })

    # 3. 逐个评估
    results = []
    for cfg in models_to_eval:
        try:
            model = cfg["model_cls"](num_classes=2, pretrained=False)
            res = evaluate_model(
                model_name=cfg["name"],
                model=model,
                weights_path=cfg["weights"],
                test_data=test_data,
                img_dir=img_dir,
                device=device,
                img_size=cfg["img_size"]
            )
            if res:
                results.append(res)
        except Exception as e:
            print(f"❌ {cfg['name']} 评估发生异常: {e}")

    # 4. 汇总
    print("\n" + "="*60)
    print("🏆 多模型在 Unified Dataset 上的对比汇总")
    print("="*60)
    print(f"| {'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} |")
    print("-" * 75)
    for r in results:
        print(f"| {r['Model']:<20} | {r['Accuracy']:<10} | {r['Precision']:<10} | {r['Recall']:<10} | {r['F1-Score']:<10} |")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Result on Unified Dataset (Other Baselines)\n")
        f.write("="*75 + "\n")
        f.write(f"| {'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} |\n")
        f.write("-" * 75 + "\n")
        for r in results:
            f.write(f"| {r['Model']:<20} | {r['Accuracy']:<10} | {r['Precision']:<10} | {r['Recall']:<10} | {r['F1-Score']:<10} |\n")
            
    print(f"\n📝 汇总报告已保存至 {report_path}")

if __name__ == "__main__":
    main()
