#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
全自动实验执行与汇总脚本 (消融实验 & 数据集划分比例差异实验)
1. 自动调用 build_experiment_datasets.py 构建对应的高级验证子集
2. 自动在建立的子集上运行 evaluate_other_models.py (或其他主评测函数)
3. 收集并记录实验指标差异，输出最终论文实验数据表。
"""

import os
import sys
import csv
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"\n⚙️ 运行命令: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {cmd}")
        return False
    return True

def run_experiment_evaluation(dataset_name, output_csv):
    """
    借用并修改原有的模型评估逻辑，只针对单一模型 YOLOv10-TPH，
    或者直接使用已经写好的评估函数。
    这里为了简单高效，复用 `evaluate_other_models.py` 内部逻辑。
    """
    # 将模型代码目录加入路径
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(curr_dir, "NN", "yolov10_tph"))
    
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from model import YOLOv10TPHClassifier
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = Path("TubeGuard_GFC_System/weights/yolov10_tph_best.pth")
    if not weights_path.exists():
        weights_path = Path("NN/yolov10_tph/processed/yolov10_tph_best.pth")
        
    model = YOLOv10TPHClassifier(model_weight='TubeGuard_GFC_System/weights/yolov10n.pt', num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((1280, 1280)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_csv = Path(f"data/experiments/{dataset_name}/test.csv")
    
    test_data = [] # [(img_path, gt_label)]
    if test_csv.exists():
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    gt = 0 if ("有缺陷" in row[1] or "Defective" in row[1]) else 1
                    test_data.append((row[0], gt))
                    
    cm = [[0, 0], [0, 0]]
    with torch.no_grad():
        for img_path_str, gt_label in test_data:
            img_path = Path(img_path_str)
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
    
    # 写入实验结果
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name, f"{accuracy:.4%}", f"{precision:.4%}", f"{recall:.4%}", f"{f1:.4f}"])
        

def main():
    results_csv = Path("data/experiments/experiment_results_summary.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment_Name", "Accuracy", "Precision", "Recall", "F1_Score"])
    
    # 实验组定义
    # 测试集比例的变体实验 (只留全量数据，更改比例)
    split_experiments = [
        ("dataset_all_811", "8:1:1"),
        ("dataset_all_622", "6:2:2"),
        ("dataset_all_532", "5:3:2")
    ]
    
    for split_name, ratio in split_experiments:
        print(f"\n{'='*50}\n🧪 [分割对比] 构建并评估 {ratio} 比例 \n{'='*50}")
        # 这里用 python.exe 确保虚拟环境无误，因为这被 subprocess 执行
        run_cmd(f"python build_experiment_datasets.py --ratios {ratio} --ablation_mode all --output_name {split_name}")
        run_experiment_evaluation(split_name, results_csv)
        
    print(f"\n🎉 所有自动化数据集切分差异与消融评估已完成！")
    print(f"📄 查看汇总结果: {results_csv}")

if __name__ == "__main__":
    main()
