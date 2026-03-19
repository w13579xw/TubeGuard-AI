#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
基线消融实验自动训练与评估脚本 (Ablation Study for YOLOv10 Baseline)
按照科研设计，跑纯粹的原始版本 YOLOv10 基线模型 (无 TPH 结构)，
在纯化后的单一缺陷增强数据集上测试其抗干扰能力的短板。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================================
# 紧急环境修复：应对某些 Linux 服务器底层 cuDNN 动态匹配引擎不兼容问题
# "RuntimeError: GET was unable to find an engine to execute this computation"
# =========================================================================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import csv
from tqdm import tqdm

# =========================================================================
# 导入并配置原版基线模型
# =========================================================================
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

try:
    from NN.yolov10.baseline_model import YOLOv10BaselineClassifier
except ImportError as e:
    print(f"❌ 导入失败，请检查项目目录下是否存在 NN/yolov10/baseline_model.py")
    raise e


class CSVImageDataset(Dataset):
    """自定义从 CSV 读取绝对路径的数据集 (支持免拷贝逻辑)"""
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) >= 2:
                    img_path = row[0]
                    lbl = row[1]
                    gt = 0 if ("有缺陷" in lbl or "Defective" in lbl) else 1
                    self.data.append((img_path, gt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path_str, label = self.data[idx]
        img_path = Path(img_path_str)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (640, 640), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def train_and_eval_ablation(ablation_name, num_epochs=50, batch_size=4, patience=15):
    """
    针对给定的消融配置进行训练和验证
    ablation_name: 预期为 dataset_scratch_811, dataset_lighting_811 等
    """
    img_size = 640  # 基线模型通常用 640
    base_dir = Path(f"data/experiments/{ablation_name}")
    
    if not base_dir.exists():
        print(f"⚠️ [跳过] \n目录不存在: {base_dir}\n提示: 请先用 build_experiment_datasets.py 生成针对 '{ablation_name}' 的消融切分集。")
        return

    print(f"\n{'='*60}")
    print(f"🚀 [Baseline Ablation] 当前评估集: {ablation_name}")
    print(f"{'='*60}")

    best_model_path = base_dir / f"yolov10_baseline_best_{ablation_name}.pth"
    checkpoint_path = base_dir / f"last_checkpoint_baseline_{ablation_name}.pth"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {x: CSVImageDataset(base_dir / f"{x}.csv", transform=data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4, pin_memory=True) 
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 硬件设备: {device}")
    
    # === 初始化原版基础 YOLOv10 分类器 ===
    base_weight = 'TubeGuard_GFC_System/weights/yolov10n.pt'
    if not os.path.exists(base_weight):
         base_weight = 'yolov10n.pt' # 本地 fallback
         from ultralytics import YOLO
         YOLO(base_weight) # download if missing

    model = YOLOv10BaselineClassifier(model_weight=base_weight, num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现断点，恢复保存状态: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)

    # 1. ==== 开始训练循环 ====
    for epoch in range(start_epoch, num_epochs):
        print(f'\n[Ablation_Train] Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            pbar = tqdm(dataloaders[phase], desc=f"  ↳ {phase}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            print(f'   ➜ {phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss, 'early_stop_counter': early_stop_counter
        }, checkpoint_path)

        if early_stop_counter >= patience:
             print(f"\n⏹️ 触发早停。")
             break

    # 2. ==== 最终测试与指标统计 (Test) ====
    print("\n🔬 [Testing] 跑分评价中...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    cm = [[0, 0], [0, 0]]
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  ↳ test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                cm[labels[i].item()][preds[i].item()] += 1

    TP, FN = cm[0][0], cm[0][1]
    FP, TN = cm[1][0], cm[1][1]
    total = sum(map(sum, cm))
    
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    results_csv = Path("data/experiments/ablation_baseline_results.csv")
    if not results_csv.exists():
        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["Ablation_Set", "Model", "Accuracy", "Precision", "Recall", "F1_Score"])
            
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([ablation_name, "YOLOv10_Baseline(NO_TPH)", f"{acc:.4%}", f"{prec:.4%}", f"{rec:.4%}", f"{f1:.4f}"])
        
    print(f"\n📊 评测完毕，指标已归档存入 {results_csv.name}")
    print(f" => F1-Score (原版模型对该特化生成的抗性): {f1:.4f}\n")


def main():
    # 本消融实验的设计：对比原版无增幅结构的 YOLOv10 与加入了 TPH 头的模型性能表现。
    # 按照指示，无需再造控制变量的剥离数据，直接使用已有的全量大一统 8:1:1 混合切分集。
    target_sets = [
        "dataset_all_811"
    ]

    print("=== YOLOv10 (Vanilla) 结构消融性能基线训练 ===")
    for ds_name in target_sets:
        train_and_eval_ablation(ds_name, num_epochs=50, batch_size=4, patience=15)
        
    print("\n✅ TPH 模块消融组 (Vanilla Baseline) 训练与评估完毕！")
    print("总览报表存放在：data/experiments/ablation_baseline_results.csv")

if __name__ == "__main__":
    main()
