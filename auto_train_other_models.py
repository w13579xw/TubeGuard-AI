#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
其他结构模型自动重训评估脚本 (Auto Train & Eval for Alternative Models)
功能：专门用于在特定的数据集（如 dataset_all_811）上，
将经典的 ResNet50, ViT-B/16, Swin V2 从头微调并直接闭环输出跑分结果，用于与 YOLOv10-TPH 对标。
一键驻留执行: nohup python auto_train_other_models.py > train_others.log 2>&1 &
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import csv
from tqdm import tqdm

# 注册其他模型的路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
for m_path in ["ResNet50", "ViT", "Swin Transformer V2"]:
    p = os.path.join(curr_dir, "NN", m_path)
    if p not in sys.path:
        sys.path.insert(0, p)

from resnet_model import ResNetBaseline
from vit_model import ViTBaseline
from swinv2_model import SwinV2Baseline

class CSVImageDataset(Dataset):
    """跨平台路径强兼容免拷贝数据集读取逻辑"""
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
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


def train_and_test_other(model_name, model_class, img_size, dataset_name, num_epochs=50, batch_size=4, patience=15):
    base_dir = Path("data/experiments") / dataset_name
    
    if not base_dir.exists():
        print(f"⚠️ [跳过] 目录不存在: {base_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 [横向大模型横评] 当前正在重训: {model_name} (分辨率: {img_size})")
    print(f"{'='*60}")

    best_model_path = base_dir / f"{model_name.replace('/', '_').replace(' ', '_')}_best_{dataset_name}.pth"
    checkpoint_path = base_dir / f"last_checkpoint_{model_name.replace('/', '_').replace(' ', '_')}.pth"
    results_csv = Path("data/experiments/experiment_results_summary.csv")

    # [断点防复写检测] 如果这份报表里已经有了它的成绩，就直接跳过
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            content = f.read()
            if dataset_name in content and model_name in content:
                print(f"⏩ [断点防复测] 【{model_name}】在【{dataset_name}】上的成绩已经存在于 CSV 大表中，已自动跨过。")
                return

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {x: CSVImageDataset(base_dir / f"{x}.csv", transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4, pin_memory=True) 
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 硬件设备: {device}")
    
    model = model_class(num_classes=2, pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现存档断点，恢复保存状态: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)

    # 1. ==== 开始多轮深度训练循环 ====
    for epoch in range(start_epoch, num_epochs):
        print(f'\n[{model_name}] Epoch {epoch + 1}/{num_epochs}')
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
             print(f"\n⏹️ 触发早停机制，后续轮次已放弃。")
             break

    # 2. ==== 提取最新干巴的最佳权重，跑测试集并出报表 ====
    print(f"\n🔬 [Testing] {model_name} 训练完毕！正在直接拉起测试流程跑分...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_dataset = CSVImageDataset(base_dir / "test.csv", transform=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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
    
    if not results_csv.exists():
        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["Dataset_Split", "Model", "Accuracy", "Precision", "Recall", "F1_Score"])
            
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([dataset_name, model_name, f"{acc:.4%}", f"{prec:.4%}", f"{rec:.4%}", f"{f1:.4f}"])
        
    print(f"\n📊 {model_name} 评测完毕，指标已落实在表：{results_csv.name}")
    print(f" => 竞品最终 F1-Score: {f1:.4f}\n")


def main():
    target_dataset = "dataset_all_811"
    
    # 按照之前的标准进行统一分辨率对齐
    models_to_train = [
        ("ResNet50", ResNetBaseline, 224),
        ("ViT-B/16", ViTBaseline, 224),
        ("SwinV2", SwinV2Baseline, 256)
    ]
    
    print(f"🚀 将在目标集 {target_dataset} 上对各大经典门派基线模型发起重训并考核...")
    
    for m_name, m_cls, m_size in models_to_train:
        # 相比于最轻量的 YOLOv10-TPH，针对其他笨重的基线模型，由于它们参数极为沉重，
        # 为了防爆显存，建议 batch_size 维持在保守的 4。
        train_and_test_other(m_name, m_cls, m_size, target_dataset, num_epochs=50, batch_size=4, patience=15)
        
    print("\n✅ 所有针对 811 的横向学术竞品模型的重训与对标完成！可打开 experiment_results_summary.csv 欣赏战果。")

if __name__ == "__main__":
    main()
