#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
实验数据集全自动训练流 (Automated Training Pipeline for Splits)
功能：针对 build_experiment_datasets 划分好的 CSV 子集（811, 622, 532），
自动轮询启动 YOLOv10-TPH 的从头训练，并为每个比例的数据集保存专属的最佳权重。

使用方法:
直接在服务器上后台运行此脚本:
nohup python auto_train_splits.py > train_splits.log 2>&1 &
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

# 加载模型路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curr_dir, "NN", "yolov10_tph"))

from model import YOLOv10TPHClassifier

class CSVImageDataset(Dataset):
    """自定义从 CSV 读取绝对路径的数据集 (支持免拷贝逻辑)"""
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform
        self.classes = ['Defective', 'Good']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
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
        except Exception as e:
            # 极少数坏图的回退保护
            image = Image.new('RGB', (1280, 1280), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def train_split(dataset_name, num_epochs=50, batch_size=4, patience=15):
    img_size = 1280
    base_dir = Path("data/experiments") / dataset_name
    best_model_path = base_dir / f"yolov10_tph_best_{dataset_name}.pth"
    checkpoint_path = base_dir / f"last_checkpoint_{dataset_name}.pth"
    
    if not base_dir.exists():
        print(f"⚠️ 跳过 {dataset_name}，目录不存在。")
        return

    print(f"\n{'='*50}")
    print(f"🚀 [Training] 当前数据集切片: {dataset_name}")
    print(f"{'='*50}")

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

    datasets = {
        'train': CSVImageDataset(base_dir / "train.csv", transform=data_transforms['train']),
        'val': CSVImageDataset(base_dir / "val.csv", transform=data_transforms['val'])
    }
    
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4, pin_memory=True) 
        for x in ['train', 'val']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 硬件设备: {device}")
    
    # 初始化模型，使用预训练 yolov10n 作为主干基座
    base_weight = 'TubeGuard_GFC_System/weights/yolov10n.pt'
    if not os.path.exists(base_weight):
        base_weight = 'NN/yolov10_tph/yolov10n.pt'
        
    model = YOLOv10TPHClassifier(model_weight=base_weight, num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0

    # 支持断点续训，即便服务器被抢占也能接着跑
    if os.path.exists(checkpoint_path):
        print(f"🔄 发现断点，恢复保存状态: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)

    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            print(f'➜ {phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_path)
                    early_stop_counter = 0
                    print(f"  🌟 验证集 Loss 改善，保存新的最佳模型 -> {best_model_path.name}")
                else:
                    early_stop_counter += 1
                    print(f"  ⚠️ 验证集 Loss 未改善 (早停计数: {early_stop_counter}/{patience})")

        # 每轮结束都记录断点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)

        if early_stop_counter >= patience:
            print(f"\n⏹️ 触发早停机制 (连续 {patience} 轮无改善)。")
            break
            
    print(f"\n🎉 目标集 {dataset_name} 的独立重训已完成！")
    print(f"💾 最佳权重存放于: {best_model_path}\n")

def main():
    splits = ["dataset_all_811", "dataset_all_622", "dataset_all_532"]
    
    print("🚀 将依序在这几套数据切分集上自动执行完整的全参数量模型训练...")
    for sp in splits:
        # 默认执行 50 个 Epoch，带断点续训练机制
        train_split(sp, num_epochs=50, batch_size=4, patience=10)
        
    print("\n✅ 所有切割集的独立训练任务均已执行完毕！你可以拿它们各自最佳的权重跑评价了。")

if __name__ == "__main__":
    main()
