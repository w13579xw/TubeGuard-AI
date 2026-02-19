#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Swin Transformer V2 (Tiny) 基线模型完整训练脚本
特点：
1. 默认输入尺寸 256x256 (适配预训练权重)。
2. 使用 AdamW 优化器。
3. 混淆矩阵颜色为绿色 (Greens)。
"""

import os
import sys
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib

# 配置绘图后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# [关键] 导入 SwinV2 模型
try:
    from swinv2_model import SwinV2Baseline

    print("✅ 成功导入 SwinV2 模型")
except ImportError:
    print("❌ 致命错误: 找不到 swinv2_model.py")
    sys.exit(1)


# =========================================================================
# 1. 训练核心函数
# =========================================================================
def train_swinv2(data_dir, num_epochs=100, batch_size=16, patience=15):
    # Swin V2 Tiny 默认预训练尺寸是 256。
    # 如果你想测试高分辨率优势，可以尝试 512 或 640，但注意 batch_size 要减小。
    img_size = 640
    checkpoint_path = 'swinv2_checkpoint.pth'
    best_model_path = 'swinv2_best.pth'

    print(f"\n[Baseline Config] SwinV2-Tiny | Size: {img_size}x{img_size} | Batch: {batch_size}")

    # --- 数据增强 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),  # 管道对称性
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

    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 初始化 SwinV2 模型 ---
    print("Initializing SwinV2-Tiny...")
    model = SwinV2Baseline(num_classes=2, pretrained=True)
    model = model.to(device)

    # 优化器: Transformer 类模型标配 AdamW
    # 学习率通常比 CNN 小一点，这里给 5e-5 或 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # 学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 断点续训 ---
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现 SwinV2 断点文件，正在恢复...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        print(f"✅ 恢复成功！从第 {start_epoch + 1} 轮继续")

    if start_epoch >= num_epochs:
        return model

    # --- 训练循环 ---
    print("\n🚀 开始 SwinV2 训练...")
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs} [LR: {optimizer.param_groups[0]["lr"]:.6f}]')
            print('-' * 10)

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
                            # 梯度裁剪：防止 Transformer 梯度爆炸
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix({'loss': loss.item()})

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    scheduler.step()
                    if epoch_loss < best_loss:
                        print(f"  ★ SwinV2 Loss Improved ({best_loss:.4f} -> {epoch_loss:.4f}). Saving BEST...")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), best_model_path)
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        print(f"  ⚠️ No improvement. Counter: {early_stop_counter}/{patience}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'early_stop_counter': early_stop_counter
            }, checkpoint_path)

            if early_stop_counter >= patience:
                print(f"\n⏹️ 早停触发 (SwinV2 Early Stopping)。")
                break

    except KeyboardInterrupt:
        print("\n⛔ 训练中断。")

    model.load_state_dict(best_model_wts)
    return model


# =========================================================================
# 2. 完整评估函数
# =========================================================================
def evaluate_comprehensive(model, data_dir, img_size=256):  # 注意与训练尺寸一致
    print("\n" + "=" * 50)
    print(">>> 启动 SwinV2 全面评估...")
    print("=" * 50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    error_dir = 'swinv2_error_images'
    if os.path.exists(error_dir):
        shutil.rmtree(error_dir)
    os.makedirs(error_dir)

    all_preds = []
    all_labels = []
    error_records = []

    class_names = val_dataset.classes
    file_paths = [s[0] for s in val_dataset.samples]
    global_idx = 0

    print("正在推理...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                current_label = labels[i].item()
                current_pred = preds[i].item()
                all_labels.append(current_label)
                all_preds.append(current_pred)

                if current_pred != current_label:
                    current_file_path = file_paths[global_idx]
                    file_name = os.path.basename(current_file_path)
                    true_name = class_names[current_label]
                    pred_name = class_names[current_pred]

                    error_records.append({
                        "文件名": file_name,
                        "真实标签": true_name,
                        "预测标签": pred_name,
                        "原始路径": current_file_path
                    })

                    try:
                        shutil.copy(current_file_path, os.path.join(error_dir, f"Err_{file_name}"))
                    except:
                        pass

                global_idx += 1

    # --- 生成报告 ---

    # (1) CSV
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv('swinv2_report.csv', encoding='utf-8-sig')

    # (2) 混淆矩阵
    label_map = {'defective': '有缺陷', 'good': '无缺陷'}
    class_names_cn = [label_map.get(name, name) for name in class_names]

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    # [关键] 使用绿色调
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_cn, yticklabels=class_names_cn)
    plt.title('混淆矩阵 - SwinV2', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.savefig('swinv2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 混淆矩阵已保存: swinv2_confusion_matrix.png")

    if error_records:
        df_errors = pd.DataFrame(error_records)
        df_errors.to_csv('swinv2_error_analysis.csv', index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 发现 {len(error_records)} 个错误样本。")
    else:
        print("\n🎉 SwinV2 模型验证集全对！")


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    DATA_DIR = '../raw_data'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    if os.path.exists(DATA_DIR):
        # 训练 (Swin 也比较大，batch_size 16 左右适中)
        trained_model = train_swinv2(DATA_DIR, num_epochs=150, batch_size=16, patience=20)

        # 评估
        evaluate_comprehensive(trained_model, DATA_DIR, img_size=640)
    else:
        print(f"❌ 找不到数据集: {DATA_DIR}")