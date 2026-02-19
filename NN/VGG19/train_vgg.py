#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
VGG19_BN 基线模型完整训练脚本
特点：
1. 显存占用极大，请注意 batch_size 设置。
2. 包含完整评估流程 (混淆矩阵为紫色 Purples)。
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

# [关键] 导入 VGG 模型
try:
    from vgg_model import VGGBaseline

    print("✅ 成功导入 VGG 模型")
except ImportError:
    print("❌ 致命错误: 找不到 vgg_model.py")
    sys.exit(1)


# =========================================================================
# 1. 训练核心函数
# =========================================================================
def train_vgg(data_dir, num_epochs=100, batch_size=2, patience=15):
    # ⚠️ VGG 警告: 1280x1280 输入下，VGG 显存占用极高。如果 OOM，请将 batch_size 降为 1。
    img_size = 640
    checkpoint_path = 'vgg_checkpoint.pth'
    best_model_path = 'vgg_best.pth'

    print(f"\n[Baseline Config] VGG19_BN | Size: {img_size}x{img_size} | Batch: {batch_size}")

    # --- 数据增强 (保持一致) ---
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

    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 初始化 VGG 模型 ---
    print("Initializing VGG19_BN...")
    model = VGGBaseline(num_classes=2, pretrained=True)
    model = model.to(device)

    # 优化器: VGG 参数多，建议使用 SGD+Momentum 防止过拟合，或者小学习率的 Adam
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 断点续训 ---
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现 VGG 断点文件，正在恢复...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        print(f"✅ 恢复成功！从第 {start_epoch + 1} 轮继续")

    if start_epoch >= num_epochs:
        print("🎉 训练已完成。")
        return model

    # --- 训练循环 ---
    print("\n🚀 开始 VGG 训练 (比较慢，请耐心等待)...")
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
                        print(f"  ★ VGG Loss Improved ({best_loss:.4f} -> {epoch_loss:.4f}). Saving BEST...")
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
                print(f"\n⏹️ 早停触发 (VGG Early Stopping)。")
                break

    except KeyboardInterrupt:
        print("\n⛔ 训练中断。")

    model.load_state_dict(best_model_wts)
    return model


# =========================================================================
# 2. 完整评估函数
# =========================================================================
def evaluate_comprehensive(model, data_dir, img_size=1280):
    print("\n" + "=" * 50)
    print(">>> 启动 VGG 全面评估...")
    print("=" * 50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    error_dir = 'vgg_error_images'
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

                    new_name = f"True_{true_name}_Pred_{pred_name}_{file_name}"
                    try:
                        shutil.copy(current_file_path, os.path.join(error_dir, new_name))
                    except Exception:
                        pass

                global_idx += 1

    # --- 生成报告 ---

    # (1) CSV
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv('vgg_report.csv', encoding='utf-8-sig')
    print("✅ VGG 报告已保存: vgg_report.csv")

    # (2) 混淆矩阵 (紫色 Purples)
    label_map = {'defective': '有缺陷', 'good': '无缺陷'}
    class_names_cn = [label_map.get(name, name) for name in class_names]

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    # [关键] 使用紫色调区分
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names_cn, yticklabels=class_names_cn)
    plt.title('混淆矩阵 - VGG19 (Baseline)', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.savefig('vgg_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 混淆矩阵已保存: vgg_confusion_matrix.png")

    if error_records:
        df_errors = pd.DataFrame(error_records)
        df_errors.to_csv('vgg_error_analysis.csv', index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 发现 {len(error_records)} 个错误样本，详情见 vgg_error_analysis.csv")
    else:
        print("\n🎉 VGG 模型验证集全对！")

    print("✅ 评估完成。")


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    DATA_DIR = '../raw_data'  # 你的数据集路径

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    if os.path.exists(DATA_DIR):
        # 训练 (VGG 显存大，建议 batch_size=2)
        trained_model = train_vgg(DATA_DIR, num_epochs=150, batch_size=8, patience=20)

        # 评估
        evaluate_comprehensive(trained_model, DATA_DIR, img_size=640)
    else:
        print(f"❌ 找不到数据集: {DATA_DIR}")