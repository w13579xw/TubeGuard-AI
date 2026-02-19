#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
YOLOv10 基线模型 (Baseline) 完整训练脚本
功能：
1. 训练纯净版 YOLOv10 Classifier (无注意力机制)。
2. 包含断点续训、早停机制 (Early Stopping)。
3. [重点] 包含完整的模型评估：混淆矩阵图、CSV报告、坏例捕捉 (Bad Case Analysis)。
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

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# [关键] 导入基线模型
try:
    from baseline_model import YOLOv10BaselineClassifier

    print("✅ 成功导入基线模型: YOLOv10BaselineClassifier")
except ImportError:
    print("❌ 致命错误: 找不到 baseline_model.py")
    sys.exit(1)


# =========================================================================
# 1. 训练核心函数
# =========================================================================
def train_baseline(data_dir, num_epochs=100, batch_size=4, patience=15):
    img_size = 1280
    checkpoint_path = 'baseline_checkpoint.pth'
    best_model_path = 'baseline_best.pth'

    print(f"\n[Baseline Config] YOLOv10-Vanilla | Size: {img_size}x{img_size} | Batch: {batch_size}")

    # --- 数据增强 (保持与改进模型一致，控制变量) ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    # --- 初始化基线模型 ---
    print("Initializing YOLOv10 Baseline Network...")
    model = YOLOv10BaselineClassifier(num_classes=2)
    model = model.to(device)

    # 优化器 (基线通常使用 SGD 或 AdamW，这里用 SGD+Momentum 作为经典设定)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 学习率调整
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现断点文件 '{checkpoint_path}'，正在恢复...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        print(f"✅ 恢复成功！从第 {start_epoch + 1} 轮继续 (历史最佳 Loss: {best_loss:.4f})")

    if start_epoch >= num_epochs:
        print("🎉 训练已完成。")
        return model

    # --- 训练循环 ---
    print("\n🚀 开始基线训练...")
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

                # 验证与保存
                if phase == 'val':
                    scheduler.step()
                    if epoch_loss < best_loss:
                        print(f"  ★ Baseline Loss Improved ({best_loss:.4f} -> {epoch_loss:.4f}). Saving BEST...")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), best_model_path)
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        print(f"  ⚠️ No improvement. Counter: {early_stop_counter}/{patience}")

            # 保存断点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'early_stop_counter': early_stop_counter
            }, checkpoint_path)

            if early_stop_counter >= patience:
                print(f"\n⏹️ 早停触发 (Baseline Early Stopping)。")
                break

    except KeyboardInterrupt:
        print("\n⛔ 训练中断。")

    model.load_state_dict(best_model_wts)
    return model


# =========================================================================
# 2. 完整评估函数 (Full Evaluation)
# =========================================================================
def evaluate_comprehensive(model, data_dir, img_size=1280):
    print("\n" + "=" * 50)
    print(">>> 启动基线模型全面评估 (Baseline Evaluation)...")
    print("=" * 50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 1. 准备验证集 (无 shuffle)
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 2. 准备坏例存储目录 (baseline 专用)
    error_dir = 'baseline_error_images'
    if os.path.exists(error_dir):
        shutil.rmtree(error_dir)
    os.makedirs(error_dir)
    print(f"-> 创建坏例目录: {error_dir}/")

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

            # 3. 逐个样本分析
            for i in range(inputs.size(0)):
                current_label = labels[i].item()
                current_pred = preds[i].item()

                all_labels.append(current_label)
                all_preds.append(current_pred)

                # 捕捉错误
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

    # --- 4. 生成结果 ---

    # (1) CSV 报告
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv('baseline_report.csv', encoding='utf-8-sig')
    print("✅ 基线报告已保存: baseline_report.csv")

    # (2) 混淆矩阵图
    label_map = {'defective': '有缺陷', 'good': '无缺陷'}
    class_names_cn = [label_map.get(name, name) for name in class_names]

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_cn, yticklabels=class_names_cn)
    plt.title('混淆矩阵 - yolov10)', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.savefig('baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 混淆矩阵已保存: baseline_confusion_matrix.png")

    # (3) 坏例 CSV
    if error_records:
        df_errors = pd.DataFrame(error_records)
        df_errors.to_csv('baseline_error_analysis.csv', index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 发现 {len(error_records)} 个错误样本，详情见 baseline_error_analysis.csv")
    else:
        print("\n🎉 基线模型验证集全对 (这很罕见)！")

    print("✅ 基线评估完成。")


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    DATA_DIR = '../raw_data'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    if os.path.exists(DATA_DIR):
        # 1. 训练
        trained_model = train_baseline(DATA_DIR, num_epochs=150, batch_size=8, patience=30)

        # 2. 评估
        evaluate_comprehensive(trained_model, DATA_DIR, img_size=1280)
    else:
        print(f"❌ 找不到数据集: {DATA_DIR}")