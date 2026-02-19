#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
YOLOv10-TPH 训练脚本 (支持断点续训版)
功能：
1. 支持 "断点续训"：中断后再次运行，自动从上次停止的轮数继续。
2. 包含早停机制 (Early Stopping)。
3. 包含最佳模型保存 (Best Model Checkpoint)。
4. 防过拟合策略 (数据增强 + Weight Decay)。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import shutil
import pandas as pd
import matplotlib
import copy

# 防止服务器/PyCharm 绘图报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

try:
    from model import YOLOv10TPHClassifier
except ImportError:
    try:
        from model_arch import YOLOv10TPHClassifier
    except ImportError:
        print("❌ 错误: 找不到 model.py 或 model_arch.py")
        exit(1)


# =========================================================================
# 1. 训练函数 (支持断点续训)
# =========================================================================
def train_model(data_dir, num_epochs=100, batch_size=4, patience=15):
    img_size = 1280
    checkpoint_path = 'last_checkpoint.pth'  # 断点文件路径
    best_model_path = 'yolov10_tph_best.pth'  # 最佳模型路径

    print(f"\n[Training Config] Image Size: {img_size}x{img_size}, Batch: {batch_size}")
    print(f"[Config] Checkpoint File: {checkpoint_path}")

    # --- 数据增强 ---
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

    # --- 初始化模型 ---
    base_weight = 'yolov10n.pt'
    if not os.path.exists(base_weight):
        print(f"Downloading {base_weight}...")
        YOLO(base_weight)

    model = YOLOv10TPHClassifier(model_weight=base_weight, num_classes=len(image_datasets['train'].classes))
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 变量初始化 ---
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # =========================================================
    # [核心逻辑] 检查是否存在断点，如果存在则加载
    # =========================================================
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 发现断点文件 '{checkpoint_path}'，正在恢复训练状态...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 恢复优化器状态 (学习率、动量等)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复训练轮数
        start_epoch = checkpoint['epoch'] + 1
        # 恢复最佳Loss和早停计数器
        best_loss = checkpoint.get('best_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)

        # 尝试恢复最佳模型权重 (如果有的话)
        if os.path.exists(best_model_path):
            best_model_wts = torch.load(best_model_path, map_location=device)

        print(f"✅ 成功恢复！将从第 {start_epoch + 1} 轮继续训练 (历史最佳Loss: {best_loss:.4f})")
    else:
        print("✨ 未发现断点，开始新的训练...")

    # 如果已经训练完了，直接退出
    if start_epoch >= num_epochs:
        print("训练任务已完成，无需继续。")
        return model

    # =========================================================
    # 开始训练循环
    # =========================================================
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase")
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

                # --- 验证阶段：早停 + 保存最佳 ---
                if phase == 'val':
                    if epoch_loss < best_loss:
                        print(
                            f"  ✅ Validation Loss Improved ({best_loss:.4f} -> {epoch_loss:.4f}). Saving BEST model...")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), best_model_path)  # 只存纯权重
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        print(f"  ⚠️ Validation Loss did not improve. Counter: {early_stop_counter}/{patience}")

            # --- [关键] 每个 Epoch 结束，保存断点 checkpoint ---
            # 这样即使下一轮崩溃或断电，也能从这一轮恢复
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'early_stop_counter': early_stop_counter
            }, checkpoint_path)

            # 触发早停
            if early_stop_counter >= patience:
                print(f"\n⏹️ Early stopping triggered! No improvement for {patience} epochs.")
                break

    except KeyboardInterrupt:
        print("\n⛔ 检测到手动中断 (Ctrl+C)。")
        print("当前状态已通过 'last_checkpoint.pth' 保存。下次运行将自动继续训练。")

    print('Training process ended.')

    # 恢复最佳权重并返回
    model.load_state_dict(best_model_wts)
    return model


# =========================================================================
# 2. 评估函数 (CSV & 混淆矩阵)
# =========================================================================
def evaluate_comprehensive(model, data_dir, img_size=1280):
    print("\n" + "=" * 50)
    print(">>> 启动全面评估 & 坏例捕捉...")
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

    # 准备坏例目录
    error_dir = 'error_images'
    if os.path.exists(error_dir):
        shutil.rmtree(error_dir)
    os.makedirs(error_dir)

    all_preds = []
    all_labels = []
    error_records = []

    class_names = val_dataset.classes
    file_paths = [s[0] for s in val_dataset.samples]

    print("正在推理并筛选错误样本...")
    global_idx = 0

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

    # 报告生成
    label_map = {'defective': '有缺陷', 'good': '无缺陷'}
    class_names_cn = [label_map.get(name, name) for name in class_names]

    # CSV
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv('classification_report.csv', encoding='utf-8-sig')

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',

                xticklabels=class_names_cn, yticklabels=class_names_cn)
    plt.title('混淆矩阵 - yolov10_tph)', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 坏例 CSV
    if error_records:
        pd.DataFrame(error_records).to_csv('error_analysis.csv', index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 发现 {len(error_records)} 个错误样本，详情见 error_analysis.csv")
    else:
        print("\n🎉 完美！验证集全对。")
    print("✅ 评估完成。")


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    DATA_DIR = '../raw_data'

    # 动态切换路径以防止FileNotFound (可选，如果还是报错请加上之前的sys.path代码)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)



    if os.path.exists(DATA_DIR):
        # 1. 训练
        model = train_model(DATA_DIR, num_epochs=150, batch_size=8, patience=20)
        # 2. 评估
        evaluate_comprehensive(model, DATA_DIR, img_size=1280)
    else:
        print(f"❌ 未找到数据集目录: {DATA_DIR}")