#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_custom_dataset(csv_paths, source_img_dir, output_root):
    """
    将定制的CSV数据转换为YOLOv10分类所需的文件夹结构。
    csv_paths: 字典, {'train': 'train.csv', 'test': 'test.csv'}
    """
    # 标签映射：中文 -> 英文（避免编码问题）
    label_map = {
        '有缺陷': 'defective',
        '无缺陷': 'good'
    }

    # 预处理函数
    def clean_label(label_str):
        # 移除 并映射
        clean = label_str.replace('[', '').replace(']', '')
        return label_map.get(clean, 'unknown')

    # 1. 处理训练集并分割出验证集
    df_train_full = pd.read_csv(csv_paths['train'])
    df_train_full['clean_label'] = df_train_full['label'].apply(clean_label)

    # 分层抽样分割 Train/Val
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        stratify=df_train_full['clean_label'],
        random_state=42
    )

    # 2. 处理测试集
    df_test = pd.read_csv(csv_paths['test'])
    df_test['clean_label'] = df_test['label'].apply(clean_label)

    datasets = {
        'train': train_df,
        'val': val_df,
        'test': df_test
    }

    # 3. 移动文件
    for split, df in datasets.items():
        print(f"Processing {split} set...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['image']
            cls_name = row['clean_label']

            src_path = os.path.join(source_img_dir, filename)
            dst_dir = os.path.join(output_root, split, cls_name)

            # 创建目标目录
            os.makedirs(dst_dir, exist_ok=True)

            dst_path = os.path.join(dst_dir, filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                # 记录缺失文件，工业场景中常出现索引存在但文件丢失的情况
                print(f"Warning: Image {filename} not found.")

# 调用示例
process_custom_dataset(
    csv_paths={'train': 'data/train.csv', 'test': 'data/test.csv'},
    source_img_dir='../data/images',
    output_root='../yolo_cls_data'

)