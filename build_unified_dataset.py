#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
合并原始数据与合成增强缺陷数据，并按比例划分为 Train / Val / Test 集合
将文件统一拷贝到 data/unified_dataset 中，并生成分别的 csv 标注
"""

import os
import csv
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def main():
    # ==== 配置路径 ====
    orig_img_dir = Path("data/images")
    orig_csv = Path("data/train.csv")
    
    aug_img_dir = Path("data/defect_test")
    aug_csv = Path("data/defect_test/augmented.csv")
    
    output_dir = Path("data/unified_dataset")
    out_img_dir = output_dir / "images"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    # ==== 读取 CSV 提取全部条目 ====
    all_data = []  # 格式: (原路径, 目标文件名, 标签)
    
    # 1. 读原始
    print(f"📖 读取原始数据: {orig_csv}")
    with open(orig_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                img_name, label = row[0], row[1]
                src_path = orig_img_dir / img_name
                # 因为会有同名，给原始数据加个前缀
                dst_name = f"orig_{img_name}"
                all_data.append((src_path, dst_name, label))
                
    # 2. 读增强 (注意增强文件的前 2400 个其实是原 CSV 的拷贝，所以我们要判断如果在 orig_img_dir 则不管，只有真实以 aug_ 结尾或存在于 aug_img_dir 特有的才是我们要的)
    print(f"📖 读取增强数据: {aug_csv}")
    with open(aug_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) >= 2:
                img_name, label = row[0], row[1]
                # 这是新生成的数据
                if "aug_" in img_name: 
                    src_path = aug_img_dir / img_name
                    # 这个已经唯一了，不需要加前缀，但为了保险还是直接用
                    all_data.append((src_path, img_name, label))

    print(f"总计收集到有效样本数目: {len(all_data)}")
    
    # 洗牌
    random.seed(42)
    random.shuffle(all_data)
    
    # 划分比例 8:1:1
    total = len(all_data)
    train_split = int(total * 0.8)
    val_split = int(total * 0.9)
    
    train_data = all_data[:train_split]
    val_data = all_data[train_split:val_split]
    test_data = all_data[val_split:]
    
    print(f"划分结果 -> Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # ==== 拷贝文件并写入 CSV ====
    def save_split(split_name, data):
        csv_path = output_dir / f"{split_name}.csv"
        print(f"\n🚀 开始处理 {split_name} 集...")
        valid_rows = []
        for src_path, dst_name, label in tqdm(data, desc=f"Copying {split_name}"):
            dst_path = out_img_dir / dst_name
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                valid_rows.append([dst_name, label])
            else:
                print(f"⚠️ 文件找不到: {src_path}")
                
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label"])
            writer.writerows(valid_rows)
            
        print(f"✅ {split_name} 集处理完毕。生成 CSV: {csv_path}")
        
    save_split("train", train_data)
    save_split("val", val_data)
    save_split("test", test_data)
    
    # ==== 生成 YOLO YAML 配置文件 ====
    yaml_content = f"""path: {output_dir.absolute().as_posix()}
train: train.csv
val: val.csv
test: test.csv

# 此处仅为了格式合规，YOLOv10_TPH 的实际数据集读取可能是通过 Dataloader 直接读 CSV
nc: 2
names: ['Defective', 'Good']
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"\n🎉 统一数据集构建完成，保存在 {output_dir}。")

if __name__ == "__main__":
    main()
