#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
实验数据集构建工具 (用于对比、消融及比例划分实验)
支持基于不同划分比例（如 8:1:1, 6:2:2），以及挑选特定的缺陷模式进行消融实验。
"""

import os
import csv
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=str, default="8:1:1")
    parser.add_argument("--ablation_mode", type=str, default="all")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    orig_img_dir = Path("data/images")
    orig_csv = Path("data/train.csv")
    aug_img_dir = Path("data/defect_test")
    aug_csv = Path("data/defect_test/augmented.csv")
    
    parts = [float(x) for x in args.ratios.split(":")]
    assert len(parts) == 3, "ratios format error, e.g. 8:1:1"
    
    total_ratio = sum(parts)
    train_pct, val_pct, test_pct = parts[0]/total_ratio, parts[1]/total_ratio, parts[2]/total_ratio
    
    if not args.output_name:
        rat_str = args.ratios.replace(":", "")
        args.output_name = f"dataset_{args.ablation_mode}_{rat_str}"
        
    output_dir = Path("data/experiments") / args.output_name
    out_img_dir = output_dir / "images"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    all_data = []

    print(f"📖 [1/2] 读取原始正常与真实异常数据...")
    with open(orig_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        for row in reader:
            if len(row) >= 2:
                img_name, label = row[0], row[1]
                src_path = orig_img_dir / img_name
                dst_name = f"orig_{img_name}"
                en_label = "Defective" if ("有缺陷" in label or "Defective" in label) else "Good"
                all_data.append((src_path, dst_name, en_label))
                
    print(f"📖 [2/2] 读取增强缺陷数据 (消融模式: {args.ablation_mode})...")
    pred_csv_path = Path("data/defect_test_heatmaps/predictions.csv")
    predictions = {}
    if pred_csv_path.exists():
        with open(pred_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2: predictions[row[0]] = row[1]

    if aug_csv.exists():
        with open(aug_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    img_name = row[0]
                    # _ablation_mode checking
                    if "aug_" in img_name:
                        if args.ablation_mode != "all" and args.ablation_mode not in img_name:
                            continue
                            
                        label = row[1]
                        label = predictions.get(img_name, label)
                        en_label = "Defective" if ("有缺陷" in label or "Defective" in label) else "Good"
                        src_path = aug_img_dir / img_name
                        all_data.append((src_path, img_name, en_label))

    random.seed(args.seed)
    random.shuffle(all_data)
    if args.limit:
        all_data = all_data[:args.limit]
        
    total = len(all_data)
    train_split = int(total * train_pct)
    val_split = int(total * (train_pct + val_pct))
    
    splits = {
        "train": all_data[:train_split],
        "val": all_data[train_split:val_split],
        "test": all_data[val_split:]
    }
    
    print(f"\n🧩 {args.output_name} -> Train:{len(splits['train'])} | Val:{len(splits['val'])} | Test:{len(splits['test'])}")
    
    for s_name, s_data in splits.items():
        csv_path = output_dir / f"{s_name}.csv"
        valid_records = []
        for src_path, dst_name, lbl in tqdm(s_data, desc=s_name):
            if src_path.exists():
                # 我们不再做实体拷贝，直接在 CSV 中记录原始图片的真实路径 (相对路径)
                # 这极大地节省了不同切分实验占用的大量磁盘空间！
                valid_records.append([str(src_path), lbl])
                
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "label"])
            w.writerows(valid_records)

if __name__ == "__main__":
    main()
