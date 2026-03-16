#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
读取 YOLOv10-TPH 的预测结果 (predictions.csv)，
并使用这些预测结果作为统一数据集中增强图像的真实标签。
"""

import os
import csv
from pathlib import Path

def main():
    predictions_csv = Path("data/defect_test_heatmaps/predictions.csv")
    unified_dir = Path("data/unified_dataset")
    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    if not predictions_csv.exists():
        print(f"❌ 找不到预测文件: {predictions_csv}")
        return

    # 1. 加载预测结果
    print(f"📖 加载预测结果: {predictions_csv}")
    predictions = {}
    with open(predictions_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) >= 2:
                img_name = row[0]
                pred_label = row[1]
                predictions[img_name] = pred_label
                
    print(f"✅ 成功加载 {len(predictions)} 条预测记录")

    # 2. 更新统一数据集的 CSV
    for csv_filename in csv_files:
        csv_path = unified_dir / csv_filename
        if not csv_path.exists():
            print(f"⚠️ 找不到文件: {csv_path}，跳过。")
            continue
            
        print(f"\n🔄 正在更新: {csv_path}")
        updated_rows = []
        updated_count = 0
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            updated_rows.append(header)
            
            for row in reader:
                if len(row) >= 2:
                    img_name = row[0]
                    current_label = row[1]
                    
                    # 如果是增强生成的图片 (没有 orig_ 前缀，由于之前 build_unified_dataset 时增强图片没加前缀，原图加了)
                    # 我们只更新增强图片的标签，原图的标签保持不变
                    if not img_name.startswith("orig_"):
                        # 如果在预测结果里找到了这个图片
                        if img_name in predictions:
                            new_label = predictions[img_name]
                            if current_label != new_label:
                                row[1] = new_label
                                updated_count += 1
                                
                updated_rows.append(row)
                
        # 写回文件
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
            
        print(f"✅ 成功更新 {csv_filename} 中的 {updated_count} 条记录。")

if __name__ == "__main__":
    main()
