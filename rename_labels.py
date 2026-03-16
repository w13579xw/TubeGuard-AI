#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
读取 data/unified_dataset 下的 CSV 文件，
将 label 列的中文 "[有缺陷]" 和 "[无缺陷]" 分别替换为 "Defective" 和 "Good"
"""

import os
import csv
from pathlib import Path

def process_csv(csv_path):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"⚠️ {csv_file} 不存在，跳过。")
        return

    print(f"🔄 正在处理: {csv_file}")
    
    # 读入所有行并替换
    updated_rows = []
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        updated_rows.append(header)
        for row in reader:
            if len(row) >= 2:
                # 替换标签
                if "[有缺陷]" in row[1] or "有缺陷" in row[1]:
                    row[1] = "Defective"
                elif "[无缺陷]" in row[1] or "无缺陷" in row[1]:
                    row[1] = "Good"
            updated_rows.append(row)

    # 写回文件（覆盖）
    with open(csv_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)
        
    print(f"✅ {csv_file} 标签已全部转换为英文！")

def main():
    dataset_dir = Path("data/unified_dataset")
    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    for f in csv_files:
        process_csv(dataset_dir / f)

if __name__ == "__main__":
    main()
