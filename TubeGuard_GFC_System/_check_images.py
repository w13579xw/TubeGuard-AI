#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速查看样本图像信息"""
import cv2
import numpy as np
import pandas as pd

base = r"i:\桌面\TubeGuard-AI\data\images"
df = pd.read_csv(r"i:\桌面\TubeGuard-AI\data\train.csv", encoding="utf-8-sig")

labels = df["label"].unique().tolist()
print("Labels:", labels)
print("Counts:")
for lb in labels:
    print(f"  {lb}: {len(df[df['label']==lb])}")

# 查看各类样本
for lb in labels:
    sub = df[df["label"] == lb].head(3)
    print(f"\n--- {lb} ---")
    for _, row in sub.iterrows():
        path = base + "/" + row["image"]
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"  {row['image']}: {w}x{h}  mean={gray.mean():.1f}  std={gray.std():.1f}")
