#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage A v2 诊断脚本 — 在真实图片上验证新指标的区分能力

使用方法:
    python _diagnose_stage_a.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np

# 加载配置
with open("config/gfc_config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

from src.stage_a_geometric import StageAGeometricScreener
from utils.image_io import load_image

stage_a = StageAGeometricScreener(cfg)

# 读取数据标签
import pandas as pd
train_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
df = pd.read_csv(train_csv)

# 按标签分组
label_col = cfg['data']['label_col']
img_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')

# 自动检测正常/缺陷标签（避免编码问题）
unique_labels = df[label_col].unique().tolist()
print(f"检测到标签: {unique_labels}")

# 简单判定：含"无"的是正常，含"有"的是缺陷
normal_mask = df[label_col].str.contains('无', na=False)
defect_mask = df[label_col].str.contains('有', na=False)
normals = df[normal_mask]['image'].tolist()
defects = df[defect_mask]['image'].tolist()

# 各取 5 张样本
np.random.seed(42)
sample_normal = np.random.choice(normals, size=min(5, len(normals)), replace=False)
sample_defect = np.random.choice(defects, size=min(5, len(defects)), replace=False)

print("=" * 70)
print("  Stage A v2 诊断 — 竖直边线偏差分析指标测试")
print("=" * 70)

def run_sample(label, filenames):
    results = []
    for fname in filenames:
        fpath = os.path.join(img_dir, fname)
        img = load_image(fpath)
        if img is None:
            print(f"  [SKIP] {fname} (加载失败)")
            continue
        res = stage_a.process(img)
        m = res['metrics']
        print(f"  [{label}] {fname:>12s}  "
              f"sigma_edge={m['sigma_edge']:.4f}  "
              f"max_dev={m['max_dev']:.4f}  "
              f"roughness={m['roughness']:.4f}  "
              f"edges={m['n_edges_detected']}  "
              f"pass={'Y' if res['is_pass'] else 'N'}")
        results.append(m)
    return results

print("\n--- 正常样本 ---")
normal_results = run_sample("OK", sample_normal)

print("\n--- 缺陷样本 ---")
defect_results = run_sample("NG", sample_defect)

# 统计汇总
print("\n" + "=" * 70)
print("  指标分布统计")
print("-" * 70)

for name, data in [("正常", normal_results), ("缺陷", defect_results)]:
    if data:
        sigmas = [d['sigma_edge'] for d in data]
        devs   = [d['max_dev'] for d in data]
        roughs = [d['roughness'] for d in data]
        print(f"  {name}  sigma_edge: mean={np.mean(sigmas):.4f}  "
              f"std={np.std(sigmas):.4f}  "
              f"range=[{np.min(sigmas):.4f}, {np.max(sigmas):.4f}]")
        print(f"  {name}  max_dev:    mean={np.mean(devs):.4f}  "
              f"std={np.std(devs):.4f}  "
              f"range=[{np.min(devs):.4f}, {np.max(devs):.4f}]")
        print(f"  {name}  roughness:  mean={np.mean(roughs):.4f}  "
              f"std={np.std(roughs):.4f}  "
              f"range=[{np.min(roughs):.4f}, {np.max(roughs):.4f}]")
        print()

print("=" * 70)
