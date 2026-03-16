#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TubeGuard GFC 系统冒烟测试
运行: python _verify_system.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import torch

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = []

def check(name, fn):
    try:
        result = fn()
        print(f"  [{PASS}]  {name}" + (f"  {result}" if result else ""))
    except Exception as e:
        print(f"  [{FAIL}]  {name}  =>  {e}")
        errors.append((name, str(e)))

print("=" * 60)
print("  TubeGuard GFC System — 冒烟测试")
print("=" * 60)

# ── 1. 配置加载 ──────────────────────────────────────────────
cfg = None
def load_cfg():
    global cfg
    with open("config/gfc_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return f"stage_a keys={list(cfg.get('stage_a',{}).keys())}"
check("配置文件加载", load_cfg)

# ── 2. core_algorithms ───────────────────────────────────────
from src.core_algorithms import SubPixelEdgeDetector, fit_circle_least_squares

def test_kasa():
    thetas = np.linspace(0, 2*np.pi, 100, endpoint=False)
    pts = np.column_stack([50 + 30*np.cos(thetas), 60 + 30*np.sin(thetas)])
    pts += np.random.normal(0, 0.1, pts.shape)
    (cx, cy), r, res = fit_circle_least_squares(pts)
    assert res < 0.3, f"残差过大: {res:.4f}"
    return f"cx={cx:.2f} cy={cy:.2f} r={r:.2f} res={res:.4f}"
check("Kasa 圆拟合", test_kasa)

def test_steger():
    det = SubPixelEdgeDetector(method='steger', sigma=1.5, min_response=5.0)
    canvas = np.zeros((60, 120), dtype=np.uint8)
    canvas[28:32, 10:110] = 200
    pts = det.detect(canvas)
    return f"检测到 {len(pts)} 个亚像素点"
check("Steger 亚像素检测", test_steger)

def test_zernike():
    det = SubPixelEdgeDetector(method='zernike', sigma=1.0, min_response=15.0)
    import cv2
    img = np.zeros((80, 80), dtype=np.uint8)
    img[40:, :] = 200
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    pts = det.detect(img)
    return f"检测到 {len(pts)} 个亚像素点"
check("Zernike 亚像素检测", test_zernike)

# ── 3. Stage A ───────────────────────────────────────────────
from src.stage_a_geometric import StageAGeometricScreener

def test_stage_a():
    a = StageAGeometricScreener(cfg)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    res = a.process(img)
    keys = list(res.keys())
    assert "is_pass" in keys and "metrics" in keys
    return f"is_pass={res['is_pass']}  sigma_edge={res['metrics'].get('sigma_edge', 0):.3f}"
check("Stage A 几何初筛 (合成图)", test_stage_a)

# ── 4. YOLOv10-TPH Transformer 验证 ──────────────────────────
from models.yolov10_tph.yolov10_tph_model import TransformerBlock

def test_tph_block():
    tph = TransformerBlock(c1=64, c2=64, num_heads=4)
    x = torch.randn(1, 64, 16, 16)
    out = tph(x)
    assert out.shape == x.shape
    params = sum(p.numel() for p in tph.parameters())
    return f"in/out={tuple(x.shape)} params={params:,}"
check("TPH Transformer 前向传播", test_tph_block)

# ── 5. YOLOv10-TPH 分类推理 ──────────────────────────────────
from models.yolov10_tph.yolov10_tph_model import YOLOv10TPHInference

TPH_PATH = "weights/yolov10_tph_best.pth"

def test_tph_inference():
    inf = YOLOv10TPHInference(TPH_PATH)
    loaded = inf.model is not None
    if loaded:
        # 用白色纯色图推理（不期望有检测结果，只验证不报错）
        import cv2
        dummy = np.ones((320, 320, 3), dtype=np.uint8) * 200
        crops = [{"crop": dummy, "box": [0, 0, 320, 320], "score": 0.8}]
        defects = inf.predict(crops)
        return f"model_loaded=True  results_count={len(defects)}"
    else:
        return "model_loaded=False (降级模式，正常)"
check("YOLOv10-TPH 分类推理", test_tph_inference)

# ── 6. PatchCore 类 ──────────────────────────────────────────
from models.patchcore.patchcore_model import PatchCore, greedy_coreset

def test_patchcore_class():
    pc = PatchCore(img_size=224, coreset_ratio=0.1)
    assert pc.memory_bank is None
    # coreset 小测试
    feats = np.random.randn(200, 64).astype(np.float32)
    core  = greedy_coreset(feats, n_coreset=20)
    assert core.shape == (20, 64)
    return f"PatchCore 类OK  coreset shape={core.shape}"
check("PatchCore 类 + Coreset 采样", test_patchcore_class)

# ── 7. Stage B (SSIM fallback) ───────────────────────────────
from src.stage_b_semantic import StageBSemanticAnalyzer

def test_stage_b():
    b = StageBSemanticAnalyzer(cfg)   # 无记忆库 → SSIM
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    res = b.process(img)
    keys = list(res.keys())
    assert "is_clean" in keys and "anomaly_score" in keys
    return f"is_clean={res['is_clean']}  score={res['anomaly_score']:.3f}  defects={res['defects']}"
check("Stage B 语义分析 (SSIM 降级)", test_stage_b)

# ── 8. utils ─────────────────────────────────────────────────
def test_utils():
    from utils.image_io import load_image, draw_annotations, save_debug_image
    from utils.logger import setup_logger, export_report, log_result
    # 测试 draw_annotations 不崩溃
    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    vis = draw_annotations(img, metrics={"diameter_mm": 50.0, "wall_thickness_mm": 5.0,
                                          "inner_cx": 100, "inner_cy": 100, "inner_r": 60,
                                          "outer_cx": 100, "outer_cy": 100, "outer_r": 80})
    assert vis.shape == img.shape
    return "draw_annotations OK"
check("utils (image_io + logger)", test_utils)

# ── 9. main_pipeline 可导入 ──────────────────────────────────
def test_main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_pipeline", "main_pipeline.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    assert hasattr(mp, "main") and hasattr(mp, "parse_args")
    return "main() + parse_args() 存在"
check("main_pipeline 导入", test_main)

# ── 汇总 ────────────────────────────────────────────────────
print()
print("=" * 60)
if errors:
    print(f"  结论: {len(errors)} 项失败:")
    for name, msg in errors:
        print(f"    ✗ {name}: {msg}")
else:
    print("  结论: 全部通过！系统可以正常运行。")
    print()
    print("  下一步:")
    print("  1. 训练 PatchCore 记忆库:")
    print("     python models/patchcore/train_patchcore.py")
    print()
    print("  2. 运行完整检测流水线:")
    print("     python main_pipeline.py --csv data/test.csv --img_dir data/images")
print("=" * 60)
