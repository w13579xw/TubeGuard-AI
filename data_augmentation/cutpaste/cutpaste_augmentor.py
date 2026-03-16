#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CutPaste 真实缺陷模拟器（增强版 v2）
============================================================
在原始 CutPaste 基础上新增多种真实缺陷模拟策略：

    1. 程序化划痕/裂纹绘制 —— 使用 Bezier 曲线 + 随机游走生成
       真实感的划痕、裂纹线条，支持分叉和宽度渐变
    2. 局部光照扰动 —— 模拟工业光源照射不均匀、局部反光、
       阴影等光照异常
    3. 表面纹理劣化 —— 模拟磨损、腐蚀、气泡等面状缺陷
    4. 组合缺陷 —— 随机叠加多种缺陷类型，更贴合真实场景
    5. ROI 感知（Otsu 前景掩膜）+ 泊松融合（保留原有功能）

输出：
    - 增强后的伪缺陷图像
    - 更新后的 CSV 标注文件
"""

import os
import csv
import math
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

# =========================================================================
# 日志配置
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class CutPasteAugmentor:
    """
    CutPaste 真实缺陷模拟器（增强版 v2）

    支持的缺陷类型：
        - scratch:   程序化划痕/裂纹（Bezier 曲线 + 随机游走 + 分叉）
        - lighting:  局部光照扰动（反光、阴影、亮度渐变）
        - texture:   表面纹理劣化（磨损、颗粒噪声、腐蚀斑点）
        - cutpaste:  传统 CutPaste 裁剪粘贴（附带泊松融合）
        - combined:  随机组合上述多种缺陷

    参数：
        defect_mode:         缺陷模式 ("scratch"/"lighting"/"texture"/"cutpaste"/"combined")
        scratch_count:       每张图像生成的划痕数量范围 (min, max)
        scratch_width:       划痕线宽范围 (min, max)，单位：像素
        scratch_opacity:     划痕不透明度范围 (min, max)，0.0~1.0
        lighting_strength:   光照扰动强度 (0.0~1.0)
        texture_severity:    纹理劣化程度 (0.0~1.0)
        roi_aware:           是否启用 ROI 感知（Otsu 前景提取）
        use_poisson_blend:   CutPaste 模式下是否使用泊松融合
        seed:                随机种子
    """

    def __init__(
        self,
        defect_mode: str = "combined",
        scratch_count: Tuple[int, int] = (1, 4),
        scratch_width: Tuple[int, int] = (1, 3),
        scratch_opacity: Tuple[float, float] = (0.3, 0.8),
        lighting_strength: float = 0.5,
        texture_severity: float = 0.4,
        roi_aware: bool = True,
        use_poisson_blend: bool = True,
        seed: Optional[int] = None
    ):
        self.defect_mode = defect_mode
        self.scratch_count = scratch_count
        self.scratch_width = scratch_width
        self.scratch_opacity = scratch_opacity
        self.lighting_strength = lighting_strength
        self.texture_severity = texture_severity
        self.roi_aware = roi_aware
        self.use_poisson_blend = use_poisson_blend

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ==================================================================
    #  ROI 前景掩膜提取（Otsu 阈值法）
    # ==================================================================
    def _extract_foreground_mask(self, image: Image.Image) -> np.ndarray:
        """使用 Otsu 阈值法提取管道前景区域掩膜"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if np.sum(mask > 0) < 0.1 * mask.size:
            mask = 255 - mask

        return mask

    def _get_roi_region(self, image: Image.Image) -> Optional[np.ndarray]:
        """获取 ROI 掩膜；未启用时返回 None"""
        if self.roi_aware:
            return self._extract_foreground_mask(image)
        return None

    # ==================================================================
    #  缺陷类型 1：程序化划痕 / 裂纹绘制
    # ==================================================================
    def _generate_scratch(self, image: Image.Image) -> Image.Image:
        """
        在图像上绘制真实感划痕/裂纹

        策略：
        - 使用随机游走生成不规则曲线路径
        - 沿路径绘制带宽度渐变的半透明线条
        - 在划痕两侧添加微弱的亮边（模拟光线折射）
        - 随机添加分叉（模拟应力裂纹扩展）
        - 局部降低亮度（模拟凹陷）
        """
        result = np.array(image, dtype=np.float32)
        h, w = result.shape[:2]
        roi_mask = self._get_roi_region(image)

        num_scratches = random.randint(*self.scratch_count)

        for _ in range(num_scratches):
            # --- 生成随机游走路径 ---
            points = self._random_walk_path(w, h, roi_mask)
            if len(points) < 3:
                continue

            # 计算包围盒 (BBox) 局部化内存
            pts_arr = np.array(points)
            min_x, min_y = np.min(pts_arr, axis=0) - 20
            max_x, max_y = np.max(pts_arr, axis=0) + 20
            x1, x2 = max(0, min_x), min(w, max_x)
            y1, y2 = max(0, min_y), min(h, max_y)
            if x2 <= x1 or y2 <= y1: continue

            bh, bw = y2 - y1, x2 - x1

            # 变换坐标到局部
            local_points = [(p[0] - x1, p[1] - y1) for p in points]

            # --- 获取局部区域 ---
            region = result[y1:y2, x1:x2].copy()

            # --- 获取划痕参数 ---
            base_width = random.randint(*self.scratch_width)
            opacity = random.uniform(*self.scratch_opacity)

            # --- 获取路径周围的平均颜色，划痕取更暗色调 ---
            avg_color = self._sample_local_color(result, points)
            darkness = random.uniform(0.4, 0.7)
            scratch_color = (avg_color * darkness).astype(np.float32)

            # --- 绘制主划痕（基于局部尺寸） ---
            scratch_layer = np.zeros_like(region)
            scratch_mask = np.zeros((bh, bw), dtype=np.float32)

            for i in range(len(local_points) - 1):
                seg_width = max(1, base_width + random.randint(-1, 1))
                pt1, pt2 = local_points[i], local_points[i + 1]
                cv2.line(scratch_layer, pt1, pt2, scratch_color.tolist(), seg_width, cv2.LINE_AA)
                cv2.line(scratch_mask, pt1, pt2, 1.0, seg_width, cv2.LINE_AA)

            # --- 高光边缘（基于局部尺寸） ---
            highlight_layer = np.zeros_like(region)
            highlight_mask = np.zeros((bh, bw), dtype=np.float32)
            highlight_color = np.minimum(avg_color * 1.3, 255.0)

            for i in range(len(local_points) - 1):
                pt1, pt2 = local_points[i], local_points[i + 1]
                offset = random.choice([-1, 1])
                pt1_off = (pt1[0] + offset, pt1[1] + offset)
                pt2_off = (pt2[0] + offset, pt2[1] + offset)
                cv2.line(highlight_layer, pt1_off, pt2_off, highlight_color.tolist(), 1, cv2.LINE_AA)
                cv2.line(highlight_mask, pt1_off, pt2_off, 1.0, 1, cv2.LINE_AA)

            scratch_mask = cv2.GaussianBlur(scratch_mask, (3, 3), 0.8)
            highlight_mask = cv2.GaussianBlur(highlight_mask, (3, 3), 0.5)

            for c in range(3):
                region[:, :, c] = (
                    region[:, :, c] * (1 - scratch_mask * opacity)
                    + scratch_layer[:, :, c] * scratch_mask * opacity
                )
                region[:, :, c] = (
                    region[:, :, c] * (1 - highlight_mask * opacity * 0.3)
                    + highlight_layer[:, :, c] * highlight_mask * opacity * 0.3
                )

            # --- 分叉 ---
            if random.random() < 0.4 and len(points) > 3:
                branch_start_idx = random.randint(1, len(points) - 2)
                branch_points = self._random_walk_path(
                    w, h, roi_mask,
                    start=points[branch_start_idx],
                    num_steps=random.randint(5, 15)
                )
                if len(branch_points) >= 2:
                    branch_width = max(1, base_width - 1)
                    branch_opacity = opacity * 0.6
                    
                    # 扩展当前 BBox 容纳分叉
                    b_pts = np.array(branch_points)
                    b_min_x, b_min_y = np.min(b_pts, axis=0) - 20
                    b_max_x, b_max_y = np.max(b_pts, axis=0) + 20
                    nx1, nx2 = max(0, min(x1, b_min_x)), min(w, max(x2, b_max_x))
                    ny1, ny2 = max(0, min(y1, b_min_y)), min(h, max(y2, b_max_y))
                    
                    if nx2 > nx1 and ny2 > ny1:
                        nbh, nbw = ny2 - ny1, nx2 - nx1
                        n_region = result[ny1:ny2, nx1:nx2].copy()
                        
                        # 把此前的 region 贴入 n_region
                        dx, dy = x1 - nx1, y1 - ny1
                        n_region[dy:dy+bh, dx:dx+bw] = region

                        n_scratch_layer = np.zeros_like(n_region)
                        n_branch_mask = np.zeros((nbh, nbw), dtype=np.float32)
                        
                        n_local_branch = [(p[0] - nx1, p[1] - ny1) for p in branch_points]

                        for i in range(len(n_local_branch) - 1):
                            pt1, pt2 = n_local_branch[i], n_local_branch[i + 1]
                            cv2.line(n_scratch_layer, pt1, pt2, scratch_color.tolist(), branch_width, cv2.LINE_AA)
                            cv2.line(n_branch_mask, pt1, pt2, 1.0, branch_width, cv2.LINE_AA)

                        n_branch_mask = cv2.GaussianBlur(n_branch_mask, (3, 3), 0.5)
                        for c in range(3):
                            n_region[:, :, c] = (
                                n_region[:, :, c] * (1 - n_branch_mask * branch_opacity)
                                + n_scratch_layer[:, :, c] * n_branch_mask * branch_opacity
                            )
                            
                        # 更新外包围信息
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                        region = n_region

            # 最终回填到大图
            result[y1:y2, x1:x2] = region

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _random_walk_path(
        self, w: int, h: int,
        roi_mask: Optional[np.ndarray] = None,
        start: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        随机游走生成不规则曲线路径（模拟真实划痕走向）

        参数：
            w, h:      图像尺寸
            roi_mask:  前景掩膜（可选）
            start:     起点（可选，默认随机）
            num_steps: 游走步数（可选，默认根据图像大小决定）
        """
        if num_steps is None:
            num_steps = random.randint(15, 40)

        # 步长为图像短边的 1%~3%
        step_len = random.uniform(min(w, h) * 0.01, min(w, h) * 0.03)

        # 起点
        if start is None:
            for _ in range(30):
                sx = random.randint(int(w * 0.1), int(w * 0.9))
                sy = random.randint(int(h * 0.1), int(h * 0.9))
                if roi_mask is None or roi_mask[sy, sx] > 0:
                    break
            start = (sx, sy)

        # 主方向角度 + 随机偏转
        main_angle = random.uniform(0, 2 * math.pi)
        points = [start]

        for _ in range(num_steps):
            # 每步偏转 ±25°，保持一定的方向连贯性
            main_angle += random.uniform(-math.pi / 7, math.pi / 7)
            nx = int(points[-1][0] + step_len * math.cos(main_angle))
            ny = int(points[-1][1] + step_len * math.sin(main_angle))

            # 边界约束
            nx = max(1, min(nx, w - 2))
            ny = max(1, min(ny, h - 2))

            # ROI 约束（如果新点不在前景内则停止）
            if roi_mask is not None and roi_mask[ny, nx] == 0:
                break

            points.append((nx, ny))

        return points

    def _sample_local_color(self, img: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """采样路径周围像素的平均颜色"""
        colors = []
        h, w = img.shape[:2]
        for pt in points[::3]:  # 每隔 3 个点采样
            x, y = pt
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            colors.append(img[y, x])
        if colors:
            return np.mean(colors, axis=0)
        return np.array([128.0, 128.0, 128.0])

    # ==================================================================
    #  缺陷类型 2：局部光照扰动
    # ==================================================================
    def _generate_lighting_defect(self, image: Image.Image) -> Image.Image:
        """
        模拟局部光照异常

        策略组合（随机选择 1~2 种）：
        - 局部亮斑（模拟反光、高光过曝）
        - 局部暗区（模拟阴影遮挡）
        - 条纹光照渐变（模拟光源非均匀照射）
        """
        result = np.array(image, dtype=np.float32)
        h, w = result.shape[:2]
        roi_mask = self._get_roi_region(image)
        strength = self.lighting_strength

        effects = random.sample(["bright_spot", "dark_patch", "gradient_stripe"],
                                k=random.randint(1, 2))

        for effect in effects:
            if effect == "bright_spot":
                # --- 局部亮斑：椭圆形高斯亮斑 ---
                cx = random.randint(int(w * 0.2), int(w * 0.8))
                cy = random.randint(int(h * 0.2), int(h * 0.8))

                rx = random.randint(int(w * 0.05), int(w * 0.15))
                ry = random.randint(int(h * 0.05), int(h * 0.15))

                x1, x2 = max(0, cx - 3*rx), min(w, cx + 3*rx)
                y1, y2 = max(0, cy - 3*ry), min(h, cy + 3*ry)
                if x2 <= x1 or y2 <= y1: continue

                Y, X = np.ogrid[y1:y2, x1:x2]
                gauss = np.exp(-((X - cx) ** 2 / (2 * rx ** 2) + (Y - cy) ** 2 / (2 * ry ** 2)))
                gauss = gauss.astype(np.float32)

                if roi_mask is not None:
                    gauss *= (roi_mask[y1:y2, x1:x2] / 255.0).astype(np.float32)

                intensity = random.uniform(30, 80) * strength
                for c in range(3):
                    result[y1:y2, x1:x2, c] += gauss * intensity

            elif effect == "dark_patch":
                # --- 局部暗区：不规则形状的阴影 ---
                cx = random.randint(int(w * 0.15), int(w * 0.85))
                cy = random.randint(int(h * 0.15), int(h * 0.85))

                rx = random.randint(int(w * 0.04), int(w * 0.12))
                ry = random.randint(int(h * 0.04), int(h * 0.12))

                x1, x2 = max(0, cx - 3*rx), min(w, cx + 3*rx)
                y1, y2 = max(0, cy - 3*ry), min(h, cy + 3*ry)
                if x2 <= x1 or y2 <= y1: continue

                Y, X = np.ogrid[y1:y2, x1:x2]
                gauss = np.exp(-((X - cx) ** 2 / (2 * rx ** 2) + (Y - cy) ** 2 / (2 * ry ** 2)))
                gauss = gauss.astype(np.float32)

                if roi_mask is not None:
                    gauss *= (roi_mask[y1:y2, x1:x2] / 255.0).astype(np.float32)

                intensity = random.uniform(0.6, 0.85)
                darken = 1.0 - gauss * (1.0 - intensity) * strength
                for c in range(3):
                    result[y1:y2, x1:x2, c] *= darken

            elif effect == "gradient_stripe":
                # --- 条纹光照渐变：模拟光源条纹照射 ---
                angle = random.uniform(0, math.pi)
                stripe_width = random.randint(int(min(w, h) * 0.08), int(min(w, h) * 0.2))

                # 使用 ogrid（广播机制）代替 meshgrid，大幅节省内存
                Y, X = np.ogrid[:h, :w]
                proj = X * math.cos(angle) + Y * math.sin(angle)
                
                # 计算中心点
                px = w / 2
                py = h / 2
                center = px * math.cos(angle) + py * math.sin(angle) + random.uniform(-stripe_width, stripe_width)

                stripe = np.exp(-((proj - center) ** 2) / (2 * stripe_width ** 2))
                stripe = stripe.astype(np.float32)

                if roi_mask is not None:
                    stripe *= (roi_mask / 255.0).astype(np.float32)

                if random.random() < 0.5:
                    intensity = random.uniform(20, 50) * strength
                    for c in range(3):
                        result[:, :, c] += stripe * intensity
                else:
                    intensity = random.uniform(0.7, 0.9)
                    darken = 1.0 - stripe * (1.0 - intensity) * strength
                    for c in range(3):
                        result[:, :, c] *= darken

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    # ==================================================================
    #  缺陷类型 3：表面纹理劣化
    # ==================================================================
    def _generate_texture_defect(self, image: Image.Image) -> Image.Image:
        """
        模拟表面纹理劣化

        策略组合：
        - 局部磨损：降低对比度 + 轻微模糊
        - 颗粒噪声：局部添加椒盐/高斯噪声
        - 腐蚀斑点：随机小圆点簇
        """
        result = np.array(image, dtype=np.float32)
        h, w = result.shape[:2]
        roi_mask = self._get_roi_region(image)
        severity = self.texture_severity

        effects = random.sample(["wear", "grain_noise", "corrosion_spots"],
                                k=random.randint(1, 2))

        for effect in effects:
            if effect == "wear":
                # --- 局部磨损：区域内降低对比度 + 模糊 ---
                cx = random.randint(int(w * 0.2), int(w * 0.8))
                cy = random.randint(int(h * 0.2), int(h * 0.8))
                rx = random.randint(int(w * 0.05), int(w * 0.15))
                ry = random.randint(int(h * 0.05), int(h * 0.15))

                x1, x2 = max(0, cx - 3*rx), min(w, cx + 3*rx)
                y1, y2 = max(0, cy - 3*ry), min(h, cy + 3*ry)
                if x2 <= x1 or y2 <= y1: continue

                Y, X = np.ogrid[y1:y2, x1:x2]
                mask = np.exp(-((X - cx) ** 2 / (2 * rx ** 2) + (Y - cy) ** 2 / (2 * ry ** 2)))
                mask = (mask * severity).astype(np.float32)

                if roi_mask is not None:
                    mask *= (roi_mask[y1:y2, x1:x2] / 255.0).astype(np.float32)

                region = result[y1:y2, x1:x2]
                local_mean = cv2.GaussianBlur(region, (21, 21), 5)
                blurred = cv2.GaussianBlur(region, (5, 5), 1.5)

                for c in range(3):
                    # 降对比度与模糊
                    ch_region = region[:, :, c]
                    ch_region = ch_region * (1 - mask * 0.5) + local_mean[:, :, c] * mask * 0.5
                    ch_region = ch_region * (1 - mask * 0.3) + blurred[:, :, c] * mask * 0.3
                    region[:, :, c] = ch_region
                
                result[y1:y2, x1:x2] = region

            elif effect == "grain_noise":
                # --- 颗粒噪声：局部高斯噪声 ---
                cx = random.randint(int(w * 0.15), int(w * 0.85))
                cy = random.randint(int(h * 0.15), int(h * 0.85))
                radius = random.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.15))

                x1, x2 = max(0, cx - 3*radius), min(w, cx + 3*radius)
                y1, y2 = max(0, cy - 3*radius), min(h, cy + 3*radius)
                if x2 <= x1 or y2 <= y1: continue

                Y, X = np.ogrid[y1:y2, x1:x2]
                mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * radius ** 2))
                mask = (mask * severity).astype(np.float32)

                if roi_mask is not None:
                    mask *= (roi_mask[y1:y2, x1:x2] / 255.0).astype(np.float32)

                # 只生成边界框内的小块噪音数组，防炸内存
                noise = np.random.randn(y2 - y1, x2 - x1, 3).astype(np.float32) * 25
                for c in range(3):
                    result[y1:y2, x1:x2, c] += noise[:, :, c] * mask

            elif effect == "corrosion_spots":
                # --- 腐蚀斑点：随机小圆点簇 ---
                # 这一块无需修改，已经是轻量级 cv2.circle 绘制了
                num_spots = random.randint(5, 20)
                cluster_cx = random.randint(int(w * 0.2), int(w * 0.8))
                cluster_cy = random.randint(int(h * 0.2), int(h * 0.8))
                spread = random.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.15))

                for _ in range(num_spots):
                    sx = cluster_cx + random.randint(-spread, spread)
                    sy = cluster_cy + random.randint(-spread, spread)
                    sx = max(2, min(sx, w - 3))
                    sy = max(2, min(sy, h - 3))

                    if roi_mask is not None and roi_mask[sy, sx] == 0:
                        continue

                    spot_r = random.randint(1, 4)
                    local_color = result[sy, sx].copy()
                    spot_color = local_color * random.uniform(0.3, 0.6)
                    spot_color[0] = min(spot_color[0] + random.uniform(10, 30), 255)

                    cv2.circle(result, (sx, sy), spot_r, spot_color.tolist(), -1, cv2.LINE_AA)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    # ==================================================================
    #  缺陷类型 4：传统 CutPaste（保留泊松融合）
    # ==================================================================
    def _generate_cutpaste(self, image: Image.Image,
                           source_pool: Optional[List[Image.Image]] = None) -> Image.Image:
        """传统 CutPaste：裁剪 + 变换 + 泊松融合粘贴"""
        result = image.copy()

        if source_pool and random.random() < 0.3:
            source = random.choice(source_pool)
        else:
            source = image

        w, h = source.size
        # 裁剪
        area = w * h
        patch_area = random.uniform(0.02, 0.12) * area
        aspect = random.uniform(0.3, 3.0)
        ph = max(1, int(math.sqrt(patch_area / aspect)))
        pw = max(1, int(patch_area / ph))
        pw, ph = min(pw, w), min(ph, h)
        px, py = random.randint(0, w - pw), random.randint(0, h - ph)
        patch = source.crop((px, py, px + pw, py + ph))

        # 变换
        angle = random.uniform(0, 360)
        patch = patch.rotate(angle, resample=Image.BICUBIC, expand=True)
        scale = random.uniform(0.7, 1.5)
        new_w = max(1, int(patch.width * scale))
        new_h = max(1, int(patch.height * scale))
        patch = patch.resize((new_w, new_h), Image.BICUBIC)

        # 颜色抖动
        for enhancer_cls, rng in [
            (ImageEnhance.Brightness, 0.3),
            (ImageEnhance.Contrast, 0.3),
            (ImageEnhance.Color, 0.2),
        ]:
            factor = random.uniform(1 - rng, 1 + rng)
            patch = enhancer_cls(patch).enhance(factor)

        # 粘贴（泊松融合）
        tw, th = result.size
        ppw, pph = patch.size
        if ppw > tw or pph > th:
            ppw, pph = min(ppw, tw), min(pph, th)
            patch = patch.resize((ppw, pph), Image.BICUBIC)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        # ROI 感知粘贴位置
        roi_mask = self._get_roi_region(image)
        paste_x, paste_y = 0, 0
        for _ in range(50):
            paste_x = random.randint(0, tw - ppw)
            paste_y = random.randint(0, th - pph)
            if roi_mask is None:
                break
            region = roi_mask[paste_y:paste_y + pph, paste_x:paste_x + ppw]
            if np.sum(region > 0) / region.size >= 0.6:
                break

        if self.use_poisson_blend:
            try:
                target_cv = np.array(result)
                patch_cv = np.array(patch)
                mask_cv = np.ones(patch_cv.shape[:2], dtype=np.uint8) * 255
                center = (
                    max(ppw // 2, min(paste_x + ppw // 2, tw - ppw // 2 - 1)),
                    max(pph // 2, min(paste_y + pph // 2, th - pph // 2 - 1))
                )
                blended = cv2.seamlessClone(patch_cv, target_cv, mask_cv, center, cv2.NORMAL_CLONE)
                result = Image.fromarray(blended)
            except cv2.error:
                result.paste(patch, (paste_x, paste_y))
        else:
            result.paste(patch, (paste_x, paste_y))

        return result

    # ==================================================================
    #  公开接口：单张增强
    # ==================================================================
    def augment_single(
        self,
        image: Image.Image,
        source_pool: Optional[List[Image.Image]] = None
    ) -> Image.Image:
        """
        对单张图像执行缺陷模拟

        根据 defect_mode 选择不同的缺陷生成策略：
        - combined 模式下随机组合 1~3 种缺陷类型
        """
        mode = self.defect_mode

        if mode == "combined":
            # 随机选择 1~3 种缺陷类型叠加（剔除导致生硬边界的传统 cutpaste）
            available = ["scratch", "lighting", "texture"]
            chosen = random.sample(available, k=random.randint(1, 3))
        else:
            chosen = [mode]

        result = image.copy()
        for defect_type in chosen:
            if defect_type == "scratch":
                result = self._generate_scratch(result)
            elif defect_type == "lighting":
                result = self._generate_lighting_defect(result)
            elif defect_type == "texture":
                result = self._generate_texture_defect(result)
            elif defect_type == "cutpaste":
                result = self._generate_cutpaste(result, source_pool)

        return result, chosen

    # ==================================================================
    #  批量增强
    # ==================================================================
    def augment_batch(
        self,
        image_dir: str,
        csv_path: str,
        output_dir: str,
        num_augment_per_image: int = 3,
        target_label: str = "[无缺陷]",
        augmented_label: str = "[有缺陷]",
        augment_all: bool = False
    ) -> Dict[str, int]:
        """批量生成伪缺陷图像"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. 读取 CSV ---
        target_entries = []
        all_entries = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                all_entries.append(row)
                if len(row) >= 2:
                    if augment_all or row[1].strip() == target_label:
                        target_entries.append(row)

        logger.info(f"📊 CSV 总: {len(all_entries)}, 待增强样本: {len(target_entries)}")

        if not target_entries:
            logger.warning("⚠️ 未找到符合条件的样本")
            return {"total_source": 0, "total_generated": 0}

        # --- 2. 加载正常图像池 (仅传统 CutPaste 模式需要，其他模式省去此步避免爆内存) ---
        source_pool = []
        if self.defect_mode == "cutpaste":
            logger.info("🔄 正在预加载正常图像池 (用于裁剪)...")
            normal_only = [r for r in all_entries if len(r) >= 2 and r[1].strip() == target_label]
            for entry in normal_only:
                img_path = image_dir / entry[0]
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        source_pool.append(img)
                    except Exception as e:
                        pass
            logger.info(f"✅ 已加载 {len(source_pool)} 张背景 | 模式: {self.defect_mode}")
        else:
            logger.info("⚡ 当前生成模式无需背景池，直接跳过内存预加载并开始增强！")

        # --- 3. 批量生成 ---
        generated_count = 0
        new_entries = []

        for idx, entry in enumerate(target_entries):
            img_name = entry[0]
            img_path = image_dir / img_name
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"⚠️ 加载失败: {img_path} -> {e}")
                continue

            for aug_idx in range(num_augment_per_image):
                aug_image, chosen_modes = self.augment_single(image, source_pool)

                stem = Path(img_name).stem
                ext = Path(img_name).suffix
                aug_name = f"{stem}_aug_{self.defect_mode}_{aug_idx}{ext}"
                aug_path = output_dir / aug_name
                aug_image.save(aug_path, quality=95)

                # Determine if a defect was injected
                is_defective = any(mode in ["scratch", "cutpaste"] for mode in chosen_modes)
                final_label = augmented_label if is_defective else target_label

                new_entries.append([aug_name, final_label])
                generated_count += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"  ⏳ {idx + 1}/{len(target_entries)} (已生成 {generated_count})")

        # --- 4. 输出 CSV ---
        output_csv = output_dir / "augmented.csv"
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in all_entries:
                writer.writerow(row)
            for row in new_entries:
                writer.writerow(row)

        logger.info(
            f"\n{'='*60}\n"
            f"✅ 缺陷模拟完成！\n"
            f"   模式:    {self.defect_mode}\n"
            f"   增强样本: {len(target_entries)}\n"
            f"   生成数:  {generated_count}\n"
            f"   输出:    {output_dir}\n"
            f"{'='*60}"
        )

        return {
            "total_source": len(target_entries),
            "total_generated": generated_count,
            "csv_path": str(output_csv)
        }
