#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import pywt
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wavelet_enhancement(image_path, wavelet='db1', enhance_factor=2.0, levels=1):
    """
    基于小波变换的图像增强

    参数:
        image_path: 图像路径
        wavelet: 小波基函数 ('db1', 'haar', 'sym2' 等)
        enhance_factor: 高频增强系数 (>1 增强细节，<1 平滑)
        levels: 小波分解层数
    """
    # 1. 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 错误：无法找到图像 {image_path}")
        print("-> 生成随机噪声图用于演示...")
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    print(f"原始图像尺寸: {img.shape}")
    print(f"小波基函数: {wavelet}, 增强系数: {enhance_factor}, 分解层数: {levels}")

    # 2. 执行小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=levels)

    # coeffs 结构: [LL_n, (LH_n, HL_n, HH_n), ..., (LH_1, HL_1, HH_1)]
    # LL: 低频近似分量 (保留)
    # LH, HL, HH: 高频细节分量 (增强)

    # 3. 增强高频细节分量
    coeffs_enhanced = [coeffs[0]]  # 保留最低频的近似分量

    for i in range(1, len(coeffs)):
        # 对每一层的三个高频分量进行增强
        LH, HL, HH = coeffs[i]
        LH_enhanced = LH * enhance_factor
        HL_enhanced = HL * enhance_factor
        HH_enhanced = HH * enhance_factor
        coeffs_enhanced.append((LH_enhanced, HL_enhanced, HH_enhanced))

    # 4. 小波重构
    img_enhanced = pywt.waverec2(coeffs_enhanced, wavelet)

    # 5. 归一化到 [0, 255]
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    # 确保尺寸一致 (有时重构会有1-2像素偏差)
    if img_enhanced.shape != img.shape:
        img_enhanced = img_enhanced[:img.shape[0], :img.shape[1]]

    print(f"增强后图像尺寸: {img_enhanced.shape}")

    # 6. 可视化对比
    visualize_enhancement(img, img_enhanced, coeffs, coeffs_enhanced,
                          image_path, enhance_factor)

    # 7. 保存增强后的图像
    save_name = image_path.replace('.jpg', '') + '_enhanced.jpg'
    cv2.imwrite(save_name, img_enhanced)
    print(f"✅ 增强后图像已保存为: {save_name}")

    return img_enhanced


def visualize_enhancement(img_original, img_enhanced, coeffs_before,
                          coeffs_after, image_path, enhance_factor):
    """
    可视化原图、增强图以及小波系数对比
    """
    # 提取第一层小波系数
    LL_before = coeffs_before[0]
    LH_before, HL_before, HH_before = coeffs_before[1]

    LL_after = coeffs_after[0]
    LH_after, HL_after, HH_after = coeffs_after[1]

    # 创建对比图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 第一行：原图和增强图
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_enhanced, cmap='gray')
    axes[0, 1].set_title(f'Enhanced Image (×{enhance_factor})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 差异图
    diff = cv2.absdiff(img_original, img_enhanced)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 第二行：增强前的高频分量
    axes[1, 0].imshow(LH_before, cmap='gray')
    axes[1, 0].set_title('LH (Before)', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(HL_before, cmap='gray')
    axes[1, 1].set_title('HL (Before)', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(HH_before, cmap='gray')
    axes[1, 2].set_title('HH (Before)', fontsize=10)
    axes[1, 2].axis('off')

    # 第三行：增强后的高频分量
    axes[2, 0].imshow(LH_after, cmap='gray')
    axes[2, 0].set_title(f'LH (After ×{enhance_factor})', fontsize=10)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(HL_after, cmap='gray')
    axes[2, 1].set_title(f'HL (After ×{enhance_factor})', fontsize=10)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(HH_after, cmap='gray')
    axes[2, 2].set_title(f'HH (After ×{enhance_factor})', fontsize=10)
    axes[2, 2].axis('off')

    plt.tight_layout()

    # 保存可视化结果
    save_name = image_path.replace('.jpg', '') + '_comparison.png'
    plt.savefig(save_name, dpi=150)
    print(f"✅ 对比图已保存为: {save_name}")
    plt.close()


def adaptive_enhancement(image_path, wavelet='db1'):
    """
    自适应小波增强：对不同频率分量使用不同增强系数
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 错误：无法找到图像 {image_path}")
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    # 多级小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=2)

    # 自适应增强策略
    coeffs_enhanced = [coeffs[0]]  # 保留低频

    # 第一层（细节最丰富）：强增强
    LH1, HL1, HH1 = coeffs[1]
    coeffs_enhanced.append((LH1 * 2.5, HL1 * 2.5, HH1 * 1.8))

    # 第二层（中等细节）：中等增强
    LH2, HL2, HH2 = coeffs[2]
    coeffs_enhanced.append((LH2 * 1.5, HL2 * 1.5, HH2 * 1.2))

    # 重构
    img_enhanced = pywt.waverec2(coeffs_enhanced, wavelet)
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    if img_enhanced.shape != img.shape:
        img_enhanced = img_enhanced[:img.shape[0], :img.shape[1]]

    # 保存
    save_name = image_path.replace('.jpg', '') + '_adaptive_enhanced.jpg'
    cv2.imwrite(save_name, img_enhanced)
    print(f"✅ 自适应增强图像已保存为: {save_name}")

    return img_enhanced


if __name__ == "__main__":
    img_path = "1.jpg"

    print("=" * 60)
    print("小波增强处理")
    print("=" * 60)

    # 方案1：标准增强
    print("\n[方案1] 标准小波增强 (增强系数=2.0)")
    wavelet_enhancement(img_path, wavelet='db1', enhance_factor=2.0, levels=1)

    # 方案2：强增强
    print("\n[方案2] 强增强 (增强系数=3.0)")
    wavelet_enhancement(img_path, wavelet='db1', enhance_factor=3.0, levels=1)

    # 方案3：自适应增强
    print("\n[方案3] 自适应多级增强")
    adaptive_enhancement(img_path, wavelet='db1')

    print("\n" + "=" * 60)
    print("✅ 所有处理完成！")
    print("=" * 60)