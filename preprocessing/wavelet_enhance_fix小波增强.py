#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import pywt
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [关键修改] 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def wavelet_enhancement(image_path, wavelet='db1',
                        enhance_lh=2.5, enhance_hl=2.5, enhance_hh=1.5,
                        levels=1):
    """
    基于小波变换的图像增强 - 重点增强边缘细节

    参数:
        image_path: 图像路径
        wavelet: 小波基函数 ('db1', 'haar', 'sym2' 等)
        enhance_lh: LH分量增强系数 (水平边缘)
        enhance_hl: HL分量增强系数 (垂直边缘)
        enhance_hh: HH分量增强系数 (对角边缘)
        levels: 小波分解层数
    """
    # 1. 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 错误：无法找到图像 {image_path}")
        print("-> 生成随机噪声图用于演示...")
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    print(f"原始图像尺寸: {img.shape}")
    print(f"小波基函数: {wavelet}, 分解层数: {levels}")
    print(f"增强系数 - LH(水平边缘): {enhance_lh}, HL(垂直边缘): {enhance_hl}, HH(对角边缘): {enhance_hh}")

    # 2. 执行小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=levels)

    # 3. 重点增强 LH 和 HL 高频细节分量
    coeffs_enhanced = [coeffs[0]]  # 保留最低频的近似分量

    for i in range(1, len(coeffs)):
        LH, HL, HH = coeffs[i]

        # 重点加强水平和垂直边缘
        LH_enhanced = LH * enhance_lh  # 水平边缘增强
        HL_enhanced = HL * enhance_hl  # 垂直边缘增强
        HH_enhanced = HH * enhance_hh  # 对角边缘适度增强

        coeffs_enhanced.append((LH_enhanced, HL_enhanced, HH_enhanced))

        print(f"  第{i}层: LH×{enhance_lh}, HL×{enhance_hl}, HH×{enhance_hh}")

    # 4. 小波重构
    img_enhanced = pywt.waverec2(coeffs_enhanced, wavelet)

    # 5. 归一化到 [0, 255]
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    # 确保尺寸一致
    if img_enhanced.shape != img.shape:
        img_enhanced = img_enhanced[:img.shape[0], :img.shape[1]]

    print(f"增强后图像尺寸: {img_enhanced.shape}")

    # 6. 可视化对比
    visualize_enhancement(img, img_enhanced, coeffs, coeffs_enhanced,
                          image_path, enhance_lh, enhance_hl, enhance_hh)

    # 7. 保存增强后的图像
    save_name = image_path.replace('.jpg', '').replace('.png', '') + '_enhanced.jpg'
    cv2.imwrite(save_name, img_enhanced)
    print(f"✅ 增强后图像已保存为: {save_name}")

    return img_enhanced


def visualize_enhancement(img_original, img_enhanced, coeffs_before,
                          coeffs_after, image_path, enhance_lh, enhance_hl, enhance_hh):
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
    axes[0, 1].set_title(f'Enhanced Image\nLH x{enhance_lh}, HL x{enhance_hl}',
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 差异图
    diff = cv2.absdiff(img_original, img_enhanced)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 第二行：增强前的高频分量
    axes[1, 0].imshow(np.abs(LH_before), cmap='gray', vmin=0, vmax=np.max(np.abs(LH_before)))
    axes[1, 0].set_title('LH - Horizontal Edge (Before)', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.abs(HL_before), cmap='gray', vmin=0, vmax=np.max(np.abs(HL_before)))
    axes[1, 1].set_title('HL - Vertical Edge (Before)', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.abs(HH_before), cmap='gray', vmin=0, vmax=np.max(np.abs(HH_before)))
    axes[1, 2].set_title('HH - Diagonal Edge (Before)', fontsize=10)
    axes[1, 2].axis('off')

    # 第三行：增强后的高频分量
    axes[2, 0].imshow(np.abs(LH_after), cmap='gray', vmin=0, vmax=np.max(np.abs(LH_after)))
    axes[2, 0].set_title(f'LH - After (x{enhance_lh})', fontsize=10, color='red', fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.abs(HL_after), cmap='gray', vmin=0, vmax=np.max(np.abs(HL_after)))
    axes[2, 1].set_title(f'HL - After (x{enhance_hl})', fontsize=10, color='red', fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(np.abs(HH_after), cmap='gray', vmin=0, vmax=np.max(np.abs(HH_after)))
    axes[2, 2].set_title(f'HH - After (x{enhance_hh})', fontsize=10)
    axes[2, 2].axis('off')

    plt.tight_layout()

    # 保存可视化结果
    save_name = image_path.replace('.jpg', '').replace('.png', '') + '_comparison.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存为: {save_name}")
    plt.close()


def edge_enhanced_wavelet(image_path, wavelet='db1'):
    """
    边缘增强版小波变换 - 极致加强 LH 和 HL
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 错误：无法找到图像 {image_path}")
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    print("\n[边缘增强模式] 极致增强水平和垂直边缘")

    # 多级小波分解
    coeffs = pywt.wavedec2(img, wavelet, level=2)

    # 极致增强策略：重点加强 LH 和 HL
    coeffs_enhanced = [coeffs[0]]  # 保留低频

    # 第一层（细节最丰富）：极致增强边缘
    LH1, HL1, HH1 = coeffs[1]
    coeffs_enhanced.append((
        LH1 * 3.5,  # 水平边缘极致增强
        HL1 * 3.5,  # 垂直边缘极致增强
        HH1 * 1.8  # 对角边缘适度增强
    ))

    # 第二层（中等细节）：中等增强
    LH2, HL2, HH2 = coeffs[2]
    coeffs_enhanced.append((
        LH2 * 2.0,  # 水平边缘中等增强
        HL2 * 2.0,  # 垂直边缘中等增强
        HH2 * 1.2  # 对角边缘轻度增强
    ))

    # 重构
    img_enhanced = pywt.waverec2(coeffs_enhanced, wavelet)
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    if img_enhanced.shape != img.shape:
        img_enhanced = img_enhanced[:img.shape[0], :img.shape[1]]

    # 保存
    save_name = image_path.replace('.jpg', '').replace('.png', '') + '_edge_enhanced.jpg'
    cv2.imwrite(save_name, img_enhanced)
    print(f"✅ 边缘增强图像已保存为: {save_name}")

    # 简单对比可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_enhanced, cmap='gray')
    axes[1].set_title('Edge Enhanced\n(LH x3.5, HL x3.5)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    diff = cv2.absdiff(img, img_enhanced)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Enhancement Effect', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    compare_name = image_path.replace('.jpg', '').replace('.png', '') + '_edge_comparison.png'
    plt.savefig(compare_name, dpi=150, bbox_inches='tight')
    print(f"✅ 边缘增强对比图已保存为: {compare_name}")
    plt.close()

    return img_enhanced


if __name__ == "__main__":
    img_path = "1.jpg"

    print("=" * 70)
    print("小波边缘增强处理 - 重点加强 LH 和 HL")
    print("=" * 70)

    # # 方案1：标准增强 - 重点加强 LH 和 HL
    # print("\n[方案1] 标准边缘增强 (LH×2.5, HL×2.5, HH×1.5)")
    # wavelet_enhancement(img_path, wavelet='db1',
    #                     enhance_lh=2.5, enhance_hl=2.5, enhance_hh=1.5,
    #                     levels=1)

    # 方案2：强增强 - 极致加强边缘
    print("\n[方案2] 强边缘增强 (LH×10, HL×10, HH×2.0)")
    wavelet_enhancement(img_path, wavelet='db1',
                        enhance_lh=10, enhance_hl=10, enhance_hh=2,
                        levels=1)

    # # 方案3：多级自适应边缘增强
    # print("\n[方案3] 多级自适应边缘增强")
    # edge_enhanced_wavelet(img_path, wavelet='db1')

    print("\n" + "=" * 70)
    print("✅ 所有处理完成！")
    print("说明:")
    print("  - LH (水平边缘): 检测左右方向的边缘")
    print("  - HL (垂直边缘): 检测上下方向的边缘")
    print("  - HH (对角边缘): 检测斜向边缘")
    print("  - 重点增强 LH 和 HL 可以显著提升管道、线条等结构的清晰度")
    print("=" * 70)