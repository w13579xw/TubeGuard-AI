#!/usr/bin/env python
# -*- coding:utf-8 -*-


# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import cv2
import pywt
import numpy as np
import shutil
from tqdm import tqdm


def wavelet_enhancement_inplace(image_path, wavelet='db1',
                                enhance_lh=10.0, enhance_hl=10.0, enhance_hh=2.0,
                                levels=1):
    """
    读取图片 -> 小波增强 -> 转回3通道 -> 原地覆盖保存
    """
    # 1. 以灰度模式读取
    # 注意：即使原图是RGB，小波变换通常在单通道上处理效果最好
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ 无法读取: {image_path}")
        return False

    # 2. 执行小波分解
    try:
        coeffs = pywt.wavedec2(img, wavelet, level=levels)
    except Exception as e:
        print(f"⚠️ 小波分解失败 ({image_path}): {e}")
        return False

    # 3. 增强高频分量 (LH, HL, HH)
    coeffs_enhanced = [coeffs[0]]  # 保留低频近似分量

    for i in range(1, len(coeffs)):
        LH, HL, HH = coeffs[i]

        # 应用增强系数
        LH_enhanced = LH * enhance_lh
        HL_enhanced = HL * enhance_hl
        HH_enhanced = HH * enhance_hh

        coeffs_enhanced.append((LH_enhanced, HL_enhanced, HH_enhanced))

    # 4. 小波重构
    img_enhanced = pywt.waverec2(coeffs_enhanced, wavelet)

    # 5. 裁剪与归一化
    # 确保数值在 0-255 之间
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    # 处理重构后尺寸可能变大的情况 (padding)
    if img_enhanced.shape != img.shape:
        img_enhanced = img_enhanced[:img.shape[0], :img.shape[1]]

    # 6. [关键] 转回 3 通道 BGR 格式
    # 训练代码通常期待 3 通道输入，如果保存为单通道灰度图，
    # PyTorch 的 ImageFolder 加载时可能会有问题，或者 Normalize 时报错。
    # 这里我们把增强后的灰度图复制 3 份，伪装成 RGB。
    img_enhanced_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)

    # 7. 原地覆盖保存
    cv2.imwrite(image_path, img_enhanced_bgr)
    return True


def process_dataset(data_root):
    """
    遍历数据集并处理所有图片
    """
    if not os.path.exists(data_root):
        print(f"❌ 错误: 找不到数据集目录 '{data_root}'")
        return

    # 支持的图片扩展名
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 收集所有图片路径
    image_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                image_files.append(os.path.join(root, file))

    print(f"📂 扫描到 {len(image_files)} 张图片，准备进行小波增强...")
    print(f"⚙️ 参数: LHx10 (水平), HLx10 (垂直), HHx2 (对角)")

    # 再次确认
    confirm = input("⚠️ 警告: 此操作将永久修改原文件！是否继续？(y/n): ")
    if confirm.lower() != 'y':
        print("已取消操作。")
        return

    # 进度条处理
    success_count = 0
    fail_count = 0

    for img_path in tqdm(image_files, desc="Processing Images"):
        # 使用你指定的【方案2】强增强参数
        result = wavelet_enhancement_inplace(
            img_path,
            wavelet='db1',
            enhance_lh=10.0,  # 强增强水平
            enhance_hl=10.0,  # 强增强垂直
            enhance_hh=2.0,  # 适度增强对角
            levels=1
        )

        if result:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 50)
    print("✅ 处理完成！")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print("=" * 50)


if __name__ == "__main__":
    # 数据集根目录
    DATASET_DIR = "yolo_cls_data"

    # 建议先备份（可选）
    # backup_dir = DATASET_DIR + "_backup"
    # if not os.path.exists(backup_dir):
    #     print(f"正在创建备份 {backup_dir} ...")
    #     shutil.copytree(DATASET_DIR, backup_dir)

    process_dataset(DATASET_DIR)