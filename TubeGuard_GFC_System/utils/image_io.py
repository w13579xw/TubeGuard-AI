#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Image I/O Utilities
-------------------
负责图像读取、预处理、格式转换与保存。
"""

import cv2
import os

def load_image(path, to_gray=False):
    """
    安全读取图像
    :param path: 图像路径
    :param to_gray: 是否转为灰度
    :return: np.ndarray or None
    """
    if not os.path.exists(path):
        print(f"[Error] Image not found: {path}")
        return None
    
    img = cv2.imread(path)
    if img is None:
        print(f"[Error] Failed to decode image: {path}")
        return None
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    return img

def save_debug_image(image, output_dir, filename, metadata=None):
    """
    保存调试图像 (可叠加 metadata 文字)
    :param image: 图像数据
    :param output_dir: 输出目录 (e.g., data/debug_viz)
    :param filename: 文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, filename)
    
    # TODO: Draw metadata text on image if provided
    
    cv2.imwrite(save_path, image)
    # print(f"[Debug] Saved to {save_path}")
