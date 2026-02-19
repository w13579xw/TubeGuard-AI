#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
TubeGuard-AI Main Pipeline (GFC System Entry)
---------------------------------------------
主程序入口。
负责串联整个检测流程：
1. 读取配置 (Config)
2. 初始化各阶段处理器 (Stage A, Stage B)
3. 遍历数据目录进行批量检测
4. 调度数据流向 (Pass -> Stage B, NG -> Trash)
"""

import os
import yaml
import sys
from tqdm import tqdm

# 添加 src 到路径以便导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stage_a_geometric import StageAGeometricScreener
from src.stage_b_semantic import StageBSemanticAnalyzer
from utils.image_io import load_image, save_debug_image
from utils.logger import setup_logger

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. 初始化
    logger = setup_logger()
    logger.info("Starting TubeGuard GFC System...")
    
    config = load_config("config/gfc_config.yaml")
    
    # 2. 实例化处理器
    stage_a = StageAGeometricScreener(config)
    stage_b = StageBSemanticAnalyzer(config)
    
    # 3. 准备数据路径
    input_dir = "data/raw_images"
    pass_dir = "data/stage_a_pass"
    ng_dir = "data/stage_a_ng"
    
    # 模拟获取文件列表
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))] if os.path.exists(input_dir) else []
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}. Please add images to test.")
        # Create a dummy file for testing flow if directory exists
        # ...
    
    logger.info(f"Found {len(image_files)} images to process.")
    
    # 4. 批处理循环
    for img_name in tqdm(image_files):
        img_path = os.path.join(input_dir, img_name)
        image = load_image(img_path)
        
        if image is None:
            continue
            
        # --- Stage A: 几何初筛 ---
        res_a = stage_a.process(image)
        
        if not res_a['is_pass']:
            # 几何 NG -> 熔断
            logger.info(f"[NG] {img_name} - Reason: {res_a['reason']}")
            save_debug_image(image, ng_dir, img_name) # 实际应用中可能不需要保存原图，只需记录
            continue
            
        # 几何 OK -> 进入 Stage B
        # logger.info(f"[Pass Stage A] {img_name} - Metrics: {res_a['metrics']}")
        
        # --- Stage B: 纹理与分类 ---
        # 传入 Stage A 的 Mask 避免背景干扰
        res_b = stage_b.process(image, res_a['mask'])
        
        if res_b['is_clean']:
            # 最终 OK
            # logger.info(f"[OK] {img_name} is Perfect.")
            save_debug_image(image, pass_dir, img_name)
        else:
            # 纹理 NG
            logger.info(f"[Defect] {img_name} - Found: {res_b['defects']}")
            # 可以保存到单独的 defect 目录

    logger.info("Inspection Batch Completed.")

if __name__ == "__main__":
    main()
