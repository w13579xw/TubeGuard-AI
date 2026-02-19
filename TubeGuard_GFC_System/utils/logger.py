#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Logger Utility
--------------
简单的日志记录器，支持输出到控制台和文件。
"""

import logging
import os
import datetime

def setup_logger(log_dir="logs", log_name="gfc_system"):
    """
    配置全局 Logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.info("Logger initialized.")
    return logger
