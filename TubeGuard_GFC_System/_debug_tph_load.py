import torch
import sys
import os
import cv2
import numpy as np

# 添加当前目录到路径
sys.path.append(os.getcwd())

from models.yolov10_tph.yolov10_tph_model import YOLOv10TPHInference

def test_load():
    weight_path = "weights/yolov10_tph_best.pth"
    print(f"Checking weight file: {weight_path}")
    if not os.path.exists(weight_path):
        print("Error: Weight file not found!")
        return

    print("Initializing YOLOv10TPHInference...")
    try:
        classifier = YOLOv10TPHInference(weight_path=weight_path)
        if classifier.model is None:
            print("Error: classifier.model is None after initialization")
            return
        print("Success: Model loaded.")
        
        # 模拟一枚 ROI
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        crops = [{'crop': dummy_img, 'box': [0, 0, 224, 224], 'score': 0.9}]
        
        print("Running prediction on dummy image...")
        results = classifier.predict(crops)
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
