#!/usr/bin/env python
# -*- coding:utf-8 -*-

# main_pipeline.py
import cv2
from models.stage1_segformer import Stage1_GeometryScanner
from models.stage2_patchcore import Stage2_TextureScreener
from models.stage3_yolo import Stage3_DefectClassifier


class PipeInspectionSystem:
    def __init__(self):
        # 初始化所有模型
        self.stage1 = Stage1_GeometryScanner("weights/segformer_b3.pth")
        self.stage2 = Stage2_TextureScreener("weights/patchcore_memory.faiss")
        self.stage3 = Stage3_DefectClassifier("weights/yolov12_cbam.pt")

    def inspect(self, image_path):
        image = cv2.imread(image_path)
        print(f"\n--- Processing {image_path} ---")

        # === Stage 1: 几何初筛 ===
        mask, geo_pass, geo_metrics = self.stage1.process(image)

        if not geo_pass:
            return {
                "result": "NG",
                "reason": "Geometry Error",
                "details": geo_metrics
            }

        print(f"Stage 1 Pass. Geometry: {geo_metrics}")

        # === Stage 2: 纹理筛查 ===
        # 将 Stage 1 的 mask 传进去，避免背景干扰
        is_anomaly, suspicious_crops = self.stage2.process(image, mask)

        if not is_anomaly:
            return {
                "result": "OK",
                "reason": "Perfect Pipe",
                "details": geo_metrics
            }

        print(f"Stage 2 Warning. Found {len(suspicious_crops)} suspicious regions.")

        # === Stage 3: 精细分类 ===
        # 只有在 Stage 2 报警时才启动 YOLO
        defect_types = self.stage3.process(suspicious_crops)

        return {
            "result": "NG",
            "reason": "Defect Detected",
            "defects": defect_types,
            "details": geo_metrics
        }


# --- 运行测试 ---
if __name__ == "__main__":
    system = PipeInspectionSystem()

    # 模拟一张图片
    report = system.inspect("data/test_pipe.jpg")
    print("\nFinal Report:", report)
