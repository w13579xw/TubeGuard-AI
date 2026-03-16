import os
import sys
import glob
import csv
from pathlib import Path

# 添加 YOLOv10-TPH 代码目录到 Python 路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
yolo_tph_dir = os.path.join(curr_dir, "NN", "yolov10_tph")
sys.path.insert(0, yolo_tph_dir)

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib

# 防止服务器报错
matplotlib.use('Agg')

# 导入 YOLOv10-TPH 模型架构和工具
from model import YOLOv10TPHClassifier
from utils import GradCAM, save_heatmap_result


def main():
    # 配置路径
    input_dir = Path("data/defect_test")
    output_dir = Path("data/defect_test_heatmaps")
    csv_out_path = output_dir / "predictions.csv"
    
    # 优先使用根目录下的权重，如果没有则使用 NN 目录下的
    weights_path = Path("TubeGuard_GFC_System/weights/yolov10_tph_best.pth")
    if not weights_path.exists():
        weights_path = Path("NN/yolov10_tph/processed/yolov10_tph_best.pth")
        
    if not weights_path.exists():
        print(f"❌ 找不到权重文件！尝试路径: {weights_path}")
        return

    if not input_dir.exists():
        print(f"❌ 输入目录 {input_dir} 不存在！请先生成缺陷测试数据。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 设备与预处理配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 1280

    print(f"🚀 初始化 YOLOv10-TPH 分类器 (设备: {device})...")
    # model_weight="yolov10n.pt" 用于初始化骨干，实际加载我们提供的最佳权重
    model = YOLOv10TPHClassifier(model_weight='TubeGuard_GFC_System/weights/yolov10n.pt', num_classes=2).to(device)
    
    # 获取特征层，并注册 GradCAM
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # 加载已训练好的 TPH 权重
    print(f"📦 加载权重: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 收集图像
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]
    
    print(f"🔍 找到 {len(image_paths)} 张测试图像。开始批量识别和热力图生成...")

    # 3. 批量推理跑图
    label_map = {0: 'Defective', 1: 'Good'}
    
    # 建立一个记录列表
    csv_data = [["ImageName", "Prediction", "Confidence"]]
    
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            raw_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 图片读取失败: {img_path} -> {e}")
            continue

        # 前向传播 (需要 requires_grad 计算热力图)
        input_tensor = transform(raw_img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # 使用 GradCAM
        heatmap, pred_idx, raw_output = grad_cam(input_tensor)

        # 处理结果
        probs = torch.softmax(raw_output, dim=1)
        confidence = probs[0][pred_idx].item()
        
        # 结果文本标签
        res_label = label_map.get(pred_idx, "Unknown")
        
        # 保存到 csv_data
        csv_data.append([img_path.name, res_label, f"{confidence:.4f}"])
        
        # 保存路径
        save_name = str(output_dir / f"CAM_{img_path.name}")
        
        # 保存带有文本和预测的热力图
        formatted_label = f"有缺陷 ({res_label})" if pred_idx == 0 else f"无缺陷 ({res_label})"
        save_heatmap_result(str(img_path), heatmap, save_name, formatted_label, confidence)

    # 4. 写入 CSV 文件
    with open(csv_out_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"\n✅ 所有热力图已生成并保存至: {output_dir}")
    print(f"✅ 所有评估数据（标签和置信度）已汇总至: {csv_out_path}")

if __name__ == "__main__":
    main()
