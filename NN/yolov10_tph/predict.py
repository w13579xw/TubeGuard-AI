import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib

# 防止 PyCharm/服务器报错
matplotlib.use('Agg')


# [导入] 1. 模型架构  2. GradCAM工具
from model import YOLOv10TPHClassifier
from utils import GradCAM, save_heatmap_result


def run_inference(image_path, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 1280

    # 1. 构建模型
    if not os.path.exists('yolov10n.pt'):
        print("提示: 正在尝试自动下载/加载 yolov10n.pt ...")

    model = YOLOv10TPHClassifier(model_weight='yolov10n.pt', num_classes=2).to(device)

    # 2. 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("-> 权重加载成功")
    else:
        print(f"❌ 找不到权重: {model_path}");
        return

    model.eval()

    # 3. 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ 图片读取失败: {e}")
        return

    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 4. 使用 utils 中的 GradCAM
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    heatmap, pred_idx, raw_output = grad_cam(input_tensor)

    # 5. 处理结果
    probs = torch.softmax(raw_output, dim=1)
    confidence = probs[0][pred_idx].item()

    label_map = {0: '有缺陷 (Defective)', 1: '无缺陷 (Good)'}
    result_text = label_map.get(pred_idx, "Unknown")

    print(f"预测: {result_text} | 置信度: {confidence:.4f}")

    # 6. 保存热力图
    save_name = f"CAM_{os.path.basename(image_path)}"
    save_heatmap_result(image_path, heatmap, save_name, result_text, confidence)


if __name__ == "__main__":
    # 请修改这里的路径
    TRAINED_WEIGHTS = 'yolov10_tph_best.pth'
    TEST_IMAGE = '../data/images/68.jpg'

    run_inference(TEST_IMAGE, TRAINED_WEIGHTS)