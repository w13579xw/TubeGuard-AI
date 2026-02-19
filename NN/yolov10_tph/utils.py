# -*- coding:utf-8 -*-
"""
utils.py
存放通用工具类：
1. GradCAM: 用于生成模型关注区域的热力图
2. Visualization: 图像叠加与保存工具
"""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap, class_idx, output


def save_heatmap_result(img_path, heatmap, save_name, label_text=None, prob=None):
    """
    将热力图叠加到原图，并保存
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return

    # 1. 调整热力图大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # 转换为 0-255 的整数
    heatmap = np.uint8(255 * heatmap)

    # 2. 应用伪彩色
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 3. 叠加 (结果会变成 float64)
    superimposed = heatmap_colored * 0.4 + img * 0.6

    # [关键修复] 4. 强制转换回 uint8 (0-255 整数)，解决 OpenCV 警告
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    # 5. 在图上写字
    if label_text and prob is not None:
        display_text = label_text

        # [核心修改] 自动提取括号内的内容
        # 如果输入是 "有缺陷 (Defective)" -> 提取出 "Defective"
        if "(" in label_text and ")" in label_text:
            try:
                start = label_text.find("(") + 1
                end = label_text.find(")")
                display_text = label_text[start:end]
            except:
                pass  # 如果提取失败，保持原样
        else:
            # 如果没有括号，过滤掉所有非 ASCII 字符 (即去掉中文)
            display_text = "".join([c for c in label_text if ord(c) < 128]).strip()

        # 拼接置信度
        text = f"{display_text}: {prob:.2%}"

        # 绘制黑色描边 (增强对比度)
        cv2.putText(superimposed, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        # 绘制白色正文
        cv2.putText(superimposed, text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(save_name, superimposed)
    print(f"✅ 热力图已保存: {save_name}")