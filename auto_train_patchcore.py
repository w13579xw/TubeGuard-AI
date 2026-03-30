import os
import csv
import shutil
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.spatial.distance import cdist
import time

"""
🚀 CII 顶刊强对标实验: 无监督工业 SOTA 跨界对推
纯 PyTorch 实现 PatchCore 算法，零依赖 anomalib，直接在医疗管线数据集上验证。
【核心论点】：无监督会在复杂透明反光的医疗管线制造中因"环境噪光"产生极高的 FPR。

PatchCore 算法核心流程：
1. 使用预训练 WideResNet50 提取中间层特征 (layer2, layer3)
2. 将所有正常训练样本的 patch 特征聚合为 Memory Bank
3. 通过 Coreset Subsampling 压缩 Memory Bank
4. 测试时计算每个样本到 Memory Bank 的最近邻距离作为异常分数
"""


class PatchCoreModel:
    """纯 PyTorch 实现的 PatchCore 异常检测模型"""
    
    def __init__(self, backbone_name="wide_resnet50_2", layers=("layer2", "layer3"),
                 coreset_ratio=0.1, num_neighbors=9, device="cuda"):
        self.device = device
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.memory_bank = None
        
        # 加载预训练骨干网络
        print(f"🧠 正在加载预训练骨干网络: {backbone_name}...", flush=True)
        backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
        backbone.eval()
        self.backbone = backbone.to(device)
        
        # 注册 hook 提取中间层特征
        self.features = {}
        for layer_name in layers:
            layer = dict(backbone.named_children())[layer_name]
            layer.register_forward_hook(self._get_hook(layer_name))
            
        print(f"✅ 骨干网络加载完毕，提取层: {layers}", flush=True)
    
    def _get_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def _extract_features(self, dataloader):
        """从数据集中提取所有 patch 级别的特征"""
        all_features = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                _ = self.backbone(images)
                
                # 提取并拼接多层特征
                layer_features = []
                target_size = None
                
                for layer_name in self.layers:
                    feat = self.features[layer_name]  # (B, C, H, W)
                    if target_size is None:
                        target_size = feat.shape[2:]
                    else:
                        # 上采样对齐到最大的空间分辨率
                        feat = nn.functional.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                    layer_features.append(feat)
                
                # 拼接多层特征: (B, C_total, H, W)
                combined = torch.cat(layer_features, dim=1)
                B, C, H, W = combined.shape
                
                # 重排为 patch 特征: (B*H*W, C)
                patches = combined.permute(0, 2, 3, 1).reshape(-1, C)
                all_features.append(patches.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   特征提取进度: {batch_idx + 1} 批次已处理", flush=True)
        
        return np.concatenate(all_features, axis=0)
    
    def _extract_image_features(self, dataloader):
        """提取图像级别特征（用于测试时计算每张图的异常分数）"""
        all_scores = []
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                _ = self.backbone(images)
                
                layer_features = []
                target_size = None
                
                for layer_name in self.layers:
                    feat = self.features[layer_name]
                    if target_size is None:
                        target_size = feat.shape[2:]
                    else:
                        feat = nn.functional.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
                    layer_features.append(feat)
                
                combined = torch.cat(layer_features, dim=1)
                B, C, H, W = combined.shape
                
                # 对每张图计算其所有 patch 到 memory bank 的最近邻距离
                for i in range(B):
                    patches = combined[i].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
                    # 计算与 memory bank 的距离
                    distances = cdist(patches, self.memory_bank, metric="euclidean")
                    # 每个 patch 的最近邻距离
                    min_distances = np.min(distances, axis=1)
                    # 图像级别异常分数 = 所有 patch 最近邻距离的最大值
                    image_score = np.max(min_distances)
                    all_scores.append(image_score)
        
        return np.array(all_scores)
    
    def _coreset_subsampling(self, features):
        """Greedy Coreset Subsampling 压缩 Memory Bank"""
        n_samples = len(features)
        n_select = max(1, int(n_samples * self.coreset_ratio))
        
        print(f"🔧 Coreset 子采样: 从 {n_samples} 个 patch 特征中选取 {n_select} 个代表...", flush=True)
        
        if n_select >= n_samples:
            return features
            
        # 随机选取第一个种子点
        selected_indices = [np.random.randint(n_samples)]
        
        # 计算所有点到已选点集的最小距离
        min_distances = cdist(features, features[selected_indices], metric="euclidean").min(axis=1)
        
        for i in range(1, n_select):
            # 选择距离已选集合最远的点
            new_idx = np.argmax(min_distances)
            selected_indices.append(new_idx)
            
            # 更新最小距离
            new_distances = cdist(features, features[new_idx:new_idx+1], metric="euclidean").squeeze()
            min_distances = np.minimum(min_distances, new_distances)
            
            if (i + 1) % 100 == 0:
                print(f"   Coreset 进度: {i+1}/{n_select}", flush=True)
        
        return features[selected_indices]
    
    def fit(self, train_loader):
        """训练阶段：从正常样本构建 Memory Bank"""
        print("\n🏋️ [PatchCore 训练] 开始从正常样本中提取特征并构建 Memory Bank...", flush=True)
        t0 = time.time()
        
        features = self._extract_features(train_loader)
        print(f"   原始特征矩阵: {features.shape} (patches × channels)", flush=True)
        
        # Coreset 压缩
        self.memory_bank = self._coreset_subsampling(features)
        print(f"   压缩后 Memory Bank: {self.memory_bank.shape}", flush=True)
        print(f"✅ Memory Bank 构建完成！耗时: {time.time()-t0:.1f}s", flush=True)
    
    def predict(self, test_loader):
        """测试阶段：计算每张图的异常分数"""
        print("\n📊 [PatchCore 测试] 开始在测试集上计算异常分数...", flush=True)
        t0 = time.time()
        scores = self._extract_image_features(test_loader)
        print(f"✅ 异常分数计算完成！耗时: {time.time()-t0:.1f}s", flush=True)
        return scores


class CSVImageDataset(torch.utils.data.Dataset):
    """从 CSV 中读取图片路径和标签"""
    def __init__(self, csv_file, transform=None, label_filter=None):
        self.data = []
        self.transform = transform
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2: continue
                img_path = row[0]
                lbl = row[1]
                # 0=异常(有缺陷), 1=正常
                is_normal = not ("有缺陷" in lbl or "Defective" in lbl)
                label = 1 if is_normal else 0
                
                if label_filter is not None and label != label_filter:
                    continue
                self.data.append((img_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    print("\n" + "="*70, flush=True)
    print("🔬 [Industrial Baseline] PatchCore 无监督抗干扰性极限测试", flush=True)
    print("   纯 PyTorch 实现，零依赖 anomalib", flush=True)
    print("="*70, flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 使用设备: {device}", flush=True)
    
    src_dataset = Path("data/experiments/dataset_all_811")
    if not src_dataset.exists():
        print(f"⚠️ 找不到数据集: {src_dataset}")
        exit(1)
    
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集：只需要正常样本 (label=1)
    print("\n📦 正在加载数据集...", flush=True)
    train_dataset = CSVImageDataset(src_dataset / 'train.csv', transform=transform, label_filter=1)
    # 测试集：正常 + 异常都要
    test_dataset = CSVImageDataset(src_dataset / 'test.csv', transform=transform)
    
    print(f"   训练集 (仅正常样本): {len(train_dataset)} 张", flush=True)
    print(f"   测试集 (全部): {len(test_dataset)} 张", flush=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # 构建并训练 PatchCore
    model = PatchCoreModel(
        backbone_name="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_ratio=0.1,
        num_neighbors=9,
        device=str(device)
    )
    
    model.fit(train_loader)
    
    # 测试
    scores = model.predict(test_loader)
    
    # 收集真实标签
    test_labels = [label for _, label in test_dataset]
    test_labels = np.array(test_labels)
    # 在 PatchCore 中：分数越高 = 越异常
    # 我们的标签：0=异常, 1=正常
    # 所以需要反转：高分数 -> 预测为异常(0)
    
    # 计算 AUROC（分数越高越异常，标签 0 代表异常）
    # 对于 AUROC，需要 y_true 中 1 为正类。这里"异常"是我们关注的正类
    y_true_anomaly = 1 - test_labels  # 1=异常, 0=正常
    auroc = roc_auc_score(y_true_anomaly, scores)
    
    # 使用阈值进行二分类预测
    # 用正常训练集的分数分布来确定阈值
    train_scores = model.predict(train_loader)
    threshold = np.percentile(train_scores, 95)  # 正常样本的 95 百分位作为阈值
    
    print(f"\n📈 自动阈值 (正常样本 95th percentile): {threshold:.4f}", flush=True)
    
    # 预测：分数 > 阈值 -> 异常(0)，否则 -> 正常(1)
    predictions = np.where(scores > threshold, 0, 1)
    
    acc = accuracy_score(test_labels, predictions)
    # 计算对"异常"类别的 precision/recall/f1
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, predictions, pos_label=0, average='binary', zero_division=0)
    
    print("\n" + "="*70, flush=True)
    print("📊 PatchCore 无监督基线测试结果", flush=True)
    print("="*70, flush=True)
    print(f"  AUROC:     {auroc:.4f}", flush=True)
    print(f"  Accuracy:  {acc*100:.2f}%", flush=True)
    print(f"  Precision: {prec*100:.2f}%  (对异常类)", flush=True)
    print(f"  Recall:    {rec*100:.2f}%  (对异常类)", flush=True)
    print(f"  F1-Score:  {f1:.4f}", flush=True)
    print("="*70, flush=True)
    
    # 保存结果到 CSV
    import pandas as pd
    results = pd.DataFrame([{
        "Model": "PatchCore (Unsupervised)",
        "Backbone": "WideResNet50",
        "AUROC": f"{auroc:.4f}",
        "Accuracy": f"{acc*100:.2f}%",
        "Precision": f"{prec*100:.2f}%",
        "Recall": f"{rec*100:.2f}%",
        "F1-Score": f"{f1:.4f}",
        "Threshold": f"{threshold:.4f}"
    }])
    
    out_csv = "data/experiments/patchcore_baseline_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results.to_csv(out_csv, index=False)
    
    print(f"\n✅ 结果已保存至: {out_csv}", flush=True)
    print("重点关注 Precision（低则意味着高 FPR 误报率）和 F1-Score 用于论文对线。", flush=True)
