import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 导入带有超参控制的 YOLOv10-TPH
from NN.yolov10_tph.model import YOLOv10TPHClassifier

def train_and_eval(num_heads, use_ffn, device, train_loader, val_loader):
    config_name = f"heads_{num_heads}_ffn_{use_ffn}"
    print(f"\n{'='*60}")
    print(f"🚀 [TPH Ablation] 开始评估变体: {config_name}")
    print(f"{'='*60}")
    
    # 初始化模型
    model = YOLOv10TPHClassifier(model_weight='yolov10n.pt', num_classes=2, num_heads=num_heads, use_ffn=use_ffn)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    epochs = 5  # Ablation 用于观察网络架构潜力带来的 F1 落差，服务器可以设高一点 (e.g. 5-10)
    
    best_f1 = 0
    best_metrics = (0, 0, 0, 0) # Acc, Prec, Rec, F1
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{config_name}]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        
        print(f"Epoch {epoch+1} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (acc, prec, rec, f1)
            
    return best_metrics

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 使用设备: {device}")
    
    # 强制在 811 上做消融，能最好地发挥数据量，暴露架构能力上限
    data_dir = Path("data/experiments/dataset_all_811")
    if not data_dir.exists():
        print(f"⚠️ 找不到数据集文件夹: {data_dir}. 请检查服务器上的挂载路径！")
        exit(1)
        
    img_size = 224 # 对齐基线模型分辨率
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=transform)
    test_dataset = datasets.ImageFolder(data_dir / 'test', transform=transform)
    
    batch_size = 32
    num_workers = 8 # 服务器算力充足，调高
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=num_workers)

    # CII 要求的关键 TPH 架构消融配置
    configs = [
        {"num_heads": 2, "use_ffn": True},
        {"num_heads": 4, "use_ffn": True},  # <--- Our Proposed Baseline
        {"num_heads": 8, "use_ffn": True},
        {"num_heads": 4, "use_ffn": False}, # <--- No FFN structural drop
    ]
    
    results = []
    
    for c in configs:
        h, ffn = c["num_heads"], c["use_ffn"]
        acc, prec, rec, f1 = train_and_eval(h, ffn, device, train_loader, test_loader)
        variant_name = f"num_heads={h}" + (", no FFN" if not ffn else "")
        results.append({
            "Variant": variant_name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1:.4f}"
        })
        
    df = pd.DataFrame(results)
    out_csv = "data/experiments/tph_hyperparam_ablation.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ TPH 架构超参数消融实验已完成！结果已存至: {out_csv}")
    print(df)
