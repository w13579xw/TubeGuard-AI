import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pandas as pd
from pathlib import Path
import csv
from PIL import Image
import argparse
import subprocess
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入带有超参控制的 YOLOv10-TPH
from NN.yolov10_tph.model import YOLOv10TPHClassifier

class CSVImageDataset(torch.utils.data.Dataset):
    """跨平台路径强兼容免拷贝数据集读取逻辑"""
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if len(row) >= 2:
                    img_path = row[0]
                    lbl = row[1]
                    gt = 0 if ("有缺陷" in lbl or "Defective" in lbl) else 1
                    self.data.append((img_path, gt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path_str, label = self.data[idx]
        img_path = Path(img_path_str)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (640, 640), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
        return image, label


def train_and_eval(num_heads, use_ffn, device, train_loader, val_loader):
    config_name = f"heads_{num_heads}_ffn_{use_ffn}"
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 [TPH Ablation] 开始评估变体: {config_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 初始化模型
    model = YOLOv10TPHClassifier(model_weight='yolov10n.pt', num_classes=2, num_heads=num_heads, use_ffn=use_ffn)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    max_epochs = 300   # 较高轮次的上限拦截
    patience = 30      # 早停耐心值: 如果连续30轮F1无提升，则停止
    
    best_f1 = 0
    best_metrics = (0, 0, 0, 0) # Acc, Prec, Rec, F1
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
        
        avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"[{config_name}] Epoch {epoch+1:03d}/{max_epochs} -> Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | (No improve: {epochs_no_improve})", flush=True)
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (acc, prec, rec, f1)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"[{config_name}] 📉 验证集 F1 分数在连续 {patience} 轮中未提升，触发早停机制！(停止于第 {epoch+1} 轮)", flush=True)
            break
            
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=-1, help='TPH 注意力头数 (-1 为启动所有)')
    parser.add_argument('--use_ffn', type=int, default=-1, help='是否使用 FFN (1/0)')
    args = parser.parse_args()

    configs = [
        {"num_heads": 2, "use_ffn": True},
        {"num_heads": 4, "use_ffn": True},  # <--- Our Proposed Baseline
        {"num_heads": 8, "use_ffn": True},
        {"num_heads": 4, "use_ffn": False}, # <--- No FFN structural drop
    ]
    
    out_csv = "data/experiments/tph_hyperparam_ablation.csv"
    log_dir = "data/experiments/logs"

    if args.num_heads == -1:
        # 并行调度模式
        os.makedirs(log_dir, exist_ok=True)
        print("🚀 正在启动并行消融实验（带有基于验证集的早停机制）...", flush=True)
        
        processes = []
        log_files = []
        
        for c in configs:
            h, ffn = c["num_heads"], c["use_ffn"]
            ffn_int = 1 if ffn else 0
            
            # 为每个单独的子进程配置独立的日志输出文件（既避免交错干扰也方便查阅）
            log_path = os.path.join(log_dir, f"train_log_heads{h}_ffn{ffn_int}.txt")
            log_f = open(log_path, "w", encoding="utf-8")
            log_files.append(log_f)
            
            # 启动子进程，将标准输出和错误流重定向到该文件
            cmd = [sys.executable, __file__, '--num_heads', str(h), '--use_ffn', str(ffn_int)]
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            processes.append(p)
            print(f"   ▶️ 成功派生并行子进程 -> heads={h}, ffn={bool(ffn_int)} (日志将被持续写入至: {log_path})")
            
        print(f"⏱️ 所有 {len(processes)} 个子进程已发车，请在对应日志文件中查阅各自的训练进度...", flush=True)
        
        for p in processes:
            p.wait()
            
        for log_f in log_files:
            log_f.close()
            
        print("\n✅ 所有并行训练子进程已退出，正在汇总最终早停收敛结果...", flush=True)
        
        # 结果汇总
        results_df_list = []
        for c in configs:
            h, ffn = c["num_heads"], c["use_ffn"]
            res_file = f"data/experiments/tmp_res_h{h}_f{1 if ffn else 0}.csv"
            if os.path.exists(res_file):
                results_df_list.append(pd.read_csv(res_file))
                os.remove(res_file) # 删除临时文件
                
        if results_df_list:
            final_df = pd.concat(results_df_list, ignore_index=True)
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            final_df.to_csv(out_csv, index=False)
            print(f"✅ TPH 架构并行消融实验已跑齐！融合结果已存至: {out_csv}", flush=True)
            print(final_df, flush=True)
        else:
            print("❌ 未能聚合汇总临时结果表，请检查对应子进程日志探查是否存在故障报错。", flush=True)
            
    else:
        # 实际训练执行逻辑
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_name = f"heads_{args.num_heads}_ffn_{args.use_ffn}"
        print(f"[{config_name}] 📌 检查使用算力设备: {device}", flush=True)
        
        data_dir = Path("data/experiments/dataset_all_811")
        if not data_dir.exists():
            print(f"[{config_name}] ⚠️ 致命异常：找不到数据集文件夹: {data_dir}. 是不是服务器的挂载绝对路径有变动？", flush=True)
            sys.exit(1)
            
        img_size = 224 # 对齐基底模型分辨率
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = CSVImageDataset(data_dir / 'train.csv', transform=transform)
        test_dataset = CSVImageDataset(data_dir / 'test.csv', transform=transform)
        
        batch_size = 32
        num_workers = 4 # 并行状态中拉低预抓取并发上限即可防止系统运存不足 OOM
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=num_workers)

        h = args.num_heads
        ffn = bool(args.use_ffn)
        
        acc, prec, rec, f1 = train_and_eval(h, ffn, device, train_loader, test_loader)
        
        variant_name = f"num_heads={h}" + (", no FFN" if not ffn else "")
        df = pd.DataFrame([{
            "Variant": variant_name,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1:.4f}"
        }])
        
        # 保存独占该进程临时快照，防止并行竞态冲突
        res_file = f"data/experiments/tmp_res_h{h}_f{1 if ffn else 0}.csv"
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        df.to_csv(res_file, index=False)
        print(f"[{config_name}] 🎉 当前子进程消融实验完毕，已将最佳快照结果剥离写入至: {res_file}", flush=True)
