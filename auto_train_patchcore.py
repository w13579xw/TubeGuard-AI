import os
import shutil
import subprocess
import yaml
from pathlib import Path

"""
🚀 CII 顶刊强对标实验: 无监督工业 SOTA 跨界对推
本脚本配置并调用著名的 Anomalib 框架运行 PatchCore 算法，并在医疗管线数据集上验证
【核心论点】：无监督会在复杂透明反光的医疗管线制造中因“环境噪光”产生极高的 FPR (False Positive Rate，误报)。
"""

import csv

def setup_anomalib_dataset_format(src_dir: str, target_dir: str):
    """
    基于我们特定的 CSV 指引数据集结构，抽取图片组织为 Anomalib 的 Folder 结构。
    Anomalib 需要:
    target_dir/
        normal/
            train/
            test/
        abnormal/
            test/
    """
    print("🔄 正在通过解析 CSV 倒流图片，构建 Anomalib 的 Folder 数据集目录结构...")
    src = Path(src_dir)
    tgt = Path(target_dir)
    
    if tgt.exists():
        shutil.rmtree(tgt)
        
    os.makedirs(tgt / "normal" / "train", exist_ok=True)
    os.makedirs(tgt / "normal" / "test", exist_ok=True)
    os.makedirs(tgt / "abnormal" / "test", exist_ok=True)
    
    def process_csv(csv_path, dst_normal, dst_abnormal=None):
        if not csv_path.exists():
            print(f"⚠️ 找不到 CSV: {csv_path}")
            return
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for i, row in enumerate(reader):
                if len(row) < 2: continue
                img_path = str(Path(row[0]).absolute()) # CSV里可能存的是绝对或相对路径
                lbl = row[1]
                
                # '1' or '正常'/'Normal' indicates good/normal sample
                is_normal = False
                if "有缺陷" not in lbl and "Defective" not in lbl and "0" not in lbl:
                    is_normal = True
                
                try:
                    if is_normal and dst_normal:
                        shutil.copy2(img_path, dst_normal / f"{i}_{Path(img_path).name}")
                    elif not is_normal and dst_abnormal:
                        shutil.copy2(img_path, dst_abnormal / f"{i}_{Path(img_path).name}")
                except Exception as e:
                    pass
                    
    # 只拿 Normal 练 Train
    process_csv(src / "train.csv", tgt / "normal" / "train", None)
    # Test 则区分 Normal 和 Abnormal
    process_csv(src / "test.csv", tgt / "normal" / "test", tgt / "abnormal" / "test")
    
    print("✅ Anomalib 格式数据集准备完成！")

def generate_patchcore_config(dataset_path: str, output_path: str):
    """
    生成 Anomalib v2+ (Lightning CLI) 支持的 yaml 配置文件
    新版框架不能写扁平结构，必需指定 class_path 和 init_args。
    """
    config = {
        "data": {
            "class_path": "anomalib.data.Folder",
            "init_args": {
                "name": "tubeguard_medical",
                "root": dataset_path,
                "normal_dir": "normal/train",
                "abnormal_dir": "abnormal/test",
                "normal_test_dir": "normal/test",
                "image_size": [224, 224],
                "train_batch_size": 32,
                "eval_batch_size": 32,
                "num_workers": 8
            }
        },
        "model": {
            "class_path": "anomalib.models.Patchcore",
            "init_args": {
                "backbone": "wide_resnet50_2",
                "pre_trained": True,
                "layers": ["layer2", "layer3"],
                "coreset_sampling_ratio": 0.1,
                "num_neighbors": 9
            }
        },
        "metrics": {
            "image": ["F1Score", "AUROC", "Accuracy", "Precision", "Recall"]
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": 1
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✅ 生成新版 (LightningCLI) PatchCore 配置文件: {output_path}")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 [Industrial Baseline] Anomalib PatchCore 无监督抗干扰性极限测试")
    print("="*70)
    
    # 检查 anomalib 是否安装
    try:
        import anomalib
        print(f"✓ 检测到 anomalib 已安装, 版本: {anomalib.__version__}")
    except ImportError:
        print("❌ 未检测到 anomalib。请在服务器上运行: pip install anomalib")
        print("如遇包分发策略变更，请使用: pip install anomalib[full] 或根据官方指南安装。")
        exit(1)
        
    src_dataset = "data/experiments/dataset_all_811"
    ano_dataset = "data/experiments/anomalib_dataset"
    config_file = "patchcore_config.yaml"
    
    if not os.path.exists(src_dataset):
        print(f"⚠️ 找不到预划分数据集 {src_dataset}。")
        exit(1)
        
    setup_anomalib_dataset_format(src_dataset, ano_dataset)
    generate_patchcore_config(ano_dataset, config_file)
    
    print("\n🚀 正在拉起 Anomalib PatchCore 训练与评估线程...")
    print("提示：服务器上执行如果有异常报错，请确保服务器网络可以通过 HF_ENDPOINT 或配置了合适的预训练权重下载路线 (wide_resnet50_2)")
    
    # 构建命令调用 anomalib CLI
    # 新版 anomalib 引擎调用方式为: anomalib train --config patchcore_config.yaml
    cmd = f"anomalib train --config {config_file}"
    try:
        # Popen to stream output
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        print("\n✅ PatchCore 评测结束。")
        print("请前往 ./results/patchcore/ 目录查看导出的 CSV 指标日志。")
        print("重点提取并报告 Precision (以计算高假阳性 FPR) 和 F1-Score 指标用于对线。")
        
    except Exception as e:
        print(f"\n❌ 执行时发生错误: {e}")
