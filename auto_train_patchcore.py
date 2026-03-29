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

def setup_anomalib_dataset_format(src_dir: str, target_dir: str):
    """
    将我们的分类格式转化为 Anomalib 的 Folder/MVTec 结构。
    Anomalib 需要:
    target_dir/
        normal/
            train/
            test/
        abnormal/
            test/
    """
    print("🔄 正在构建 Anomalib 对应的 Folder 数据集目录结构...")
    src = Path(src_dir)
    tgt = Path(target_dir)
    
    if tgt.exists():
        shutil.rmtree(tgt)
        
    os.makedirs(tgt / "normal" / "train", exist_ok=True)
    os.makedirs(tgt / "normal" / "test", exist_ok=True)
    os.makedirs(tgt / "abnormal" / "test", exist_ok=True)
    
    # 拷贝良品 (Normal) Train/Test
    # 注意我们在二分类里：0通常是Defect，1通常是Normal。以 ImageFolder 默认字母排序，d 为 abnormal, n 为 normal。
    # 假设源目录结构为 src/train/0_defect, src/train/1_normal
    def get_class_folders(split_dir):
        folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        # 简单判定找 'normal'
        normal_f = next((f for f in folders if 'normal' in f.lower() or f == '1_normal' or f == '1'), None)
        defect_f = next((f for f in folders if 'defect' in f.lower() or 'abnormal' in f.lower() or f == '0_defect' or f == '0'), None)
        return normal_f, defect_f
        
    train_norm, _ = get_class_folders(src / "train")
    test_norm, test_defect = get_class_folders(src / "test")
    
    if not train_norm or not test_norm or not test_defect:
        print("⚠️ 无法自动推断 Normal/Defect 的文件夹名称，请检查源数据目录。")
        exit(1)
        
    print(f"  -> Normal Train: {train_norm}")
    print(f"  -> Normal Test: {test_norm}")
    print(f"  -> Abnormal Test: {test_defect}")
    
    # 拷贝函数
    def copy_files(s, d):
        for f in os.listdir(s):
            if f.endswith(('png', 'jpg', 'jpeg')):
                shutil.copy2(os.path.join(s, f), os.path.join(d, f))
                
    copy_files(src / "train" / train_norm, tgt / "normal" / "train")
    copy_files(src / "test" / test_norm, tgt / "normal" / "test")
    copy_files(src / "test" / test_defect, tgt / "abnormal" / "test")
    
    print("✅ Anomalib 格式数据集准备完成！")

def generate_patchcore_config(dataset_path: str, output_path: str):
    """生成 Anomalib 需要的 yaml 配置文件"""
    config = {
        "dataset": {
            "name": "tubeguard_medical",
            "format": "folder",
            "path": dataset_path,
            "normal_dir": "normal/train",
            "abnormal_dir": "abnormal/test",
            "normal_test_dir": "normal/test",
            "task": "classification",
            "image_size": [224, 224],
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "num_workers": 8,
            "transform_config": {"train": None, "eval": None}
        },
        "model": {
            "name": "patchcore",
            "backbone": "wide_resnet50_2",
            "pre_trained": True,
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9
        },
        "metrics": {
            "image": ["F1Score", "AUROC", "Accuracy", "Precision", "Recall"]
        },
        "project": {
            "seed": 42,
            "path": "./results/patchcore"
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": 1
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ 生成 PatchCore 配置文件: {output_path}")

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
