import os
import shutil
import csv
from pathlib import Path

"""
🚀 CII 顶刊强对标实验: 无监督工业 SOTA 跨界对推
本脚本直接通过 Python API 调用 Anomalib PatchCore 算法，在医疗管线数据集上验证。
【核心论点】：无监督会在复杂透明反光的医疗管线制造中因"环境噪光"产生极高的 FPR (False Positive Rate，误报)。

⚠️ 不再使用 anomalib CLI（anomalib train），因为 CLI 对可选依赖（wandb/openvino）
   缺失时会直接禁用 train 子命令。改为直接使用 Python API 调用 Engine，更稳定可靠。
"""


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
    print("🔄 正在通过解析 CSV 倒流图片，构建 Anomalib 的 Folder 数据集目录结构...", flush=True)
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
                img_path = str(Path(row[0]).absolute())
                lbl = row[1]
                
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
    
    # 统计输出
    n_train = len(list((tgt / "normal" / "train").glob("*")))
    n_test_normal = len(list((tgt / "normal" / "test").glob("*")))
    n_test_abnormal = len(list((tgt / "abnormal" / "test").glob("*")))
    print(f"✅ Anomalib 格式数据集准备完成！训练正常样本: {n_train}, 测试正常: {n_test_normal}, 测试异常: {n_test_abnormal}", flush=True)


if __name__ == '__main__':
    print("\n" + "="*70, flush=True)
    print("🔬 [Industrial Baseline] Anomalib PatchCore 无监督抗干扰性极限测试", flush=True)
    print("="*70, flush=True)
    
    # 检查 anomalib 是否安装
    try:
        import anomalib
        print(f"✓ 检测到 anomalib 已安装, 版本: {anomalib.__version__}", flush=True)
    except ImportError:
        print("❌ 未检测到 anomalib。请在服务器上运行: pip install anomalib")
        exit(1)
        
    src_dataset = "data/experiments/dataset_all_811"
    ano_dataset = "data/experiments/anomalib_dataset"
    
    if not os.path.exists(src_dataset):
        print(f"⚠️ 找不到预划分数据集 {src_dataset}。")
        exit(1)
        
    setup_anomalib_dataset_format(src_dataset, ano_dataset)
    
    # ========================================================
    # 直接使用 Python API 调用 anomalib，不走 CLI
    # ========================================================
    print("\n🚀 正在通过 Python API 直接拉起 PatchCore 训练与评估...", flush=True)
    print("提示：首次运行需下载 wide_resnet50_2 预训练权重，请确保服务器有网络连接。", flush=True)
    
    try:
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        
        # 构建数据模块
        print("📦 正在构建 Anomalib Folder 数据模块...", flush=True)
        datamodule = Folder(
            name="tubeguard_medical",
            root=ano_dataset,
            normal_dir="normal/train",
            abnormal_dir="abnormal/test",
            normal_test_dir="normal/test",
            task="classification",
            image_size=(224, 224),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=8,
        )
        print("✅ 数据模块构建完成！", flush=True)
        
        # 构建模型
        print("🧠 正在构建 PatchCore 模型...", flush=True)
        model = Patchcore(
            backbone="wide_resnet50_2",
            pre_trained=True,
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        print("✅ 模型构建完成！", flush=True)
        
        # 构建 Engine 并训练+测试
        print("⚙️ 正在初始化 Anomalib Engine...", flush=True)
        engine = Engine(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            default_root_dir="results/patchcore",
        )
        
        print("🏋️ 开始 PatchCore 训练（基于 coreset 特征提取，通常只需 1 轮）...", flush=True)
        engine.fit(model=model, datamodule=datamodule)
        
        print("📊 训练完成，开始在测试集上评估...", flush=True)
        test_results = engine.test(model=model, datamodule=datamodule)
        
        print("\n" + "="*70, flush=True)
        print("✅ PatchCore 评测结束！测试结果如下:", flush=True)
        print("="*70, flush=True)
        
        if test_results:
            for result in test_results:
                for key, value in result.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}", flush=True)
                    
        print(f"\n📁 详细结果请查看: results/patchcore/ 目录", flush=True)
        print("重点提取并报告 Precision (以计算高假阳性 FPR) 和 F1-Score 指标用于论文对线。", flush=True)
        
    except TypeError as e:
        # 如果某些参数在当前版本不被接受，尝试精简参数重试
        print(f"\n⚠️ 参数兼容性问题: {e}", flush=True)
        print("🔄 尝试使用精简参数重新构建...", flush=True)
        
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        
        datamodule = Folder(
            name="tubeguard_medical",
            root=ano_dataset,
            normal_dir="normal/train",
            abnormal_dir="abnormal/test",
            normal_test_dir="normal/test",
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=8,
        )
        
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        
        engine = Engine(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            default_root_dir="results/patchcore",
        )
        
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        
        print("\n" + "="*70, flush=True)
        print("✅ PatchCore 评测结束！测试结果如下:", flush=True)
        print("="*70, flush=True)
        
        if test_results:
            for result in test_results:
                for key, value in result.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}", flush=True)
                    
    except Exception as e:
        print(f"\n❌ 执行时发生错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
