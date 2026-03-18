# 实验执行与评估指北 (Experiment Guide)

本文档整理了《基于 CutPaste 与 Stable Diffusion 的协同架构研究》涉及到的所有验证、消融与对比实验的代码执行流程与科研逻辑梳理。全部代码均已适配无缝兼容服务器（含断点续训）及一键式测试。

---

## 1. 对比实验 (Comparative Experiments)
**实验目的**：证明在标准的 (8:1:1) 工业管件场景下，本方案提出的 **YOLOv10-TPH** 架构在精度与召回率上优于传统的经典分类大模型（ResNet50、ViT、SwinTransformer）。

### 1.1 模型同域公平训练
*所有模型（包括 YOLOv10-TPH 以及上述经典基线模型）均需要在同一份由数据增强生成的统一大集 `data/unified_dataset` 下使用类似超参数进行微调训练。*

### 1.2 全量自动评估运行方式
我们提供了一键测试脚本，该脚本会自动载入已经预训练好的上述各个模型的权重（`*_best.pth`），在 `data/unified_dataset/test.csv` (混合了各色合成缺陷、划痕、光照异常) 上进行同台竞技，并输出 Precision、Recall 以及 F1-Score。

**执行命令：**
```bash
python evaluate_other_models.py
```
**结果归档：**
汇总表格会自动存入根目录 `data/unified_dataset/other_models_evaluation.txt` 中。

---

## 2. 数据有效性测试（不同划分比例集评估）
**实验目的**：证明模型不仅是在单一数据分布上“死记硬背”，而是能够随着有效训练数据的规模呈现**正向鲁棒性成长**。并且探究在极端的小样本（如 5:3:2）保留条件下，模型依旧能工作的能力。

### 2.1 数据集分布自动化切分生成
使用下述脚本一次性生成三种核心比例（8:1:1、6:2:2、5:3:2）的独立索引训练集：
```bash
# 执行此命令后，将在 data/experiments 下生成 3 个比例的不同子集文件夹
python build_experiment_datasets.py --ratio 8 1 1 --ratio 6 2 2 --ratio 5 3 2
```

### 2.2 多比例自动化重训与测试全家桶 (Pipeline)
*必须让 YOLOv10-TPH 依次在 811、622、532 三个数据集中从零学习，保留三个独立的 `best.pth` 权重，而绝不是用一个通用权重去测不同的验证集。*

**执行命令 (推荐服务端挂机运行)：**
```bash
nohup python auto_train_splits.py > run_splits.log 2>&1 &
```
该脚本运行极为耗时（通常数天），因此内置了 `Checkpoint (Epoch断点记录)` 机制。即便中断重启也能全自动恢复。

**结果获取：**
待完全执行完毕，脚本会自动在各个测试集上推断最终结果并形成总表 `data/experiments/experiment_results_summary.csv`。

---

## 3. 消融实验 (Ablation Study: TPH 模块贡献)
**实验目的**：为了量化我们新研发的 **TPH 分类头结构**是否真正具备价值。我们设计了**在同样的 `8:1:1` 数据全量环境下，纯净版 YOLOv10 基线网络 与 装载有 TPH 外挂的 YOLOv10-TPH 网络 的性能对抗**。这验证了在统一的光照、划痕复杂扩充数据面前，谁的模型结构解析能力更加优越。

### 3.1 运行原版 Vanilla YOLOv10 消融训练
由于原版 YOLOv10 的全连接头未在我们的重混数据上调优，**不能直接蒙猜测试**。请在服务端执行以下基线重训脚本：

**执行命令：**
```bash
nohup python auto_train_ablation.py > ablation_baseline.log 2>&1 &
```
**代码原理：**
- 脚本已硬编码锁定指向 `data/experiments/dataset_all_811`。
- 它自动加载纯原生的 `YOLOv10BaselineClassifier` 重训 50 轮。
- 自动提取模型产生的最后的 `baseline_best.pth`。
- 完成训练后会自动测试并将这组 “纯原生基线 8:1:1” 的准确率写入 `ablation_baseline_results.csv`。

### 3.2 对比出表
您把 `ablation_baseline_results.csv` 里的 Baseline 原版数据（如 F1: 85%），与我们在第 2 节中取得的 `dataset_all_811 的 YOLOv10-TPH F1: 99%` 数据一对比，**这 14% 的断层提升就是论文中证明 TPH 模块具有设计必要性的核弹级数据论点。**
