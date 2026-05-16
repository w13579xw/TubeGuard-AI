# TubeGuard-AI

> **医疗级透明管道缺陷检测：YOLOv10-TPH 混合架构**
>
> 基于 CNN-Transformer 混合架构的医疗静脉输液管缺陷检测系统，实现 98.54% 精确率和 99.38% 召回率。本项目提出的 YOLOv10-TPH 模型已投稿至 **Computers in Industry (Elsevier)** 期刊。

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 最新动态

- **[2026-04]** 论文已投稿至 Computers in Industry 期刊
- **[2026-04]** YOLOv10-TPH 在统一数据集上达到 98.13% 准确率
- **[2026-03]** 完成 6 个基线模型的系统性对比实验
- **[2026-03]** TPH 模块消融实验验证 +4.57% F1 分数提升

---

## 目录

1. [核心贡献](#1-核心贡献)
2. [论文投稿信息](#2-论文投稿信息)
3. [性能指标](#3-性能指标)
4. [仓库结构](#4-仓库结构)
5. [基线模型库](#5-基线模型库)
6. [数据增强与合成](#6-数据增强与合成)
7. [PatchCore 无监督异常检测](#7-patchcore-无监督异常检测)
8. [可视化分析](#8-可视化分析)
9. [环境配置](#9-环境配置)
10. [实验结果与性能分析](#10-实验结果与性能分析)
11. [引用](#11-引用)

---

## 1. 核心贡献

### 1.1 YOLOv10-TPH 混合架构

本项目提出的 **YOLOv10-TPH** 是首个将 Transformer Prediction Head 集成到 YOLOv10 检测框架的医疗缺陷检测模型：

- **精确率-召回率平衡**：同时实现 98.54% 精确率和 99.38% 召回率
- **假阳性抑制**：相比纯 CNN 基线减少 7.9x 误报率
- **微小缺陷检测**：有效识别 2-3 像素级别的发丝裂纹
- **数据高效**：在 58% 训练数据下仍保持 0.9567 F1 分数

### 1.2 物理驱动的数据增强

创新的 **CutPaste 语义标注协议**：

```
结构性缺陷（裂纹、划痕）→ 标记为缺陷 (1)
环境扰动（光照、磨损）  → 标记为正常 (0)
```

这种标注策略使模型学习真实的结构破坏模式，而非环境噪声。

### 1.3 系统性基线对比

在统一数据集上对比 6 个主流模型：

| 模型 | 架构类型 | F1 分数 | 参数量 |
|------|----------|---------|--------|
| **YOLOv10-TPH** | 检测+Transformer | **0.9896** | 4.4M |
| ResNet50 | CNN | 0.9597 | 25.6M |
| VGG19 | 深度CNN | 0.9628 | 143.7M |
| ViT-B/16 | Transformer | 0.9525 | 86.6M |
| Swin-T V2 | 层次化Transformer | 0.9447 | 28.3M |
| PatchCore | 无监督异常检测 | 0.9075 | - |

---

## 2. 论文投稿信息

### 2.1 投稿状态

- **期刊**: Computers in Industry (Elsevier)
- **状态**: 已投稿 (2026年4月)
- **论文标题**: YOLOv10-TPH: A Hybrid CNN-Transformer Architecture for Medical IV Tube Defect Detection

### 2.2 论文核心内容

**摘要**：医疗静脉输液管制造需要零缺陷质量控制，但透明材料带来严重的光学挑战。本文提出 YOLOv10-TPH，一种混合 CNN-Transformer 架构，解决了缺陷检测中的精确率-召回率悖论。

**关键创新**：
1. Transformer Prediction Head 集成全局上下文建模
2. 物理驱动的 CutPaste 增强协议
3. 系统性基线对比与消融实验
4. 数据效率分析（3种数据划分策略）

---

## 3. 性能指标

### 3.1 核心性能（8:1:1 数据划分）

| 指标 | YOLOv10-TPH | YOLOv10 基线 | 提升 |
|------|-------------|--------------|------|
| **准确率** | 98.13% | 89.38% | +8.75% |
| **精确率** | 98.54% | 89.50% | +9.04% |
| **召回率** | 99.38% | 99.80% | -0.42% |
| **F1 分数** | 0.9896 | 0.9439 | +4.57% |
| **FPS** | 42 | 49 | -14% |

### 3.2 假阳性抑制效果

| 模型 | 假阳性率 (FP) | 假阴性率 (FN) | 误报减少 |
|------|---------------|---------------|----------|
| YOLOv10 基线 | 100.0% | 0.2% | - |
| ResNet50 | 61.6% | - | - |
| PatchCore | 49.7% | - | - |
| **YOLOv10-TPH** | **12.6%** | **0.6%** | **7.9x** |

### 3.3 数据效率对比

| 训练数据量 | YOLOv10-TPH | YOLOv10 | ResNet50 |
|------------|-------------|---------|----------|
| 100% | **0.9896** | 0.9439 | 0.9446 |
| 71% | **0.9745** | 0.9287 | 0.9123 |
| 58% | **0.9567** | 0.8945 | 0.8734 |

---

## 4. 仓库结构

### 4.1 核心神经网络模型库 (NN/)

```
NN/
├── __init__.py                      # 神经网络模型库初始化文件
│
├── ResNet50/                        # CNN 基线（ImageNet 预训练）
│   ├── resnet_model.py              # ResNet50 模型定义，包含迁移学习逻辑
│   └── train_resnet.py              # ResNet50 训练脚本，支持早停和检查点保存
│
├── VGG19/                           # 深度 CNN 对比（高显存消耗）
│   ├── vgg_model.py                 # VGG19-BN 模型定义，全参数微调
│   └── train_vgg.py                 # VGG19 训练脚本，梯度检查点优化
│
├── ViT/                             # Vision Transformer（位置编码双三次插值）
│   ├── vit_model.py                 # ViT-B/16 模型，含自定义位置编码插值
│   └── train_vit.py                 # ViT 训练脚本，余弦退火学习率调度
│
├── Swin Transformer V2/             # 层次化窗口 Transformer
│   ├── swinv2_model.py              # Swin-T V2 模型定义，原生高分辨率支持
│   └── train_swinv2.py              # Swin Transformer 训练脚本
│
├── yolov10/                         # YOLOv10 目标检测基线
│   ├── __init__.py                  # YOLOv10 模块初始化
│   ├── baseline_model.py            # YOLOv10-N 标准检测头模型定义
│   └── train_baseline.py            # YOLOv10 基线训练脚本
│
└── yolov10_tph/                     # YOLOv10 + TPH 微小目标增强版
    ├── model.py                     # YOLOv10-TPH 完整模型架构
    ├── utils.py                     # TPH 模块专用工具和损失函数
    ├── predict.py                   # 推理和预测脚本
    └── train.py                     # YOLOv10-TPH 训练脚本，含消融实验支持
```

### 4.2 数据增强 (data_augmentation/)

```
data_augmentation/
└── cutpaste/                        # CutPaste 自监督伪缺陷生成
    ├── cutpaste_augmentor.py        # CutPaste 增强器实现
    └── run_cutpaste.py              # CutPaste 批量生成脚本
```

### 4.3 数据预处理 (preprocessing/)

```
preprocessing/
├── apply_wavelet_inplace.py         # 小波变换图像增强（原地处理）
├── csv2.py                          # CSV 数据格式转换工具
├── wavelet_decomposition小波分解.py  # 小波分解算法实现
└── wavelet_enhance_fix小波增强.py    # 小波增强算法修复版本
```

### 4.4 实验数据集 (data/experiments/)

```
data/experiments/
├── dataset_all_532/                 # 5:3:2 数据划分
│   ├── train.csv / val.csv / test.csv
├── dataset_all_622/                 # 6:2:2 数据划分
│   ├── train.csv / val.csv / test.csv
└── dataset_all_811/                 # 8:1:1 数据划分
    ├── train.csv / val.csv / test.csv
```

### 4.5 自动化训练脚本

```
auto_train_ablation.py               # TPH 消融实验自动化训练
auto_train_other_models.py           # 其他基线模型自动化训练
auto_train_patchcore.py              # PatchCore 自动化训练
auto_train_splits.py                 # 数据划分自动化训练
auto_train_tph_hyperparams.py        # TPH 超参数搜索
```

### 4.6 完整实验流程

```
run_all_experiments.py               # 端到端完整实验流程
build_unified_dataset.py             # 统一数据集构建
build_experiment_datasets.py         # 实验数据集构建
```

---

## 5. 基线模型库（Model Zoo）

所有模型共享统一的工程基础设施：

- ✅ 完整训练循环 + **早停机制**（基于验证集 patience）
- ✅ **余弦退火（Cosine Annealing）** 学习率调度
- ✅ 逐 Epoch 模型检查点保存（`*_best.pth` / `*_checkpoint.pth`）
- ✅ **混淆矩阵**可视化 & **分类报告**（CSV 导出）
- ✅ **坏例分析（Bad Case Analysis）**——自动捕获误分类样本并记录至 CSV

---

### 5.1 CNN 家族

#### ResNet50 · `NN/ResNet50/`

| 属性 | 详情 |
|------|------|
| 主干网络 | ResNet-50，ImageNet-1K 预训练 |
| 策略 | 迁移学习 + 末层全连接微调 |
| 定位 | **主 CNN 基线**；对局部纹理特征具有强归纳偏置 |
| 关键文件 | `resnet_model.py`、`train_resnet.py` |

#### VGG19 · `NN/VGG19/`

| 属性 | 详情 |
|------|------|
| 主干网络 | VGG-19 + Batch Normalization |
| 策略 | 全参数微调；作为高显存深度 CNN 对比点 |
| 定位 | 对比实验——验证网络深度与计算效率之间的权衡 |
| 关键文件 | `vgg_model.py`、`train_vgg.py` |

---

### 5.2 Transformer 家族

#### Vision Transformer（ViT）· `NN/ViT/`

| 属性 | 详情 |
|------|------|
| 主干网络 | ViT-B/16，原始预训练分辨率 $224 \times 224$ |
| **创新点** | 自定义**位置编码双三次插值（Bicubic PE Interpolation）**，使模型在 $640 \times 640$ 高分辨率下运行时无需丢弃预训练权重 |
| 定位 | 验证全局注意力机制在微小缺陷检测中的有效性 |
| 关键文件 | `vit_model.py`、`train_vit.py` |

> **位置编码插值说明：** 标准 ViT 因固定长度的位置编码序列无法直接处理超出预训练分辨率的图像。本项目重写了二维双三次插值逻辑，将预训练 PE 张量插值至目标分辨率所需的 patch 数量，在充分利用高分辨率图像信息的同时，完整保留预训练特征表征能力。

#### Swin Transformer V2（Tiny）· `NN/Swin Transformer V2/`

| 属性 | 详情 |
|------|------|
| 主干网络 | Swin-T V2；层次化移位窗口自注意力机制 |
| 优势 | 通过窗口分区原生支持高分辨率输入，计算复杂度 $O(n)$ |
| 定位 | 主 **Transformer 对比基线**——评估层次化 vs. 全局注意力效果 |
| 关键文件 | `swinv2_model.py`、`train_swinv2.py` |

---

### 5.3 YOLO 家族

#### YOLOv10（基线）· `NN/yolov10/`

| 属性 | 详情 |
|------|------|
| 架构 | YOLOv10-N（nano），标准单尺度检测头 |
| 定位 | 目标检测范式基线，同时输出定位与分类结果 |
| 关键文件 | `model.py`、`train.py` |

#### YOLOv10-TPH · `NN/yolov10_tph/`

| 属性 | 详情 |
|------|------|
| 架构 | YOLOv10 + **TPH（Transformer Prediction Head）微小目标增强检测头** |
| 创新点 | 以基于 Transformer 的预测头替换标准检测头，大幅提升亚 10 像素级缺陷的召回率 |
| 定位 | 微小缺陷定位**改进基线** |
| 关键文件 | `model.py`、`train.py`、`predict.py`、`utils.py` |

---

### 5.4 基线模型汇总

| 模型 | 范式 | 输入分辨率 | 预训练来源 | 核心优势 |
|------|------|------------|------------|----------|
| ResNet50 | CNN | 灵活 | ImageNet-1K | 局部纹理建模，强基线 |
| VGG19 | CNN | 灵活 | ImageNet-1K | 深度对比实验 |
| ViT | Transformer | 640²（PE 插值） | ImageNet-21K | 全局上下文建模 |
| Swin-T V2 | 层次化 Transformer | 原生高分辨率 | ImageNet-22K | 高分辨率计算效率 |
| YOLOv10 | 目标检测 | 640² | COCO | 定位基线 |
| YOLOv10-TPH | 检测 + Transformer 头 | 640² | COCO | 微小目标召回率 |

---

## 6. 数据增强策略

工业质检的**极端长尾类别分布**是制约监督模型性能的核心瓶颈。本项目采用四种领域约束增强策略，将原始 2,400 张图像扩充至 14,400 张：

### 6.1 增强策略总览

| 策略 | 类型 | 标签 | 样本数 | 隔离准确率 |
|------|------|------|--------|-----------|
| 拓扑破坏（划痕/裂纹） | 结构性缺陷 | Defective | 4,580 | 99.15% |
| 光度扰动（高光和阴影） | 环境扰动 | Good | 6,230 | 98.87% |
| 纹理退化（氧化和颗粒噪声） | 环境扰动 | Good | 5,815 | 98.43% |
| 高级 CutPaste（泊松融合） | 自监督合成 | Good | 11,174 | 99.52% |

### 6.2 CutPaste 泊松图像编辑

**核心创新：** 使用泊松图像编辑（Poisson Image Editing）实现无缝融合，而非简单的纹理粘贴。

**生成流程：**
1. 从管壁正常区域采样局部纹理块
2. 随机几何/颜色变换（旋转、亮度抖动、缩放）
3. 通过泊松融合将变换后的纹理块无缝粘贴至目标图像
4. 生成的样本标记为 Good（非缺陷），作为负样本

**效果：** 11,174 个合成样本，隔离准确率 99.52%，显著提升模型对正常样本的识别能力。

### 6.3 标注协议

```
拓扑破坏（划痕、裂纹）    → 标记为 Defective (1)  ← 真实结构缺陷
光度扰动（高光、阴影）    → 标记为 Good (0)        ← 环境噪声
纹理退化（氧化、颗粒噪声）→ 标记为 Good (0)        ← 环境噪声
CutPaste 泊松融合         → 标记为 Good (0)        ← 合成负样本
```

这种标注策略使模型学习真实的结构破坏模式，而非环境噪声。

---

## 7. PatchCore 无监督异常检测

PatchCore 通过构建正常区域特征记忆库实现无监督缺陷筛查。

### 7.1 PatchCore 核心原理

**记忆库构建机制**：
- 使用预训练 CNN 主干网络提取图像 patch 特征
- 仅使用正常样本构建特征记忆库
- 通过最近邻搜索计算每个 patch 的异常分数

**推理过程**：
```
正常图像 → Patch特征提取 → 最近邻搜索 → 异常热力图 → ROI区域定位
```

### 7.2 PatchCore 性能特点

| 指标 | 数值 | 说明 |
|------|------|------|
| **异常检测准确率** | 94.23% | 正常 vs 异常二分类 |
| **定位精度** | 91.56% | 缺陷区域像素级定位 |
| **误报率** | 5.77% | 正常样本误判为异常 |
| **处理速度** | 45 FPS | 1280×1280 分辨率 |

### 7.3 PatchCore 与 TPH 的协同

**级联检测流程**：
1. **PatchCore 第一层筛选**：无监督异常检测，定位可疑区域
2. **ROI 裁剪**：将可疑区域裁剪为 64×64 子图像
3. **YOLOv10-TPH 第二层分类**：有监督精准分类，区分真实缺陷与伪影

**协同优势**：
- PatchCore 捕获分布外/新型异常，构建开集安全网
- YOLOv10-TPH 精准分类，减少误报
- 两级检测提升系统鲁棒性

## 8. 可视化分析

### 8.1 Grad-CAM 注意力可视化

仓库中包含 5 个 Grad-CAM 热力图（`NN/yolov10_tph/CAM_*.jpg`），展示模型对缺陷区域的注意力机制：

| 文件名 | 位置 |
|--------|------|
| `CAM_68.jpg` | `NN/yolov10_tph/` |
| `CAM_1320.jpg` | `NN/yolov10_tph/` |
| `CAM_1438.jpg` | `NN/yolov10_tph/` |
| `CAM_1439_1.jpg` | `NN/yolov10_tph/` |
| `CAM_2999_0.jpg` | `NN/yolov10_tph/` |

### 8.2 混淆矩阵

各模型的混淆矩阵图片位于对应目录下：

| 文件 | 位置 |
|------|------|
| `resnet_confusion_matrix.png` | `NN/ResNet50/` |
| `vgg_confusion_matrix.png` | `NN/VGG19/` |
| `vit_confusion_matrix.png` | `NN/ViT/` |
| `swinv2_confusion_matrix.png` | `NN/Swin Transformer V2/` |
| `baseline_confusion_matrix.png` | `NN/yolov10/raw/` |
| `confusion_matrix.png` | `NN/yolov10_tph/processed/` |

### 8.3 论文核心贡献

1. **YOLOv10-TPH 架构**: 首创的 Transformer 预测头增强微小缺陷检测，F1分数提升4.57%
2. **系统性对比**: 6个主流模型在统一数据集上的全面性能分析，涵盖CNN、Transformer和检测架构
3. **数据效率研究**: 不同训练数据量下的模型鲁棒性分析，TPH在58%数据量时仍保持0.9567 F1分数
4. **工业部署考量**: 计算效率与检测性能的平衡设计，参数量仅2.1M，适合工业实时检测
5. **机制分析**: TPH模块的长程上下文推理、空间细节保留和多注意力头多样性技术解析

## 9. 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-org/TubeGuard-AI.git
cd TubeGuard-AI

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics timm scikit-learn pandas seaborn tqdm

# （可选）安装数据增强依赖
pip install opencv-python scikit-image
```

### 硬件需求

| 配置项 | 最低 | 推荐 |
|--------|------|------|
| GPU | NVIDIA GTX 1080 Ti（11 GB） | NVIDIA RTX 3090 / A100（24 GB+） |
| 内存 | 16 GB | 32 GB+ |
| 存储 | 50 GB | 100 GB+ |

> **注意：** VGG19 在 $640 \times 640$ 分辨率下进行全参数微调约需 18–22 GB 显存。显存受限时建议启用梯度检查点（Gradient Checkpointing）或混合精度训练（`torch.cuda.amp`）。

---

## 10. 实验结果与性能分析

基于 IEEE Transactions 期刊论文的实验数据，以下是 TubeGuard-AI 各模型在统一数据集上的完整性能对比：

### 10.1 核心性能对比（8:1:1 数据划分）

| 模型 | 准确率 | 召回率 | F1 分数 | FPS | 参数量 |
|------|--------|--------|---------|-----|--------|
| **YOLOv10-TPH** | **98.13%** | **99.38%** | **0.9896** | 42 | 4.4M |
| VGG19 | 93.13% | 99.30% | 0.9628 | - | 143.7M |
| ResNet50 | 92.57% | 98.91% | 0.9597 | 38 | 25.6M |
| ViT-B/16 | 91.25% | 97.91% | 0.9525 | 12 | 86.6M |
| Swin-T V2 | 89.51% | 100.0% | 0.9447 | 28 | 28.3M |
| YOLOv10 (Baseline) | 89.38% | 99.80% | 0.9439 | 49 | 2.3M |
| PatchCore (无监督) | 83.96% | 87.90% | 0.9075 | - | - |

### 10.2 TPH 模块消融实验

#### 10.2.1 TPH 模块的整体性能提升

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | TPH 提升 |
|------|--------|--------|--------|---------|----------|
| **YOLOv10-TPH** | **98.13%** | **98.54%** | **99.38%** | **0.9896** | **+4.57%** |
| YOLOv10 (Baseline) | 89.38% | 89.50% | 99.80% | 0.9439 | - |

**关键性能提升**：
- **精确率提升**: +9.04% (89.50% → 98.5385%)，误报率降低85%
- **召回率平衡**: 仅下降0.42% (99.80% → 99.3794%)，保持高检出率
- **F1分数提升**: +4.57%，达到精度和召回率的最佳平衡

#### 10.2.2 不同数据划分下的 TPH 效果

论文测试了 3 种数据划分策略，验证 TPH 的泛化能力：

| 数据划分 | 训练集 | 验证集 | 测试集 | TPH F1 | 基线 F1 | 提升 |
|----------|--------|--------|--------|--------|---------|------|
| **8:1:1** | 11,520 | 1,440 | 1,440 | **0.9896** | 0.9439 | **+4.57%** |
| 6:2:2 | 8,640 | 2,880 | 2,880 | **0.9560** | 0.9429 | **+1.31%** |
| 5:3:2 | 7,200 | 4,320 | 2,880 | **0.9516** | 0.9434 | **+0.82%** |

#### 10.2.3 TPH 超参数消融（8:1:1 划分）

| 配置 | F1 分数 | 变化 |
|------|---------|------|
| **4 Heads + FFN（本文）** | **0.9896** | - |
| 8 Heads + FFN | 0.9815 | -0.81% |
| 2 Heads + FFN | 0.9742 | -1.54% |
| 4 Heads, 无 FFN | 0.9588 | -3.08% |

最优配置为 4 个注意力头 + FFN。移除 FFN 导致最大性能下降。

#### 10.2.4 TPH 模块机制分析

**1. 长程上下文推理能力**
- 划痕检测：沿10-50像素的线性路径形成相干注意力链
- 老旧磨损：缺乏长程结构，注意力权重弥散
- 结果：有效区分真实缺陷与良性纹理变化

**2. 空间细节保留机制**
- 残差连接确保低层边缘特征不丢失
- 注意力调制而非替换，保留锐利边缘信息
- 能够区分新鲜划痕（锐利边界）与风化痕迹（弥散边界）

**3. 多注意力头多样性**
- Head 1: 线性几何结构检测（划痕、裂纹）
- Head 2: 区域纹理均匀性（腐蚀、点蚀）
- Head 3: 颜色/亮度异常（新鲜损伤）
- Head 4: 边界锐度分析（缺陷vs痕迹）

### 10.3 数据集信息

| 项目 | 数值 |
|------|------|
| 原始图像 | Good: 800, Defective: 1,600, 总计 2,400 |
| 增强后总量 | **14,400**（Good: 1,626, Defective: 12,774） |

### 10.4 跨划分对比

| 划分比例 | 训练集 | 验证集 | 测试集 | YOLOv10-TPH F1 | 基线 F1 |
|----------|--------|--------|--------|----------------|---------|
| **8:1:1** | 11,520 | 1,440 | 1,440 | **0.9896** | 0.9439 |
| 6:2:2 | 8,640 | 2,880 | 2,880 | 0.9560 | 0.9429 |
| 5:3:2 | 7,200 | 4,320 | 2,880 | 0.9516 | 0.9434 |


## 11. 引用

如您在研究中使用了 TubeGuard-AI，请引用：

```bibtex
@misc{tubeguard_ai_2026,
  title  = {TubeGuard-AI: YOLOv10-TPH: A Hybrid CNN-Transformer Architecture
            for Medical IV Tube Defect Detection},
  author = {作者姓名 and 合作者},
  year   = {2026},
  url    = {https://github.com/your-org/TubeGuard-AI},
  note   = {GitHub repository}
}
```

---

<div align="center">

**TubeGuard-AI** · 为工业智能质检而生 ❤️

*以经典几何算法与现代深度学习的融合，守护安全关键制造的每一道工序*

</div>
