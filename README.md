# TubeGuard-AI

> **医疗级透明管道缺陷检测：YOLOv10-TPH 混合架构**
>
> 基于 CNN-Transformer 混合架构的医疗静脉输液管缺陷检测系统，实现 98.54% 精确率和 99.38% 召回率。本项目提出的 YOLOv10-TPH 模型已投稿至 **Computers in Industry (Elsevier)** 期刊。

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Submitted-orange)](paper/submission_package/)

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
8. [论文图表与可视化](#8-论文图表与可视化)
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
| **YOLOv10-TPH** | 检测+Transformer | **0.9896** | 2.1M |
| ResNet50 | CNN | 0.9446 | 25.6M |
| VGG19 | 深度CNN | 0.9628 | 143.7M |
| ViT-B/16 | Transformer | 0.8083 | 86.6M |
| Swin-T V2 | 层次化Transformer | 0.8688 | 28.3M |
| PatchCore | 无监督异常检测 | 0.9075 | - |

---

## 2. 论文投稿信息

### 2.1 投稿状态

- **期刊**: Computers in Industry (Elsevier)
- **状态**: 已投稿 (2026年4月)
- **论文标题**: YOLOv10-TPH: A Hybrid CNN-Transformer Architecture for Medical IV Tube Defect Detection

### 2.2 论文文件

所有论文相关文件位于 `paper/submission_package/` 目录：

```
paper/submission_package/
├── ComputersInIndustry_manuscript.tex    # 主论文 LaTeX 源文件
├── manuscript.pdf                        # 编译后的 PDF
├── ComputersInIndustry_references.bib    # 参考文献
├── IEEEtran.cls                          # LaTeX 文档类
├── title_page.txt                        # 标题页（作者信息）
├── Highlights.docx                       # 论文亮点
├── Acknowledgments.docx                  # 致谢
├── Declaration of Interest Statement.docx # 利益声明
├── Measurement_Cover Letter.docx         # 投稿信
├── Measurement_title-page.docx           # 期刊标题页
├── assets/                               # 所有图表文件
│   ├── model_architecture.png            # 模型架构图
│   ├── training_curves.png               # 训练曲线
│   ├── precision_recall_radar.png        # 性能雷达图
│   ├── ablation_study_f1_comparison.png  # 消融实验
│   ├── cross_architecture_comparison.png # 跨架构对比
│   ├── full_performance_comparison.png   # 完整性能对比
│   ├── data_efficiency_analysis.png      # 数据效率分析
│   ├── f1_stability_boxplot.png          # F1 稳定性箱线图
│   ├── tph_ablation_f1.png              # TPH 超参数消融
│   ├── bottleneck.png                    # Bottleneck 模块图
│   ├── c2f_module.png                    # C2f 模块图
│   ├── mhsa_detail.png                   # MHSA 细节图
│   ├── scdown_module.png                 # SCDown 模块图
│   ├── tph_module.png                    # TPH 模块图
│   └── CAM_*.jpg                        # Grad-CAM 可视化
├── SUBMISSION_CHECKLIST.txt              # 投稿检查清单
└── FORMATTING_MODIFICATIONS_SUMMARY.txt  # 格式修改总结
```

### 2.3 论文核心内容

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

### 3.2 假阳性抑制效果

| 模型 | 假阳性率 | 误报减少 |
|------|----------|----------|
| YOLOv10 基线 | 100.0% | - |
| ResNet50 | 61.6% | - |
| **YOLOv10-TPH** | **12.6%** | **7.9x** |

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

### 4.2 论文与投稿文件 (paper/)

```
paper/
├── submission_package/                  # 期刊投稿文件包
│   ├── ComputersInIndustry_manuscript.tex  # 主论文 LaTeX 源文件
│   ├── manuscript.pdf                   # 编译后的 PDF
│   ├── ComputersInIndustry_references.bib # 参考文献
│   ├── IEEEtran.cls                     # LaTeX 文档类
│   ├── title_page.txt                   # 标题页（作者信息）
│   ├── assets/                          # 所有图表文件
│   │   ├── model_architecture.png       # 模型架构图
│   │   ├── training_curves.png          # 训练曲线
│   │   ├── precision_recall_radar.png   # 性能雷达图
│   │   ├── ablation_study_f1_comparison.png  # 消融实验
│   │   ├── cross_architecture_comparison.png # 跨架构对比
│   │   ├── full_performance_comparison.png   # 完整性能对比
│   │   ├── data_efficiency_analysis.png      # 数据效率分析
│   │   ├── f1_stability_boxplot.png     # F1 稳定性箱线图
│   │   ├── tph_ablation_f1.png          # TPH 超参数消融
│   │   ├── bottleneck.png               # Bottleneck 模块图
│   │   ├── c2f_module.png               # C2f 模块图
│   │   ├── mhsa_detail.png              # MHSA 细节图
│   │   ├── scdown_module.png            # SCDown 模块图
│   │   ├── tph_module.png               # TPH 模块图
│   │   └── CAM_*.jpg                   # Grad-CAM 可视化
│   ├── SUBMISSION_CHECKLIST.txt         # 投稿检查清单
│   └── FORMATTING_MODIFICATIONS_SUMMARY.txt  # 格式修改总结
│
├── 4.3论文修改意见.pdf                   # 审稿意见
├── writing_and_formatting.txt           # 期刊格式要求
├── COMPILATION_INSTRUCTIONS.md          # 论文编译说明
├── FIGURE_GENERATION_GUIDE.md           # 图表生成指南
├── TPH_MODULE_ANALYSIS.md               # TPH 模块详细分析
├── LABELING_PROTOCOL.md                 # 标注协议说明
├── analyze_tex.py                       # LaTeX 文件分析工具
├── compress.py                          # 图片压缩工具
├── generate_ablation_fig.py             # 消融实验图表生成
└── plot_experiment_results.py           # 实验结果绘图
```

### 4.3 数据增强与合成 (data_augmentation/)

```
data_augmentation/
├── cutpaste/                        # CutPaste 自监督伪缺陷生成
│   ├── cutpaste_augmentor.py        # CutPaste 增强器实现
│   └── run_cutpaste.py              # CutPaste 批量生成脚本
│
└── stable_diffusion/                # Stable Diffusion 图像合成
    ├── run_sd_synthesis.py          # SD 合成主脚本
    └── sd_synthesizer.py            # SD 合成器封装类
```

### 4.4 数据预处理 (preprocessing/)

```
preprocessing/
├── apply_wavelet_inplace.py         # 小波变换图像增强（原地处理）
├── csv2.py                          # CSV 数据格式转换工具
├── wavelet_decomposition小波分解.py  # 小波分解算法实现
└── wavelet_enhance_fix小波增强.py    # 小波增强算法修复版本
```

### 4.5 数据集管理 (data/)

```
data/
├── unified_dataset/                 # 统一数据集
│   └── dataset.yaml                 # 数据集配置文件
├── defect_test/                     # 缺陷测试集
├── defect_test_heatmaps/            # 缺陷热力图可视化
├── experiments/                     # 实验数据记录
├── results/                         # 实验结果输出
├── images/                          # 图像资源
├── train.csv                        # 训练集标注
└── test.csv                         # 测试集标注
```

### 4.6 数据存储 (raw_data/ & processed_data/)

```
raw_data/                            # 原始数据
├── train/                           # 训练集
├── val/                             # 验证集
└── test/                            # 测试集

processed_data/                      # 预处理后数据
├── train/                           # 训练集
├── val/                             # 验证集
└── test/                            # 测试集
```

### 4.7 自动化训练与评估脚本

```
auto_train_ablation.py               # TPH 消融实验自动化训练
auto_train_other_models.py           # 其他基线模型自动化训练
auto_train_patchcore.py              # PatchCore 自动化训练
auto_train_splits.py                 # 数据划分自动化训练
auto_train_tph_hyperparams.py        # TPH 超参数搜索
```

### 4.8 评估与分析脚本

```
evaluate_baseline_unified.py         # 基线模型统一评估
evaluate_defects_yolov10tph.py       # YOLOv10-TPH 缺陷评估
evaluate_other_models.py             # 其他模型评估
evaluate_unified_yolov10tph.py       # 统一数据集评估
plot_experiment_results.py           # 实验结果绘图
generate_ablation_fig.py             # 消融实验图表生成
```

### 4.9 资源与工具 (res/)

```
res/
├── yolov10n-cls-custom.yaml         # YOLOv10 自定义配置
├── DeepLabV3.py                     # DeepLabV3 模型实现
├── PatchCore.py                     # PatchCore 异常检测
├── VGG19(Baseline).py               # VGG19 基线实现
├── YOLOv10-CBAM.py                  # YOLOv10-CBAM 模型
├── YOLOv10.py                       # YOLOv10 标准实现
├── augment.py                       # 数据增强实现
├── augment2.py                      # 增强算法版本2
├── main.py                          # 主程序入口
├── model.py                         # 通用模型定义
├── preprocessor.py                  # 数据预处理器
└── vmamba/                          # VMamba 模型
    ├── train_vmamba.py              # VMamba 训练脚本
    └── vmamba_model.py              # VMamba 模型定义
```

### 4.10 完整实验流程

```
run_all_experiments.py               # 端到端完整实验流程
build_unified_dataset.py             # 统一数据集构建
build_experiment_datasets.py         # 实验数据集构建
```

### 4.11 训练日志 (log/)

```
log/                                 # 训练日志与输出记录
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
| 关键文件 | `swin_model.py`、`train_swin.py` |

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

## 6. 数据增强与合成

工业质检的**极端长尾类别分布**是制约监督模型性能的核心瓶颈。本项目提出两种互补的数据合成策略：

### 6.1 CutPaste 自监督伪缺陷生成

**目标：** 从正常管道图像中生成伪缺陷图像，为 PatchCore 等无监督异常检测模型提供自监督预训练信号，无需人工标注。

**生成流程：**

```
正常管道图像
      │
      ▼
┌──────────────────────────────┐
│  1. 从管壁正常区域采样局部纹理块 │
│                              │
│  2. 随机几何/颜色变换：        │
│     · 旋转（0°–360°）        │
│     · 亮度/色彩抖动           │
│     · 缩放                   │
│                              │
│  3. 将变换后的纹理块           │
│     随机粘贴至原图或其他图像   │
└──────────────────────────────┘
      │
      ▼
伪缺陷图像（自监督训练信号）
```

**应用场景：** 生成的伪缺陷图像作为 PatchCore 记忆库构建阶段的负样本，提升模型对真实异常区域的敏感度，全程无需缺陷区域的人工标注。

---

### 6.2 Stable Diffusion 图像合成

**目标：** 利用图生图（img2img）/ Inpainting 技术，合成高逼真度的极细微裂纹和复杂气泡图像，丰富缺陷特征空间，缓解训练集类别不平衡问题。

**合成流程：**

```
真实管道图像（条件输入）
      │
   img2img / Inpainting
      │
SD 模型 + 缺陷提示词工程
例："hairline crack on transparent
    polymer tube wall, industrial
    inspection, macro photography"
      │
      ▼
合成缺陷图像
      │
      ▼
质量筛选（LPIPS 感知相似度 / 人工复审）
      │
      ▼
扩充后的训练数据集
```

**预期效果：**
- 解决稀有缺陷类别（如宽度 < 5px 的发丝裂纹）样本严重不足的问题
- 扩大缺陷形态多样性，提升分类器泛化能力
- 在不增加额外人工标注成本的前提下支持监督分类器训练

---

## 7. PatchCore 无监督异常检测

PatchCore 通过构建正常区域特征记忆库来实现工业缺陷的自动筛查。

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

## 8. 论文图表与可视化

本项目的实验结果已通过专业的 IEEE Transactions 格式图表展示，所有图表均已生成并保存在 `paper/assets/` 目录中：

### 8.1 核心性能图表

| 图表编号 | 文件名 | 描述 | 位置 |
|----------|--------|------|------|
| Fig. 1 | `training_curves.png` | 训练进度曲线 - 展示模型在100个epoch内的收敛过程 | Section III.A |
| Fig. 2 | `data_efficiency_analysis.png` | 数据效率分析 - 不同训练数据量下的性能对比 | Section III.B |
| Fig. 3 | `full_performance_comparison.png` | 完整性能对比 - 6个模型的4个指标综合对比 | Section III.E |
| Fig. 4 | `ablation_study_f1_comparison.png` | 消融实验 - TPH模块的性能提升效果 | Section III.B |
| Fig. 5 | `cross_architecture_comparison.png` | 跨架构对比 - 6个模型在3种数据划分下的F1对比 | Section III.C |
| Fig. 6 | `precision_recall_radar.png` | 雷达图 - 精确率、召回率、准确率三维度对比 | Section III.C |
| Fig. 7 | `model_architecture.png` | 模型架构图 - YOLOv10-TPH的详细网络结构 | Section II.A |

### 8.2 Grad-CAM 注意力可视化

4个 Grad-CAM 热力图展示模型对缺陷区域的注意力机制：

| 文件名 | 描述 | 缺陷类型 |
|--------|------|----------|
| `CAM_232_aug_combined_1.jpg` | 复合损伤 - 高度局部化激活 | 多缺陷 |
| `CAM_25_aug_combined_3.jpg` | 隐藏裂纹 - 识别细微缺陷 | 裂纹 |
| `CAM_72_aug_combined_3.jpg` | 划痕检测 - 沿划痕路径的强激活 | 划痕 |
| `CAM_253_aug_combined_4.jpg` | 清洁样本 - 分布式低强度背景注意 | 正常 |

### 8.3 图表质量规格

所有图表均为印刷级质量：
- **格式**: PNG (300 DPI)
- **字体**: Times New Roman (与LaTeX兼容)
- **尺寸**: 适合 IEEE Transactions 双栏格式 (0.48\textwidth)
- **色彩**: 专业的配色方案，适合学术论文发表

### 8.4 编译说明

**Overleaf (推荐)**:
1. 上传 `submission_package/` 目录下所有文件
2. 点击 **Recompile**
3. 等待编译完成

**本地编译**:
```bash
cd paper/submission_package
pdflatex ComputersInIndustry_manuscript.tex
bibtex ComputersInIndustry_manuscript
pdflatex ComputersInIndustry_manuscript.tex
pdflatex ComputersInIndustry_manuscript.tex
```

### 8.5 论文核心贡献

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

# （可选）安装数据增强与合成依赖
pip install opencv-python scikit-image diffusers accelerate
```

### 硬件需求

| 配置项 | 最低 | 推荐 |
|--------|------|------|
| GPU | NVIDIA GTX 1080 Ti（11 GB） | NVIDIA RTX 3090 / A100（24 GB+） |
| 内存 | 16 GB | 32 GB+ |
| 存储 | 50 GB | 200 GB+（含 SD 合成数据集） |

> **注意：** VGG19 在 $640 \times 640$ 分辨率下进行全参数微调约需 18–22 GB 显存。显存受限时建议启用梯度检查点（Gradient Checkpointing）或混合精度训练（`torch.cuda.amp`）。

---

## 10. 实验结果与性能分析

基于 IEEE Transactions 期刊论文的实验数据，以下是 TubeGuard-AI 各模型在统一数据集上的完整性能对比：

### 10.1 核心性能对比（完整模型库）

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | 输入分辨率 | 参数量 | 架构类型 |
|------|--------|--------|--------|---------|------------|---------|----------|
| **YOLOv10-TPH (Ours)** | **98.1250%** | **98.5385%** | **99.3794%** | **0.9896** | 1280×1280 | 2.1M | 检测+Transformer |
| ResNet50 | 89.5139% | 89.5688% | 99.9224% | 0.9446 | 640×640 | 25.6M | CNN |
| ViT-B/16 | 70.6250% | 97.1678% | 69.2009% | 0.8083 | 224×224 | 86.6M | Transformer |
| Swin-T V2 | 78.9583% | 98.3333% | 77.8123% | 0.8688 | 256×256 | 28.3M | 层次化Transformer |

### 10.2 TPH 模块消融实验

#### 10.2.1 TPH 模块的整体性能提升

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | TPH 提升 |
|------|--------|--------|--------|---------|----------|
| **YOLOv10-TPH (Ours)** | **98.1250%** | **98.5385%** | **99.3794%** | **0.9896** | **+4.57%** |
| YOLOv10 (Baseline) | 94.38% | 89.50% | 99.80% | 0.9439 | - |

**关键性能提升**：
- **精确率提升**: +9.04% (89.50% → 98.5385%)，误报率降低85%
- **召回率平衡**: 仅下降0.42% (99.80% → 99.3794%)，保持高检出率
- **F1分数提升**: +4.57%，达到精度和召回率的最佳平衡

#### 10.2.2 不同数据划分下的 TPH 效果

论文测试了 3 种数据划分策略，验证 TPH 的泛化能力：

| 数据划分 | 训练集 | 验证集 | 测试集 | TPH 提升 |
|----------|--------|--------|--------|----------|
| **标准划分** | 70% | 15% | 15% | **+4.57%** |
| 均衡划分 | 50% | 25% | 25% | **+3.89%** |
| 小样本划分 | 30% | 35% | 35% | **+5.23%** |

**数据效率分析**：
- **小样本场景**：TPH 提升最显著 (+5.23%)，证明注意力机制在数据稀缺时的优势
- **标准场景**：TPH 保持稳定提升 (+4.57%)，验证方法的可靠性
- **所有划分**：TPH 均带来正向提升，证明其鲁棒性

#### 10.2.3 TPH 模块机制分析

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

### 10.3 数据效率分析

在不同训练数据量下的性能表现（基于论文 Fig. 2 数据效率分析）：

| 数据比例 | YOLOv10-TPH | YOLOv10 | ResNet50 | ViT-B/16 |
|----------|-------------|---------|----------|----------|
| **100%** | **0.9896** | 0.9439 | 0.9446 | 0.8083 |
| **71%** | **0.9745** | 0.9287 | 0.9123 | 0.7564 |
| **58%** | **0.9567** | 0.8945 | 0.8734 | 0.6987 |

**数据效率优势**：
- 在58%数据量时，TPH仍保持0.9567 F1分数，高于基线YOLOv10在100%数据量时的性能
- 注意力机制放大有限训练样本中的弱信号，提升小样本场景下的泛化能力
- 所有数据量下，TPH均保持最高性能，证明其鲁棒性

### 10.4 数据集划分策略

论文采用 3 种数据划分策略验证模型泛化能力：

| 划分类型 | 训练集 | 验证集 | 测试集 | 特点 |
|----------|--------|--------|--------|------|
| **标准划分** | 70% (1008张) | 15% (216张) | 15% (216张) | 常规机器学习划分 |
| **均衡划分** | 50% (720张) | 25% (360张) | 25% (360张) | 验证集和测试集更大 |
| **小样本划分** | 30% (432张) | 35% (504张) | 35% (504张) | 模拟数据稀缺场景 |

**标注协议**：
- **结构性缺陷**：划痕、裂纹、CutPaste → 标记为缺陷 (1)
- **环境扰动**：老旧磨损、光照变化 → 标记为正常 (0)
- **关键原则**：只有真正影响结构完整性的破坏才被视为缺陷


### 10.5 可视化分析

论文包含 6 个核心图表：

1. **训练曲线** (`training_curves.png`): 100个epoch收敛过程，验证集性能持续优于训练集
2. **数据效率分析** (`data_efficiency_analysis.png`): TPH在不同数据量下的鲁棒性
3. **消融实验F1对比** (`ablation_study_f1_comparison.png`): TPH模块的+4.57%性能提升
4. **跨架构对比** (`cross_architecture_comparison.png`): 6个模型在3种数据划分下的全面对比
5. **性能雷达图** (`precision_recall_radar.png`): 精确率、召回率、准确率三维度可视化
6. **完整性能对比** (`full_performance_comparison.png`): 6个模型的4个指标综合柱状图

### 10.6 Grad-CAM 注意力可视化

4个Grad-CAM热力图展示模型决策过程：

- **复合损伤**: 高度局部化激活，精准定位多缺陷区域
- **隐藏裂纹**: 识别肉眼难以察觉的细微缺陷
- **划痕检测**: 沿划痕路径的强注意力响应
- **清洁样本**: 分布式低强度背景注意，无误报激活

这些可视化直接证实TPH模块：
- 激活集中在真正的缺陷形态上
- 有效抑制背景伪影（镜面反射、阴影）
- 保留空间细节进行精确分类

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
