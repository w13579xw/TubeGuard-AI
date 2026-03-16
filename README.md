# TubeGuard-AI 🔬

> **工业透明/半透明管道精密缺陷检测框架**
>
> 融合经典计算机视觉几何算法、自监督异常检测与前沿深度学习，面向高分辨率（$4024 \times 3036$）工业管道图像的微小缺陷（裂纹、气泡、划痕等）识别与分类研究平台。各模型依据架构与显存约束，在 $224^2$ ～ $1280^2$ 范围内对原图进行缩放训练。

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 目录

1. [项目简介](#1-项目简介)
2. [仓库结构](#2-仓库结构)
3. [基线模型库（Model Zoo）](#3-基线模型库model-zoo)
4. [未来工作——数据增强与合成](#4-未来工作数据增强与合成)
5. [未来工作——GFC 混合检测架构](#5-未来工作gfc-混合检测架构)
6. [环境配置](#6-环境配置)
7. [引用](#7-引用)

---

## 1. 项目简介

工业透明/半透明管道质检面临三项核心挑战，构成本项目的研究动机：

| 挑战 | 描述 |
|------|------|
| **超高分辨率** | 原始采集图像分辨率达 $4024 \times 3036$（约 1200 万像素），微小缺陷仅占数个像素，细节极易在降采样中丢失 |
| **长尾样本分布** | 缺陷样本极度稀缺，正负样本比例严重失衡 |
| **可解释性鸿沟** | 纯深度学习"黑盒"模型难以满足工业安全认证要求，需融合物理先验知识 |

TubeGuard-AI 通过**三个协同模块**系统性地应对上述挑战：

```
TubeGuard-AI
├── 🧠  基线模型库    — 系统化对比实验（CNN / Transformer / YOLO 三大家族）
├── 🎨  数据合成      — CutPaste 自监督生成 & Stable Diffusion 图像合成
└── ⚙️  GFC 架构      — 几何筛查 + 语义分类两阶段工业级检测流水线
```

---

## 2. 仓库结构

```
TubeGuard-AI/
│
├── NN/                              # 神经网络基线模型库
│   ├── ResNet50/                    # CNN 基线（ImageNet 预训练）
│   │   ├── resnet_model.py
│   │   └── train_resnet.py
│   ├── VGG19/                       # 深度 CNN 对比（高显存消耗）
│   │   ├── vgg_model.py
│   │   └── train_vgg.py
│   ├── ViT/                         # Vision Transformer（位置编码双三次插值）
│   │   ├── vit_model.py
│   │   └── train_vit.py
│   ├── Swin Transformer V2/         # 层次化窗口 Transformer
│   │   ├── swin_model.py
│   │   └── train_swin.py
│   ├── yolov10/                     # YOLOv10 目标检测基线
│   │   ├── model.py
│   │   └── train.py
│   └── yolov10_tph/                 # YOLOv10 + TPH 微小目标增强版
│       ├── model.py
│       ├── utils.py
│       ├── predict.py
│       └── train.py
│
├── TubeGuard_GFC_System/            # GFC 混合检测系统（核心生产级模块）
│   ├── config/
│   │   └── gfc_config.yaml
│   ├── src/
│   │   ├── core_algorithms.py       # Steger / Zernike 亚像素算法
│   │   ├── stage_a_geometric.py     # 阶段 A：几何筛查（σ_L、ΔT、ε）
│   │   └── stage_b_semantic.py      # 阶段 B：PatchCore + YOLOv10-CBAM 分类
│   ├── utils/
│   └── main_pipeline.py
│
├── preprocessing/                   # 数据预处理脚本
├── data/                            # 数据集（原始 / 划分后）
├── raw_data/                        # 原始采集图像
├── processed_data/                  # 预处理后数据
├── logs/                            # 训练日志与指标记录
├── res/                             # 结果输出（图表、报告）
└── error_images/                    # 误分类样本（坏例捕捉）
```

---

## 3. 基线模型库（Model Zoo）

所有模型共享统一的工程基础设施：

- ✅ 完整训练循环 + **早停机制**（基于验证集 patience）
- ✅ **余弦退火（Cosine Annealing）** 学习率调度
- ✅ 逐 Epoch 模型检查点保存（`*_best.pth` / `*_checkpoint.pth`）
- ✅ **混淆矩阵**可视化 & **分类报告**（CSV 导出）
- ✅ **坏例分析（Bad Case Analysis）**——自动捕获误分类样本并记录至 CSV

---

### 3.1 CNN 家族

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

### 3.2 Transformer 家族

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

### 3.3 YOLO 家族

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

### 3.4 基线模型汇总

| 模型 | 范式 | 输入分辨率 | 预训练来源 | 核心优势 |
|------|------|------------|------------|----------|
| ResNet50 | CNN | 灵活 | ImageNet-1K | 局部纹理建模，强基线 |
| VGG19 | CNN | 灵活 | ImageNet-1K | 深度对比实验 |
| ViT | Transformer | 640²（PE 插值） | ImageNet-21K | 全局上下文建模 |
| Swin-T V2 | 层次化 Transformer | 原生高分辨率 | ImageNet-22K | 高分辨率计算效率 |
| YOLOv10 | 目标检测 | 640² | COCO | 定位基线 |
| YOLOv10-TPH | 检测 + Transformer 头 | 640² | COCO | 微小目标召回率 |

---

## 4. 未来工作——数据增强与合成

工业质检的**极端长尾类别分布**是制约监督模型性能的核心瓶颈。本项目提出两种互补的数据合成策略：

### 4.1 CutPaste 自监督伪缺陷生成

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

### 4.2 Stable Diffusion 图像合成

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

## 5. 未来工作——GFC 混合检测架构

**GFC（Geometric Filtering & Classification，几何筛查与语义分类）** 系统是 TubeGuard-AI 的生产级核心架构。它将**基于物理规律的几何推理**与**AI 语义分类**融合为一套两阶段工业检测流水线，直接解决深度学习缺乏物理可解释性和存在算力浪费的痛点。

```
输入帧（原始 4024×3036，预处理后缩放）
        │
        ▼
╔══════════════════════════════════════════╗
║       阶段 A — 几何第一道防线              ║  ← 物理先验驱动
║                                          ║
║  · 亚像素精度边缘提取                     ║
║    （Steger 算法 / Zernike 矩方法）       ║
║                                          ║
║  · 三项几何偏差指标计算：                  ║
║    - σ_L  线性偏离度                      ║
║    - ΔT   壁厚不均度                      ║
║    - ε    截面收缩率                      ║
║                                          ║
║  超出阈值？                               ║
║    是 → ❌ 直接剔除（宏观形变废品）         ║
║    否 → 进入阶段 B                        ║
╚══════════════════════════════════════════╝
        │（几何合格）
        ▼
╔══════════════════════════════════════════╗
║       阶段 B — AI 第二道防线              ║  ← 数据驱动语义理解
║                                          ║
║  1. PatchCore（无监督异常筛查）            ║
║     · 构建正常区域 patch 特征记忆库        ║
║     · 逐 patch 计算异常分数               ║
║     · 定位可疑区域并裁剪为 ROI            ║
║                                          ║
║  2. YOLOv10 + CBAM（强监督精准分类）      ║
║     · 输入：PatchCore 报警的 ROI 区域     ║
║     · CBAM：通道 + 空间双重注意力重标定   ║
║     · 输出：缺陷类别 + 边界框坐标         ║
╚══════════════════════════════════════════╝
        │
        ▼
最终决策：{合格 | 缺陷类型 | 未知异常}
```

### 5.1 阶段 A — 物理几何筛查（`stage_a_geometric.py`）

利用 **Steger 算法**与 **Zernike 矩**方法实现亚像素精度管壁轮廓提取，逐帧计算三项偏差指标：

| 指标 | 符号 | 物理含义 | 拒绝准则 |
|------|------|----------|----------|
| 线性偏离度 | $\sigma_L$ | 管壁中心线直线度 | $\sigma_L > \tau_L$ |
| 壁厚不均度 | $\Delta T$ | 截面处管壁厚度方差 | $\Delta T > \tau_T$ |
| 截面收缩率 | $\epsilon$ | 局部截面面积缩小比例 | $\epsilon > \tau_\epsilon$ |

任意指标超出阈值的帧将被**即时熔断剔除**，无需调用神经网络推理，同时提供完整可追溯的拒绝原因。

### 5.2 阶段 B — AI 语义分类（`stage_b_semantic.py`）

几何合格的图像进入两层 AI 防线：

**第一层 — PatchCore（无监督异常检测）**
- 以预训练主干网络构建正常区域 patch 嵌入的记忆库
- 推理时通过最近邻搜索计算每个 patch 的异常分数，生成异常热力图
- 自动定位并裁剪疑似区域作为后续分类的 ROI 候选框

**第二层 — YOLOv10 + CBAM（强监督精准分类）**
- 接收 PatchCore 报警的 ROI 裁剪图像
- **CBAM（Convolutional Block Attention Module）** 对通道维度和空间维度分别进行注意力重标定，使检测器聚焦于透明管壁背景中的缺陷像素
- 输出缺陷类别（裂纹 / 气泡 / 划痕 / 其他）与精确边界框坐标

### 5.3 架构设计动机

| 设计决策 | 工程动机 |
|----------|----------|
| 几何在 AI 之前 | 对显著宏观形变废品直接熔断，节省 DL 推理算力；拒绝原因可审计、可追溯 |
| PatchCore 在 YOLOv10 之前 | 捕获监督分类器会遗漏的分布外/新型异常，构建开集安全网 |
| YOLOv10 引入 CBAM | 在复杂透明管壁背景中将注意力精准导向缺陷像素 |
| 两阶段解耦部署 | 每个阶段可独立验证与工业认证，降低整体系统审计复杂度 |

---

## 6. 环境配置

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

# （可选）安装 GFC 系统附加依赖
pip install opencv-python scikit-image anomalib ultralytics diffusers accelerate
```

### 硬件需求

| 配置项 | 最低 | 推荐 |
|--------|------|------|
| GPU | NVIDIA GTX 1080 Ti（11 GB） | NVIDIA RTX 3090 / A100（24 GB+） |
| 内存 | 16 GB | 32 GB+ |
| 存储 | 50 GB | 200 GB+（含 SD 合成数据集） |

> **注意：** VGG19 在 $640 \times 640$ 分辨率下进行全参数微调约需 18–22 GB 显存。显存受限时建议启用梯度检查点（Gradient Checkpointing）或混合精度训练（`torch.cuda.amp`）。

---

## 7. 引用

如您在研究中使用了 TubeGuard-AI，请引用：

```bibtex
@misc{tubeguard_ai_2025,
  title  = {TubeGuard-AI: A Geometric-Semantic Hybrid Framework for Precision
            Defect Detection in Industrial Transparent Pipelines},
  author = {作者姓名 and 合作者},
  year   = {2025},
  url    = {https://github.com/your-org/TubeGuard-AI},
  note   = {GitHub repository}
}
```

---

<div align="center">

**TubeGuard-AI** · 为工业智能质检而生 ❤️

*以经典几何算法与现代深度学习的融合，守护安全关键制造的每一道工序*

</div>
