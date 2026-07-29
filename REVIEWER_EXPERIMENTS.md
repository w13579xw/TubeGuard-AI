# 审稿意见补充实验执行说明

本流程专门对应以下三项审稿意见：

1. 增加其他 YOLO 系列与 DETR 系列模型；
2. 在独立真实测试集上单独评价；
3. 完成六组组件级消融实验。

## 1. 数据协议

- `data/train.csv`（2400 张真实图像）按类别分层划分为训练集和验证集；
- `data/test.csv`（600 张真实图像）固定为最终真实测试集；
- 合成增强图像只允许进入训练集；
- 合成图像按源图像 ID 归组，源图像若在验证集，对应合成图不会进入训练集；
- 真实测试集不参与早停、选模、硬负样本挖掘或阈值选择。

生成并审计清单：

```bash
python run_reviewer_experiments.py --mode prepare
```

输出位于 `data/reviewer_revision/`。当前固定随机种子 42 的清单规模为：

| 清单 | 样本数 | 用途 |
|---|---:|---|
| train_original | 2040 | 真实训练样本 |
| train_augmented | 10200 | 仅训练使用的合成样本 |
| val | 360 | 早停与选模 |
| real_test | 600 | 最终独立评价 |

## 2. YOLO/DETR 系列横向对比

默认模型：

- YOLOv8n
- YOLOv10n
- YOLO11n
- DETR-R50
- Deformable-DETR-R50
- YOLOv10-TPH（本文模型）

所有模型统一采用相同的训练/验证/真实测试清单、输入尺寸、优化器、随机种子和指标实现。由于本项目标签为图像级缺陷标签而非边界框，YOLO 与 DETR 均使用其预训练视觉骨干/编码器，加全局池化二分类头进行公平的图像级比较。

```bash
python run_reviewer_experiments.py --mode comparison \
  --epochs 100 --batch-size 16 --workers 8 --amp
```

训练日志会实时同时输出到终端和以下文件，重复运行时追加新的时间戳区段：

```text
log/reviewer_revision/comparison_yolov8n.log
log/reviewer_revision/comparison_yolov10n.log
log/reviewer_revision/comparison_yolo11n.log
log/reviewer_revision/comparison_detr_r50.log
log/reviewer_revision/comparison_deformable_detr_r50.log
log/reviewer_revision/comparison_yolov10_tph.log
```

可分卡运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run_reviewer_experiments.py --mode comparison --models yolov8n yolov10n --amp
CUDA_VISIBLE_DEVICES=1 python run_reviewer_experiments.py --mode comparison --models yolo11n detr_r50 --amp
CUDA_VISIBLE_DEVICES=2 python run_reviewer_experiments.py --mode comparison --models deformable_detr_r50 yolov10_tph --amp
```

## 3. 六组组件级消融

| 参数值 | 模型配置 |
|---|---|
| `yolov10_only` | YOLOv10 only |
| `yolov10_tph` | YOLOv10 + TPH |
| `yolov10_augmentation` | YOLOv10 + augmentation |
| `yolov10_hnm` | YOLOv10 + hard negative mining |
| `yolov10_tph_augmentation` | YOLOv10 + TPH + augmentation |
| `complete` | YOLOv10 + TPH + augmentation + hard negative mining |

```bash
python run_reviewer_experiments.py --mode ablation \
  --epochs 100 --batch-size 16 --workers 8 --amp
```

每项消融同样写入独立日志，例如
`log/reviewer_revision/ablation_complete.log`。

硬负样本挖掘默认从第 5 轮开始，仅对真实训练集中的无缺陷图像排序，选取缺陷误报概率最高的 25%，下一轮赋予 3 倍采样权重。每轮入选样本及置信度保存在对应实验目录，便于论文复核。

## 4. 独立真实测试集汇总

每个模型训练结束后会自动释放训练模型、加载最佳权重并评价真实测试集，无需另跑评估命令。每个实验生成：

```text
runs/reviewer_revision/<experiment>/real_test_metrics.json
runs/reviewer_revision/<experiment>/real_test_metrics.csv
```

同时自动增量更新总表（相同实验重复运行时覆盖旧行）：

```text
runs/reviewer_revision/real_test_summary.csv
```

总表写入使用文件锁，因此可以安全地分卡并行训练。也可在已有权重上统一重新评价：

```bash
python run_reviewer_experiments.py --mode evaluate \
  --batch-size 16 --workers 8 --amp
```

报告指标包括 Accuracy、Precision、Recall、F1、Specificity、ROC-AUC 和完整混淆矩阵计数（TP/FP/TN/FN）。论文中应只填写实际生成的结果，不应沿用旧脚本中的示例数值。

## 5. 环境

```bash
pip install -r requirements-training.txt
```

DETR 首次运行会从 Hugging Face 下载官方预训练权重；YOLO 首次运行会下载对应 Ultralytics 权重。
