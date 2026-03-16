#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
CutPaste 真实缺陷模拟 —— 命令行入口脚本（v2）
============================================================
支持 5 种缺陷模式：

  scratch   - 程序化划痕/裂纹（随机游走 + 高光折射 + 分叉）
  lighting  - 局部光照扰动（亮斑 / 暗区 / 条纹渐变）
  texture   - 表面纹理劣化（磨损 / 颗粒噪声 / 腐蚀斑点）
  cutpaste  - 传统 CutPaste（裁剪粘贴 + 泊松融合）
  combined  - 随机组合多种缺陷（默认，最贴近真实）

用法示例：

  # 组合模式（默认，推荐）
  python run_cutpaste.py \
      --image_dir ../../data/images \
      --csv_path  ../../data/train.csv \
      --output_dir ../../data/defect_output \
      --num_augment 3

  # 仅划痕模式
  python run_cutpaste.py --defect_mode scratch ...

  # 仅光照异常
  python run_cutpaste.py --defect_mode lighting --lighting_strength 0.6 ...
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutpaste_augmentor import CutPasteAugmentor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CutPaste 真实缺陷模拟器 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === 必要参数 ===
    parser.add_argument("--image_dir", type=str, required=True, help="原始图像目录")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 标注文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # === 缺陷模式 ===
    parser.add_argument("--defect_mode", type=str, default="combined",
                        choices=["scratch", "lighting", "texture", "cutpaste", "combined"],
                        help="缺陷模式（默认: combined）")

    # === 生成配置 ===
    parser.add_argument("--num_augment", type=int, default=3, help="每张生成数（默认: 3）")
    parser.add_argument("--target_label", type=str, default="[无缺陷]", help="正常样本标签")
    parser.add_argument("--augmented_label", type=str, default="[有缺陷]", help="增强标签")
    parser.add_argument("--augment_all", action="store_true", help="是否对整个数据集的样本进行增强（不仅限正常样本）")

    # === 划痕参数 ===
    parser.add_argument("--scratch_count_min", type=int, default=1, help="最少划痕数")
    parser.add_argument("--scratch_count_max", type=int, default=4, help="最多划痕数")
    parser.add_argument("--scratch_width_min", type=int, default=1, help="划痕最小宽度")
    parser.add_argument("--scratch_width_max", type=int, default=3, help="划痕最大宽度")

    # === 光照参数 ===
    parser.add_argument("--lighting_strength", type=float, default=0.5, help="光照扰动强度")

    # === 纹理参数 ===
    parser.add_argument("--texture_severity", type=float, default=0.4, help="纹理劣化程度")

    # === ROI 和融合 ===
    parser.add_argument("--no_roi", action="store_true", help="禁用 ROI 感知")
    parser.add_argument("--no_poisson", action="store_true", help="禁用泊松融合")

    # === 其他 ===
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    mode_desc = {
        "scratch": "🔪 划痕/裂纹（随机游走 + 高光 + 分叉）",
        "lighting": "💡 局部光照扰动（亮斑 / 暗区 / 条纹）",
        "texture": "🧱 表面纹理劣化（磨损 / 噪声 / 腐蚀）",
        "cutpaste": "✂️ 传统 CutPaste（裁剪 + 泊松融合）",
        "combined": "🎲 组合模式（随机叠加多种缺陷）"
    }

    print("=" * 60)
    print("🎨 TubeGuard-AI 真实缺陷模拟器 v2")
    print("=" * 60)
    print(f"  图像目录:    {args.image_dir}")
    print(f"  CSV 文件:    {args.csv_path}")
    print(f"  输出目录:    {args.output_dir}")
    print(f"  缺陷模式:    {mode_desc.get(args.defect_mode, args.defect_mode)}")
    print(f"  样本范围:    {'全量数据集' if args.augment_all else '仅正常样本'}")
    print(f"  每张生成数:  {args.num_augment}")
    print(f"  ROI 感知:    {'❌' if args.no_roi else '✅'}")
    print(f"  泊松融合:    {'❌' if args.no_poisson else '✅'}")

    augmentor = CutPasteAugmentor(
        defect_mode=args.defect_mode,
        scratch_count=(args.scratch_count_min, args.scratch_count_max),
        scratch_width=(args.scratch_width_min, args.scratch_width_max),
        lighting_strength=args.lighting_strength,
        texture_severity=args.texture_severity,
        roi_aware=not args.no_roi,
        use_poisson_blend=not args.no_poisson,
        seed=args.seed
    )

    start_time = time.time()

    stats = augmentor.augment_batch(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_augment_per_image=args.num_augment,
        target_label=args.target_label,
        augmented_label=args.augmented_label,
        augment_all=args.augment_all
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️  耗时: {elapsed:.1f} 秒")
    if stats["total_generated"] > 0:
        print(f"📈 平均每张: {elapsed / stats['total_generated']:.2f} 秒")
    print(f"\n🎉 完成！")


if __name__ == "__main__":
    main()
