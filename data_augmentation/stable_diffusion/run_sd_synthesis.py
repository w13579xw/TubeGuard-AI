#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Stable Diffusion 缺陷图像合成 —— 命令行入口脚本（调研报告增强版）
============================================================
根据调研报告新增：
    --hf_mirror       HF 镜像（默认 hf-mirror.com）
    --lora_path       LoRA 权重文件路径
    --trigger_word    LoRA Trigger Word
    --use_crack_mask  使用裂痕形态掩膜
    --enable_ssim     SSIM 质量筛选

用法示例：

  # Img2Img 模式 + HF 镜像（默认）
  python run_sd_synthesis.py \
      --image_dir ../../data/images \
      --csv_path  ../../data/train.csv \
      --output_dir ../../data/sd_output \
      --num_generate 2 --strength 0.5

  # Inpainting + 裂痕形态掩膜
  python run_sd_synthesis.py \
      --image_dir ../../data/images \
      --csv_path  ../../data/train.csv \
      --output_dir ../../data/sd_inpaint \
      --mode inpainting --use_crack_mask

  # 加载 LoRA 微调权重
  python run_sd_synthesis.py \
      --image_dir ../../data/images \
      --csv_path  ../../data/train.csv \
      --output_dir ../../data/sd_lora \
      --lora_path ./lora_weights/crack_lora.safetensors \
      --trigger_word "[V*] crack"

  # 快速测试（5 张图像）
  python run_sd_synthesis.py \
      --image_dir ../../data/images \
      --csv_path  ../../data/train.csv \
      --output_dir ../../data/sd_test \
      --num_generate 1 --max_images 5
"""

import argparse
import time
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_synthesizer import StableDiffusionSynthesizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 工业缺陷图像合成器（调研报告增强版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === 必要参数 ===
    parser.add_argument("--image_dir", type=str, required=True, help="原始图像目录")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 标注文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # === 模型配置 ===
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="HF 模型 ID 或本地路径")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp32", action="store_true", help="使用 FP32")

    # === 报告新增：HF 镜像 ===
    parser.add_argument("--hf_mirror", type=str, default="https://hf-mirror.com",
                        help="HF 镜像 URL（默认: hf-mirror.com）")
    parser.add_argument("--no_mirror", action="store_true", help="禁用 HF 镜像")

    # === 报告新增：LoRA ===
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重文件路径")
    parser.add_argument("--lora_scale", type=float, default=0.8,
                        help="LoRA 缩放因子（默认: 0.8）")
    parser.add_argument("--trigger_word", type=str, default="[V*]",
                        help="LoRA Trigger Word（默认: [V*]）")

    # === 合成配置 ===
    parser.add_argument("--mode", type=str, default="img2img",
                        choices=["img2img", "inpainting"])
    parser.add_argument("--num_generate", type=int, default=2, help="每张生成数")
    parser.add_argument("--custom_prompt", type=str, default=None, help="自定义提示词")
    parser.add_argument("--strength", type=float, default=0.5, help="Img2Img 强度")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导系数")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--target_size", type=int, default=512, help="SD 推理尺寸")

    # === 报告新增：裂痕掩膜 ===
    parser.add_argument("--use_crack_mask", action="store_true", default=True,
                        help="Inpainting 模式使用裂痕形态掩膜（默认开启）")
    parser.add_argument("--no_crack_mask", action="store_true",
                        help="使用随机矩形掩膜")

    # === 质量筛选 ===
    parser.add_argument("--enable_lpips", action="store_true", help="启用 LPIPS")
    parser.add_argument("--lpips_threshold", type=float, default=0.1)
    parser.add_argument("--enable_ssim", action="store_true", default=True,
                        help="启用 SSIM 筛选（默认开启）")
    parser.add_argument("--no_ssim", action="store_true", help="禁用 SSIM 筛选")
    parser.add_argument("--ssim_max", type=float, default=0.95,
                        help="SSIM 上限（默认: 0.95）")

    # === 数据 ===
    parser.add_argument("--source_label", type=str, default="[无缺陷]")
    parser.add_argument("--generated_label", type=str, default="[有缺陷]")
    parser.add_argument("--max_images", type=int, default=None, help="最大处理数")
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    dtype = torch.float32 if args.fp32 else torch.float16
    hf_mirror = args.hf_mirror if not args.no_mirror else None
    use_crack_mask = args.use_crack_mask and not args.no_crack_mask
    enable_ssim = args.enable_ssim and not args.no_ssim

    print("=" * 60)
    print("🎨 TubeGuard-AI SD 缺陷合成器（调研报告增强版）")
    print("=" * 60)
    print(f"  模型:         {args.model_id}")
    print(f"  设备:         {args.device}")
    print(f"  精度:         {'FP32' if args.fp32 else 'FP16'}")
    print(f"  HF 镜像:     {hf_mirror or '未使用'}")
    print(f"  合成模式:     {args.mode}")
    print(f"  LoRA:         {args.lora_path or '未使用'}")
    print(f"  Trigger Word: {args.trigger_word}")
    print(f"  裂痕掩膜:     {'✅' if use_crack_mask else '❌'}")
    print(f"  SSIM 筛选:    {'✅' if enable_ssim else '❌'}")
    print(f"  LPIPS 筛选:   {'✅' if args.enable_lpips else '❌'}")
    print(f"  每张生成数:   {args.num_generate}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n⚠️ CUDA 不可用，自动切换至 CPU")
        args.device = "cpu"
        dtype = torch.float32

    synthesizer = StableDiffusionSynthesizer(
        model_id=args.model_id,
        device=args.device,
        dtype=dtype,
        enable_lpips=args.enable_lpips,
        lpips_threshold=args.lpips_threshold,
        enable_ssim=enable_ssim,
        ssim_max=args.ssim_max,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        trigger_word=args.trigger_word,
        hf_mirror=hf_mirror,
        seed=args.seed
    )

    start_time = time.time()

    stats = synthesizer.synthesize_batch(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_generate_per_image=args.num_generate,
        mode=args.mode,
        custom_prompt=args.custom_prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        target_size=args.target_size,
        source_label=args.source_label,
        generated_label=args.generated_label,
        max_images=args.max_images,
        use_crack_mask=use_crack_mask
    )

    elapsed = time.time() - start_time

    print(f"\n⏱️  总耗时:       {elapsed:.1f} 秒")
    if stats["total_generated"] > 0:
        print(f"📈 平均每张:     {elapsed / stats['total_generated']:.2f} 秒")
    print(f"📊 质量过滤数:   {stats.get('total_filtered', 0)}")
    print(f"\n🎉 全部完成！")


if __name__ == "__main__":
    main()
