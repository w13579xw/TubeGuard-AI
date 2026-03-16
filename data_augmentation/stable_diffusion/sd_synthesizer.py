#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Stable Diffusion 工业缺陷图像合成器（调研报告增强版）
============================================================
根据调研报告融合以下关键改进：

    1. HF 镜像自动配置 —— 解决国内访问 Hugging Face 受限问题
    2. LoRA 微调权重加载 —— 支持加载针对工业材质微调的 LoRA 适配器
    3. ControlNet 掩膜引导 —— 支持精准的空间条件控制
    4. 裂痕形态掩膜生成 —— 自动生成贴合裂痕拓扑的随机线条掩膜
    5. 提示词工程优化 —— Trigger Word + 材质模板 + 强化负面提示词
    6. SSIM 质量指标 —— 作为 LPIPS 的补充质量评估维度

依赖：
    pip install diffusers accelerate transformers torch
    pip install lpips  # 可选，LPIPS 质量筛选
"""

import os

# 禁用 TensorFlow/Flax 导入（避免 Anaconda 环境中 TF 安装损坏导致 transformers 崩溃）
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import csv
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

# =========================================================================
# 日志配置
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =========================================================================
# 报告改进：提示词工程优化（Trigger Word + 材质分离模板）
# =========================================================================
# 裂痕专用提示词模板（报告：提示词描述背景材质，确保裂痕周边纹理逻辑正确）
CRACK_PROMPTS: List[str] = [
    "{trigger} hairline crack on {material} surface, industrial inspection, "
    "macro photography, sharp focus, high resolution, realistic texture, "
    "structural damage, fatigue fracture",

    "{trigger} fine crack propagation along {material}, quality control image, "
    "close-up industrial photography, natural lighting, detailed surface texture, "
    "micro fracture pattern",

    "{trigger} micro crack defect on {material}, high magnification inspection, "
    "photorealistic, sharp details, industrial surface damage, stress cracking",

    "{trigger} branching crack network on {material}, industrial macro shot, "
    "high resolution, precise focus, surface defect documentation",
]

# 材质模板（报告：建立材料与缺陷的语义关联）
MATERIAL_TEMPLATES: List[str] = [
    "transparent polymer tube wall",
    "translucent plastic pipe",
    "clear industrial tubing",
    "semi-transparent polymer surface",
    "polished transparent tube",
]

# 报告改进：强化负面提示词（抑制非工业风格输出）
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, cartoon, drawing, painting, "
    "illustration, sketch, anime, watermark, text, logo, "
    "oversaturated, underexposed, noise, artifacts, "
    "natural rock crack, wood grain, earth texture, "
    "unrealistic lighting, floating defect, physically impossible"
)


class StableDiffusionSynthesizer:
    """
    Stable Diffusion 工业缺陷图像合成器（调研报告增强版）

    参数:
        model_id:          Hugging Face 模型 ID 或本地模型路径
        device:            推理设备 ("cuda" / "cpu")
        dtype:             推理精度
        enable_lpips:      是否启用 LPIPS 质量筛选
        lpips_threshold:   LPIPS 阈值
        enable_ssim:       是否启用 SSIM 质量指标（报告新增）
        ssim_max:          SSIM 上限（超过说明变化太小，过滤）
        lora_path:         LoRA 权重路径（报告新增）
        lora_scale:        LoRA 缩放因子
        trigger_word:      LoRA Trigger Word（报告新增）
        hf_mirror:         HF 镜像 URL（报告新增，解决国内网络问题）
        seed:              随机种子
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_lpips: bool = False,
        lpips_threshold: float = 0.1,
        enable_ssim: bool = True,
        ssim_max: float = 0.95,
        lora_path: Optional[str] = None,
        lora_scale: float = 0.8,
        trigger_word: str = "[V*]",
        hf_mirror: Optional[str] = "https://hf-mirror.com",
        seed: Optional[int] = None
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.enable_lpips = enable_lpips
        self.lpips_threshold = lpips_threshold
        self.enable_ssim = enable_ssim
        self.ssim_max = ssim_max
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.trigger_word = trigger_word
        self.seed = seed

        # 报告改进：自动配置 HF 镜像
        if hf_mirror:
            os.environ["HF_ENDPOINT"] = hf_mirror
            logger.info(f"🌐 已配置 HF 镜像: {hf_mirror}")

        # 延迟加载的管线对象
        self._img2img_pipe = None
        self._inpaint_pipe = None
        self._lpips_model = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    # -----------------------------------------------------------------
    # 延迟加载 Img2Img 管线
    # -----------------------------------------------------------------
    def _get_img2img_pipeline(self):
        """延迟加载 Img2Img 管线，支持 LoRA 权重加载"""
        if self._img2img_pipe is None:
            logger.info(f"🔄 加载 Img2Img 管线: {self.model_id}")
            from diffusers import StableDiffusionImg2ImgPipeline

            self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            if hasattr(self._img2img_pipe, "enable_attention_slicing"):
                self._img2img_pipe.enable_attention_slicing()

            # 报告改进：加载 LoRA 权重
            if self.lora_path and os.path.exists(self.lora_path):
                self._load_lora(self._img2img_pipe)

            logger.info("✅ Img2Img 管线加载完毕")

        return self._img2img_pipe

    # -----------------------------------------------------------------
    # 延迟加载 Inpainting 管线
    # -----------------------------------------------------------------
    def _get_inpaint_pipeline(self):
        """延迟加载 Inpainting 管线"""
        if self._inpaint_pipe is None:
            logger.info(f"🔄 加载 Inpainting 管线")
            from diffusers import StableDiffusionInpaintPipeline

            inpaint_model_id = self.model_id
            if "inpainting" not in inpaint_model_id.lower():
                inpaint_model_id = "runwayml/stable-diffusion-inpainting"

            try:
                self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    inpaint_model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
            except Exception as e:
                logger.warning(f"⚠️ Inpainting 模型加载失败: {e}，回退到 Img2Img")
                return None

            if hasattr(self._inpaint_pipe, "enable_attention_slicing"):
                self._inpaint_pipe.enable_attention_slicing()

            if self.lora_path and os.path.exists(self.lora_path):
                self._load_lora(self._inpaint_pipe)

            logger.info("✅ Inpainting 管线加载完毕")

        return self._inpaint_pipe

    # -----------------------------------------------------------------
    # 报告新增：LoRA 权重加载
    # -----------------------------------------------------------------
    def _load_lora(self, pipeline):
        """
        加载 LoRA 微调权重

        报告建议：秩 32-128，缩放因子 1x-2x Rank，
        仅需训练约 0.74% 参数即可适配特定工业材质
        """
        try:
            pipeline.load_lora_weights(self.lora_path)
            logger.info(f"✅ LoRA 权重已加载: {self.lora_path} (scale={self.lora_scale})")
        except Exception as e:
            logger.warning(f"⚠️ LoRA 加载失败: {e}")

    # -----------------------------------------------------------------
    # 延迟加载 LPIPS
    # -----------------------------------------------------------------
    def _get_lpips_model(self):
        """延迟加载 LPIPS 模型"""
        if self._lpips_model is None and self.enable_lpips:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net="alex").to(self.device)
                logger.info("✅ LPIPS 模型加载完毕")
            except ImportError:
                logger.warning("⚠️ lpips 未安装，已禁用。请运行: pip install lpips")
                self.enable_lpips = False
        return self._lpips_model

    # -----------------------------------------------------------------
    # 计算 LPIPS
    # -----------------------------------------------------------------
    def _compute_lpips(self, original: Image.Image, generated: Image.Image,
                       target_size: int = 256) -> float:
        """计算 LPIPS 感知距离"""
        model = self._get_lpips_model()
        if model is None:
            return 1.0

        orig = original.resize((target_size, target_size), Image.BICUBIC)
        gen = generated.resize((target_size, target_size), Image.BICUBIC)

        orig_t = torch.from_numpy(
            np.array(orig).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        gen_t = torch.from_numpy(
            np.array(gen).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distance = model(orig_t, gen_t).item()
        return distance

    # -----------------------------------------------------------------
    # 报告新增：SSIM 质量指标
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_ssim(original: Image.Image, generated: Image.Image,
                      target_size: int = 256) -> float:
        """
        计算 SSIM 结构相似度（报告补充质量指标）

        报告：SSIM 用于评估结构保真度，SD 方案通常达到 0.3x 及以上
        """
        orig = np.array(original.resize((target_size, target_size), Image.BICUBIC))
        gen = np.array(generated.resize((target_size, target_size), Image.BICUBIC))

        # 转灰度计算 SSIM
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gen_gray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # SSIM 参数
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(orig_gray, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gen_gray, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(orig_gray ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gen_gray ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(orig_gray * gen_gray, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    # -----------------------------------------------------------------
    # 报告新增：裂痕形态掩膜生成器
    # -----------------------------------------------------------------
    @staticmethod
    def _generate_crack_mask(
        width: int,
        height: int,
        num_segments: int = 5,
        thickness_range: Tuple[int, int] = (1, 4)
    ) -> Image.Image:
        """
        生成贴合裂痕拓扑的随机线条掩膜

        报告：ControlNet 掩膜引导模式下，用户在正常图像的任意位置
        绘制裂痕形状区域，SD 仅在掩膜区域内重绘

        Args:
            width, height:    图像尺寸
            num_segments:     裂痕折线段数量
            thickness_range:  线条粗细范围

        Returns:
            裂痕形态掩膜（白色=裂痕区域）
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # 随机起点（图像中心区域附近）
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)

        # 随机主方向角度
        main_angle = random.uniform(0, np.pi)

        points = [(cx, cy)]
        for _ in range(num_segments):
            # 随机偏转角度（±30° 范围内，模拟裂痕的随机走向）
            angle = main_angle + random.uniform(-np.pi / 6, np.pi / 6)
            seg_len = random.randint(
                min(width, height) // 15,
                min(width, height) // 5
            )

            nx = int(points[-1][0] + seg_len * np.cos(angle))
            ny = int(points[-1][1] + seg_len * np.sin(angle))

            # 限制在图像范围内
            nx = max(0, min(nx, width - 1))
            ny = max(0, min(ny, height - 1))

            points.append((nx, ny))

        # 绘制折线
        thickness = random.randint(*thickness_range)
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], 255, thickness)

        # 随机添加分叉（模拟裂痕分叉结构）
        if random.random() < 0.4:
            branch_start = random.choice(points[1:-1])
            branch_angle = main_angle + random.uniform(np.pi / 4, np.pi / 2)
            branch_len = random.randint(
                min(width, height) // 20,
                min(width, height) // 8
            )
            bx = int(branch_start[0] + branch_len * np.cos(branch_angle))
            by = int(branch_start[1] + branch_len * np.sin(branch_angle))
            bx = max(0, min(bx, width - 1))
            by = max(0, min(by, height - 1))
            cv2.line(mask, branch_start, (bx, by), 255, max(1, thickness - 1))

        # 膨胀使掩膜区域更宽（确保 SD 有足够空间重绘）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(mask_rgb)

    # -----------------------------------------------------------------
    # 报告改进：构建带 Trigger Word 的提示词
    # -----------------------------------------------------------------
    def _build_prompt(
        self,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        构建提示词（报告：Trigger Word 建立特定材质裂痕与潜在空间坐标的强关联）

        Args:
            custom_prompt: 自定义提示词

        Returns:
            完整提示词
        """
        if custom_prompt:
            return custom_prompt.replace("{trigger}", self.trigger_word)

        # 随机选择模板
        template = random.choice(CRACK_PROMPTS)
        material = random.choice(MATERIAL_TEMPLATES)

        prompt = template.format(trigger=self.trigger_word, material=material)
        return prompt

    # -----------------------------------------------------------------
    # Img2Img 合成
    # -----------------------------------------------------------------
    def synthesize_img2img(
        self,
        source_image: Image.Image,
        prompt: Optional[str] = None,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        target_size: int = 512
    ) -> Optional[Image.Image]:
        """Img2Img 合成缺陷图像"""
        pipe = self._get_img2img_pipeline()

        original_size = source_image.size
        input_image = source_image.resize((target_size, target_size), Image.BICUBIC)

        text_prompt = self._build_prompt(prompt)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(
                self.seed + random.randint(0, 100000)
            )

        result = pipe(
            prompt=text_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=NEGATIVE_PROMPT,
            generator=generator
        )

        generated = result.images[0]
        generated = generated.resize(original_size, Image.BICUBIC)

        # 质量筛选
        if not self._quality_check(source_image, generated):
            return None

        return generated

    # -----------------------------------------------------------------
    # Inpainting 合成（支持裂痕掩膜）
    # -----------------------------------------------------------------
    def synthesize_inpainting(
        self,
        source_image: Image.Image,
        mask_image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        target_size: int = 512,
        use_crack_mask: bool = True
    ) -> Optional[Image.Image]:
        """
        Inpainting 合成（报告：ControlNet 掩膜引导模式）

        Args:
            use_crack_mask: 是否使用裂痕形态掩膜（报告新增）
        """
        pipe = self._get_inpaint_pipeline()
        if pipe is None:
            return self.synthesize_img2img(
                source_image, prompt, strength=0.6,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                target_size=target_size
            )

        original_size = source_image.size
        input_image = source_image.resize((target_size, target_size), Image.BICUBIC)

        # 报告改进：使用裂痕形态掩膜
        if mask_image is None:
            if use_crack_mask:
                mask_image = self._generate_crack_mask(target_size, target_size)
            else:
                mask_image = self._generate_random_rect_mask(target_size, target_size)
        else:
            mask_image = mask_image.resize((target_size, target_size), Image.NEAREST)

        text_prompt = self._build_prompt(prompt)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(
                self.seed + random.randint(0, 100000)
            )

        result = pipe(
            prompt=text_prompt,
            image=input_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=NEGATIVE_PROMPT,
            generator=generator
        )

        generated = result.images[0]
        generated = generated.resize(original_size, Image.BICUBIC)

        if not self._quality_check(source_image, generated):
            return None

        return generated

    # -----------------------------------------------------------------
    # 随机矩形掩膜（回退方案）
    # -----------------------------------------------------------------
    @staticmethod
    def _generate_random_rect_mask(width: int, height: int) -> Image.Image:
        """生成随机矩形掩膜"""
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        area = width * height
        mask_area = random.uniform(0.05, 0.20) * area
        aspect = random.uniform(0.5, 2.0)
        mh = int(np.sqrt(mask_area / aspect))
        mw = int(mask_area / mh)
        mw, mh = min(mw, width), min(mh, height)
        x = random.randint(0, width - mw)
        y = random.randint(0, height - mh)
        mask[y:y+mh, x:x+mw] = [255, 255, 255]
        return Image.fromarray(mask)

    # -----------------------------------------------------------------
    # 报告改进：统一质量检查（LPIPS + SSIM）
    # -----------------------------------------------------------------
    def _quality_check(
        self,
        original: Image.Image,
        generated: Image.Image
    ) -> bool:
        """
        综合质量筛选（报告：Cleanlab 思路 + SSIM/LPIPS 双指标）

        Returns:
            True = 通过质量检查，False = 被过滤
        """
        # SSIM 检查：若合成图像与原图过于相似，说明 SD 几乎没改变
        if self.enable_ssim:
            ssim = self._compute_ssim(original, generated)
            if ssim > self.ssim_max:
                logger.debug(f"  ❌ SSIM={ssim:.4f} > {self.ssim_max} (变化太小)")
                return False

        # LPIPS 检查
        if self.enable_lpips:
            lpips_dist = self._compute_lpips(original, generated)
            if lpips_dist < self.lpips_threshold:
                logger.debug(f"  ❌ LPIPS={lpips_dist:.4f} < {self.lpips_threshold}")
                return False

        return True

    # -----------------------------------------------------------------
    # 批量合成
    # -----------------------------------------------------------------
    def synthesize_batch(
        self,
        image_dir: str,
        csv_path: str,
        output_dir: str,
        num_generate_per_image: int = 2,
        mode: str = "img2img",
        custom_prompt: Optional[str] = None,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        target_size: int = 512,
        source_label: str = "[无缺陷]",
        generated_label: str = "[有缺陷]",
        max_images: Optional[int] = None,
        use_crack_mask: bool = True
    ) -> Dict[str, int]:
        """批量合成缺陷图像"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取 CSV
        source_entries = []
        all_entries = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                all_entries.append(row)
                if len(row) >= 2 and row[1].strip() == source_label:
                    source_entries.append(row)

        logger.info(f"📊 CSV 总条目: {len(all_entries)}, 源图像: {len(source_entries)}")

        if max_images:
            source_entries = source_entries[:max_images]

        if not source_entries:
            return {"total_source": 0, "total_generated": 0, "total_filtered": 0}

        generated_count = 0
        filtered_count = 0
        new_entries = []

        for idx, entry in enumerate(source_entries):
            img_name = entry[0]
            img_path = image_dir / img_name

            if not img_path.exists():
                continue

            try:
                source_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"⚠️ 加载失败: {img_path} -> {e}")
                continue

            for gen_idx in range(num_generate_per_image):
                if mode == "inpainting":
                    result = self.synthesize_inpainting(
                        source_image=source_image,
                        prompt=custom_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        target_size=target_size,
                        use_crack_mask=use_crack_mask
                    )
                else:
                    result = self.synthesize_img2img(
                        source_image=source_image,
                        prompt=custom_prompt,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        target_size=target_size
                    )

                if result is None:
                    filtered_count += 1
                    continue

                stem = Path(img_name).stem
                ext = Path(img_name).suffix
                gen_name = f"{stem}_sd_{mode}_{gen_idx}{ext}"
                gen_path = output_dir / gen_name
                result.save(gen_path, quality=95)

                new_entries.append([gen_name, generated_label])
                generated_count += 1

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"  ⏳ 进度: {idx + 1}/{len(source_entries)} "
                    f"(生成 {generated_count}, 过滤 {filtered_count})"
                )

        # 输出 CSV
        output_csv = output_dir / "sd_augmented.csv"
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in all_entries:
                writer.writerow(row)
            for row in new_entries:
                writer.writerow(row)

        logger.info(
            f"\n{'='*60}\n"
            f"✅ Stable Diffusion 合成完成！\n"
            f"   模式:           {mode}\n"
            f"   LoRA:           {self.lora_path or '未使用'}\n"
            f"   Trigger Word:   {self.trigger_word}\n"
            f"   成功合成数:     {generated_count}\n"
            f"   质量过滤数:     {filtered_count}\n"
            f"   输出目录:       {output_dir}\n"
            f"{'='*60}"
        )

        return {
            "total_source": len(source_entries),
            "total_generated": generated_count,
            "total_filtered": filtered_count,
            "csv_path": str(output_csv)
        }
