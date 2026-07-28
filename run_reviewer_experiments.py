#!/usr/bin/env python
"""Run cross-architecture comparisons and component-level ablations.

All final numbers are reported on the untouched real test manifest. Validation
is used for checkpoint selection; the real test set is never used for training,
hard-negative mining, early stopping, or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from revision_experiments.data import (
    ManifestDataset,
    prepare_manifests,
    read_manifest,
)
from revision_experiments.engine import evaluate, save_evaluation, seed_everything, train
from revision_experiments.models import build_model


ABLATIONS = {
    "yolov10_only": dict(model="yolov10n", augmentation=False, hnm=False),
    "yolov10_tph": dict(model="yolov10_tph", augmentation=False, hnm=False),
    "yolov10_augmentation": dict(model="yolov10n", augmentation=True, hnm=False),
    "yolov10_hnm": dict(model="yolov10n", augmentation=False, hnm=True),
    "yolov10_tph_augmentation": dict(
        model="yolov10_tph", augmentation=True, hnm=False
    ),
    "complete": dict(model="yolov10_tph", augmentation=True, hnm=True),
}
COMPARISON_MODELS = [
    "yolov8n",
    "yolov10n",
    "yolo11n",
    "detr_r50",
    "deformable_detr_r50",
    "yolov10_tph",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["prepare", "comparison", "ablation", "evaluate"], required=True
    )
    parser.add_argument("--models", nargs="+", default=COMPARISON_MODELS)
    parser.add_argument("--variants", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/reviewer_revision"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reviewer_revision"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--real-test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--augmentation-csv", type=Path, default=Path("data/defect_test/augmented.csv")
    )
    parser.add_argument(
        "--augmentation-dir", type=Path, default=Path("data/defect_test")
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--hnm-warmup", type=int, default=5)
    parser.add_argument("--hnm-fraction", type=float, default=0.25)
    parser.add_argument("--hnm-boost", type=float, default=3.0)
    return parser.parse_args()


def prepare(args):
    summary = prepare_manifests(
        args.manifest_dir,
        args.train_csv,
        args.real_test_csv,
        args.image_dir,
        args.augmentation_csv,
        args.augmentation_dir,
        args.val_fraction,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


def make_transforms(img_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    plain = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
    )
    augmented = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return plain, augmented


def run_one(args, name, spec):
    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    plain_transform, augmentation_transform = make_transforms(args.img_size)
    originals = read_manifest(args.manifest_dir / "train_original.csv")
    records = list(originals)
    if spec["augmentation"]:
        records += read_manifest(args.manifest_dir / "train_augmented.csv")
    train_dataset = ManifestDataset(
        records, augmentation_transform if spec["augmentation"] else plain_transform
    )
    val_dataset = ManifestDataset(
        read_manifest(args.manifest_dir / "val.csv"), plain_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = build_model(spec["model"]).to(device)
    experiment_dir = args.output_dir / name
    config = {
        **vars(args),
        **spec,
        "name": name,
        "seed": args.seed,
        "manifest_dir": str(args.manifest_dir.resolve()),
        "output_dir": str(experiment_dir.resolve()),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    train(
        model,
        train_dataset,
        val_loader,
        experiment_dir,
        spec["model"],
        name,
        device,
        args.epochs,
        args.batch_size,
        args.workers,
        args.lr,
        args.weight_decay,
        args.patience,
        args.amp,
        spec["hnm"],
        args.hnm_warmup,
        args.hnm_fraction,
        args.hnm_boost,
        config,
    )
    evaluate_checkpoint(args, experiment_dir)


def evaluate_checkpoint(args, experiment_dir: Path):
    checkpoint_path = experiment_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    img_size = int(saved_config.get("img_size", args.img_size))
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    plain_transform, _ = make_transforms(img_size)
    real_dataset = ManifestDataset(
        read_manifest(args.manifest_dir / "real_test.csv"), plain_transform
    )
    loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    result, _ = evaluate(model, loader, device, amp=args.amp)
    result.update(
        model=checkpoint["model_name"],
        experiment=checkpoint["variant"],
        split="real_test",
    )
    save_evaluation(experiment_dir, "real_test", result)
    print(json.dumps(result, indent=2))
    return result


def evaluate_all(args):
    results = []
    for checkpoint in sorted(args.output_dir.glob("*/best_model.pth")):
        results.append(evaluate_checkpoint(args, checkpoint.parent))
    if not results:
        raise FileNotFoundError(f"No checkpoints found below {args.output_dir}")
    summary = args.output_dir / "real_test_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {summary}")


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
        return
    required = [args.manifest_dir / name for name in ("train_original.csv", "val.csv", "real_test.csv")]
    if not all(path.exists() for path in required):
        prepare(args)
    if args.mode == "comparison":
        for model_name in args.models:
            run_one(
                args,
                f"comparison_{model_name}",
                dict(model=model_name, augmentation=True, hnm=False),
            )
    elif args.mode == "ablation":
        for variant in args.variants:
            if variant not in ABLATIONS:
                raise ValueError(f"Unknown variant: {variant}")
            run_one(args, f"ablation_{variant}", ABLATIONS[variant])
    else:
        evaluate_all(args)


if __name__ == "__main__":
    main()
