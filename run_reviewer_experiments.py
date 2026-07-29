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
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from torchvision import transforms

from revision_experiments.data import (
    ManifestDataset,
    prepare_manifests,
    read_manifest,
)
from revision_experiments.engine import (
    bootstrap_confidence_intervals,
    evaluate,
    save_evaluation,
    seed_everything,
    train,
)
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
RESULT_FIELDS = [
    "experiment",
    "model",
    "split",
    "n",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "pr_auc",
    "defect_precision",
    "defect_recall",
    "defect_f1",
    "normal_precision",
    "normal_recall",
    "normal_f1",
    "n_defective",
    "n_normal",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "mcc_ci_low",
    "mcc_ci_high",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "tp",
    "fp",
    "tn",
    "fn",
]


class Tee:
    """Write stdout/stderr to both the terminal and a persistent log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def experiment_logging(log_dir: Path, name: str):
    """Append one complete training/evaluation run to log/<...>/<name>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            started = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"\n{'=' * 80}\nrun={name} started={started}\nlog={log_path}")
            try:
                yield log_path
            except Exception:
                print(f"run={name} status=FAILED")
                traceback.print_exc()
                raise
            else:
                finished = datetime.now().astimezone().isoformat(timespec="seconds")
                print(f"run={name} status=COMPLETED finished={finished}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["prepare", "comparison", "ablation", "evaluate"], required=True
    )
    parser.add_argument("--models", nargs="+", default=COMPARISON_MODELS)
    parser.add_argument("--variants", nargs="+", default=list(ABLATIONS))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/reviewer_revision"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reviewer_revision"))
    parser.add_argument("--log-dir", type=Path, default=Path("log/reviewer_revision"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Default: <output-dir>/real_test_summary.csv",
    )
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


def _run_one(args, name, spec):
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
    # Avoid holding two full networks on the GPU while the best checkpoint is
    # reloaded for the mandatory post-training evaluation.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = evaluate_checkpoint(args, experiment_dir)
    for row in result:
        update_summary_csv(args.summary_csv, row)
    return result


def run_one(args, name, spec):
    with experiment_logging(args.log_dir, name):
        print(
            f"experiment={name} model={spec['model']} "
            f"augmentation={spec['augmentation']} hnm={spec['hnm']}"
        )
        return _run_one(args, name, spec)


def write_result_csv(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})


def update_summary_csv(path: Path, result: dict):
    """Atomically upsert a result; FileLock makes parallel GPU jobs safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=120)
    with lock:
        rows = []
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        key = (str(result["experiment"]), str(result["split"]))
        rows = [
            row
            for row in rows
            if (row.get("experiment"), row.get("split")) != key
        ]
        rows.append({field: result.get(field, "") for field in RESULT_FIELDS})
        rows.sort(key=lambda row: (str(row["experiment"]), str(row["split"])))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    print(f"Updated summary CSV: {path}")


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
    results = []
    manifests = [("real_test", args.manifest_dir / "real_test.csv")]
    balanced_path = args.manifest_dir / "real_test_balanced.csv"
    if balanced_path.exists():
        manifests.append(("real_test_balanced", balanced_path))
    for split, manifest_path in manifests:
        dataset = ManifestDataset(read_manifest(manifest_path), plain_transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        result, scores = evaluate(model, loader, device, amp=args.amp)
        predictions = [int(score >= 0.5) for score in scores]
        result.update(
            bootstrap_confidence_intervals(
                dataset.targets, predictions, scores, seed=args.seed
            )
        )
        result.update(
            model=checkpoint["model_name"],
            experiment=checkpoint["variant"],
            split=split,
        )
        save_evaluation(experiment_dir, split, result)
        write_result_csv(experiment_dir / f"{split}_metrics.csv", result)
        print(json.dumps(result, indent=2))
        results.append(result)
    return results


def evaluate_all(args):
    results = []
    for checkpoint in sorted(args.output_dir.glob("*/best_model.pth")):
        name = checkpoint.parent.name
        with experiment_logging(args.log_dir, f"evaluate_{name}"):
            evaluated = evaluate_checkpoint(args, checkpoint.parent)
            for result in evaluated:
                update_summary_csv(args.summary_csv, result)
            results.extend(evaluated)
    if not results:
        raise FileNotFoundError(f"No checkpoints found below {args.output_dir}")
    print(f"Saved {len(results)} results to {args.summary_csv}")


def run_jobs(args, jobs):
    failures = []
    for name, spec in jobs:
        try:
            run_one(args, name, spec)
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"Continuing after failed experiment {name}: {exc}", file=sys.stderr)
    if failures:
        details = "; ".join(f"{name}: {message}" for name, message in failures)
        raise RuntimeError(f"{len(failures)} experiment(s) failed: {details}")


def main():
    args = parse_args()
    if args.summary_csv is None:
        args.summary_csv = args.output_dir / "real_test_summary.csv"
    if args.mode == "prepare":
        with experiment_logging(args.log_dir, "prepare"):
            prepare(args)
        return
    required = [
        args.manifest_dir / name
        for name in (
            "train_original.csv",
            "val.csv",
            "real_test.csv",
            "real_test_balanced.csv",
        )
    ]
    if not all(path.exists() for path in required):
        prepare(args)
    if args.mode == "comparison":
        jobs = [
            (
                f"comparison_{model_name}",
                dict(model=model_name, augmentation=True, hnm=False),
            )
            for model_name in args.models
        ]
        run_jobs(args, jobs)
    elif args.mode == "ablation":
        jobs = []
        for variant in args.variants:
            if variant not in ABLATIONS:
                raise ValueError(f"Unknown variant: {variant}")
            jobs.append((f"ablation_{variant}", ABLATIONS[variant]))
        run_jobs(args, jobs)
    else:
        evaluate_all(args)


if __name__ == "__main__":
    main()
