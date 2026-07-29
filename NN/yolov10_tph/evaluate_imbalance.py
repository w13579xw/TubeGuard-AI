#!/usr/bin/env python
"""Post-hoc imbalance-aware evaluation for a completed YOLOv10-TPH run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from model import YOLOv10TPHClassifier  # noqa: E402
from revision_experiments.engine import (  # noqa: E402
    bootstrap_confidence_intervals,
    metrics,
)
from train import build_transforms  # noqa: E402


RESULT_FIELDS = [
    "split",
    "n",
    "n_defective",
    "n_normal",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
    "defect_precision",
    "defect_recall",
    "defect_f1",
    "normal_precision",
    "normal_recall",
    "normal_f1",
    "roc_auc",
    "pr_auc",
    "tp",
    "fp",
    "tn",
    "fn",
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
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", default="yolov10n.pt")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/yolov10_tph/evaluation")
    )
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--no-ffn", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def defective_class_index(classes):
    candidates = [
        index
        for index, name in enumerate(classes)
        if any(token in name.lower() for token in ("defect", "ng", "有缺陷"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify exactly one defective class from {classes}"
        )
    return candidates[0]


def evaluate_indices(labels, predictions, scores, indices, split, args):
    selected_labels = np.asarray(labels)[indices]
    selected_predictions = np.asarray(predictions)[indices]
    selected_scores = np.asarray(scores)[indices]
    result = metrics(selected_labels, selected_predictions, selected_scores)
    result.update(
        bootstrap_confidence_intervals(
            selected_labels,
            selected_predictions,
            selected_scores,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    result["split"] = split
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    split_dir = args.data_dir / "test"
    if not split_dir.is_dir():
        split_dir = args.data_dir / "val"
    _, transform = build_transforms(args.img_size)
    dataset = ImageFolder(split_dir, transform)
    defect_index = defective_class_index(dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = YOLOv10TPHClassifier(
        model_weight=args.weights,
        num_classes=len(dataset.classes),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_ffn=not args.no_ffn,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device).eval()

    labels, predictions, scores, paths = [], [], [], []
    cursor = 0
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc="real-only evaluation"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                probabilities = model(images).softmax(dim=1)
            defect_scores = probabilities[:, defect_index]
            predicted_classes = probabilities.argmax(dim=1)
            labels.extend((batch_labels == defect_index).int().tolist())
            predictions.extend(
                (predicted_classes.cpu() == defect_index).int().tolist()
            )
            scores.extend(defect_scores.cpu().tolist())
            paths.extend(
                dataset.samples[index][0]
                for index in range(cursor, cursor + len(batch_labels))
            )
            cursor += len(batch_labels)

    all_indices = np.arange(len(labels))
    by_class = [np.flatnonzero(np.asarray(labels) == label) for label in (0, 1)]
    balanced_count = min(len(group) for group in by_class)
    rng = np.random.default_rng(args.seed)
    balanced_indices = np.sort(
        np.concatenate(
            [
                rng.choice(group, size=balanced_count, replace=False)
                for group in by_class
            ]
        )
    )
    results = [
        evaluate_indices(
            labels, predictions, scores, all_indices, "real_test", args
        ),
        evaluate_indices(
            labels,
            predictions,
            scores,
            balanced_indices,
            "real_test_balanced",
            args,
        ),
    ]

    with (args.output_dir / "imbalance_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in RESULT_FIELDS}
            for row in results
        )
    with (args.output_dir / "imbalance_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)
    with (args.output_dir / "real_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "defective_label", "defective_prediction", "defect_probability"]
        )
        writer.writerows(zip(paths, labels, predictions, scores))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
