"""Shared training, hard-negative mining and evaluation utilities."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(labels, predictions, scores) -> dict[str, float | int]:
    defect_precision, defect_recall, defect_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    normal_precision, normal_recall, normal_f1, _ = (
        precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=0, zero_division=0
        )
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(defect_precision),
        "recall": float(defect_recall),
        "f1": float(defect_f1),
        "defect_precision": float(defect_precision),
        "defect_recall": float(defect_recall),
        "defect_f1": float(defect_f1),
        "normal_precision": float(normal_precision),
        "normal_recall": float(normal_recall),
        "normal_f1": float(normal_f1),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": len(labels),
        "n_defective": int(sum(label == 1 for label in labels)),
        "n_normal": int(sum(label == 0 for label in labels)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    )
    result["pr_auc"] = (
        float(average_precision_score(labels, scores))
        if len(set(labels)) == 2
        else float("nan")
    )
    return result


def bootstrap_confidence_intervals(
    labels,
    predictions,
    scores,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Stratified bootstrap CIs for prevalence-robust headline metrics."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    tracked = ("balanced_accuracy", "macro_f1", "mcc", "roc_auc", "pr_auc")
    values = {name: [] for name in tracked}
    for _ in range(samples):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        sampled = metrics(
            labels[indices], predictions[indices], scores[indices]
        )
        for name in tracked:
            values[name].append(sampled[name])
    alpha = (1.0 - confidence) / 2.0
    result = {}
    for name, observations in values.items():
        result[f"{name}_ci_low"] = float(np.quantile(observations, alpha))
        result[f"{name}_ci_high"] = float(np.quantile(observations, 1.0 - alpha))
    return result


def evaluate(model, loader, device, criterion=None, amp=False):
    model.eval()
    labels_all, predictions_all, scores_all = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels) if criterion else None
            probabilities = logits.softmax(dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            if loss is not None:
                total_loss += loss.item() * images.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            scores_all.extend(probabilities.cpu().tolist())
    result = metrics(labels_all, predictions_all, scores_all)
    if criterion:
        result["loss"] = total_loss / len(loader.dataset)
    return result, scores_all


def hard_negative_weights(dataset, scores, fraction: float, boost: float):
    """Upweight real good samples with the largest false-defect probabilities."""
    candidates = [
        (score, index)
        for index, (row, score) in enumerate(zip(dataset.records, scores))
        if row.label == 0 and not row.synthetic
    ]
    count = max(1, round(len(candidates) * fraction))
    selected = {index for _, index in sorted(candidates, reverse=True)[:count]}
    weights = [boost if index in selected else 1.0 for index in range(len(dataset))]
    return weights, selected


def train(
    model,
    train_dataset,
    val_loader,
    output_dir: Path,
    model_name: str,
    variant: str,
    device,
    epochs: int,
    batch_size: int,
    workers: int,
    lr: float,
    weight_decay: float,
    patience: int,
    amp: bool,
    hard_negative_mining: bool,
    hnm_warmup: int,
    hnm_fraction: float,
    hnm_boost: float,
    config: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp and device.type == "cuda"
    )
    best_f1, stale, sample_weights = -1.0, 0, None
    history = []

    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        sampler = None
        shuffle = True
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=generator,
        )
        model.train()
        total_loss = 0.0
        for images, labels, _ in tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, criterion, amp)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_dataset),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch + 1:03d} train_loss={row['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1, stale = val_metrics["f1"], 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "variant": variant,
                    "config": config,
                },
                output_dir / "best_model.pth",
            )
        else:
            stale += 1

        if hard_negative_mining and epoch + 1 >= hnm_warmup:
            mining_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )
            _, train_scores = evaluate(model, mining_loader, device, amp=amp)
            sample_weights, selected = hard_negative_weights(
                train_dataset, train_scores, hnm_fraction, hnm_boost
            )
            with (output_dir / f"hard_negatives_epoch_{epoch + 1:03d}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "defect_probability"])
                for index in sorted(selected):
                    writer.writerow(
                        [train_dataset.records[index].image, train_scores[index]]
                    )

        if stale >= patience:
            break

    if history:
        with (output_dir / "history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_evaluation(output_dir: Path, split: str, result: dict) -> None:
    with (output_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
