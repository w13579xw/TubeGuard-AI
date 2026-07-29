#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the revised YOLOv10-TPH classifier on an ImageFolder dataset.

Expected layout:
    DATA_ROOT/train/{defective,good}/
    DATA_ROOT/val/{defective,good}/
    DATA_ROOT/test/{defective,good}/    # optional
"""

import argparse
import copy
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import YOLOv10TPHClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the detail-preserving YOLOv10-TPH classifier."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=str, default="yolov10n.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolov10_tph"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--no-ffn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def setup_logger(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir / f"yolov10_tph_train_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logger = logging.getLogger("yolov10_tph")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger, log_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_transforms(img_size):
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(data_dir, img_size, batch_size, workers, pin_memory):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"{data_dir} must contain train/ and val/ ImageFolder directories."
        )

    train_transform, eval_transform = build_transforms(img_size)
    datasets = {
        "train": ImageFolder(train_dir, train_transform),
        "val": ImageFolder(val_dir, eval_transform),
    }
    if test_dir.is_dir():
        datasets["test"] = ImageFolder(test_dir, eval_transform)

    for split in datasets:
        if datasets[split].classes != datasets["train"].classes:
            raise ValueError(f"Class folders in {split}/ do not match train/.")

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=(workers > 0),
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def binary_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=0,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all = []
    predictions_all = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp and training)

    for inputs, labels in tqdm(
        loader, desc="train" if training else "eval", leave=False
    ):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    metrics = binary_metrics(labels_all, predictions_all)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, labels_all, predictions_all


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_f1,
    epochs_without_improvement,
    args,
    classes,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
            "classes": classes,
        },
        path,
    )


def evaluate_and_save(
    model, loader, criterion, device, amp, classes, output_dir, split
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, labels, predictions = run_epoch(
        model, loader, criterion, device, optimizer=None, amp=amp
    )
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(classes)))
    )

    with (output_dir / "evaluation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)
    with (output_dir / "classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    with (output_dir / "evaluation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["split", "loss", "accuracy", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerow({"split": split, **metrics})

    with (output_dir / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class", "precision", "recall", "f1-score", "support"],
        )
        writer.writeheader()
        for name, values in report.items():
            if isinstance(values, dict):
                writer.writerow({"class": name, **values})
            else:
                writer.writerow(
                    {
                        "class": name,
                        "precision": values,
                        "recall": "",
                        "f1-score": "",
                        "support": "",
                    }
                )

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *classes])
        for class_name, row in zip(classes, matrix.tolist()):
            writer.writerow([class_name, *row])
    return metrics


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger(args.log_dir)
    logger.info("Log file: %s", log_path)
    script_path = Path(__file__).resolve()
    logger.info("Training script: %s", script_path)
    logger.info(
        "Command: %s",
        " ".join([str(script_path), *sys.argv[1:]]),
    )
    logger.info("Arguments: %s", json.dumps(vars(args), default=str, ensure_ascii=False))

    seed_everything(args.seed)
    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    use_amp = args.amp and device.type == "cuda"
    datasets, loaders = build_dataloaders(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = YOLOv10TPHClassifier(
        model_weight=args.weights,
        num_classes=len(datasets["train"].classes),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_ffn=not args.no_ffn,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0
    best_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch + 1)

    best_state = copy.deepcopy(model.state_dict())
    history_path = args.output_dir / "history.csv"
    append_history = history_path.exists() and start_epoch > 0
    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_f1",
                "val_loss",
                "val_accuracy",
                "val_f1",
                "lr",
            ],
        )
        if not append_history:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            train_metrics, _, _ = run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                optimizer=optimizer,
                amp=use_amp,
            )
            val_metrics, _, _ = run_epoch(
                model, loaders["val"], criterion, device, amp=use_amp
            )
            scheduler.step()

            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            history_file.flush()

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.output_dir / "best_model.pth")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                args.output_dir / "last_checkpoint.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_f1,
                epochs_without_improvement,
                args,
                datasets["train"].classes,
            )
            logger.info(
                f"epoch={epoch + 1:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"best_f1={best_f1:.4f}"
            )

            if epochs_without_improvement >= args.patience:
                logger.info(
                    "Early stopping after %d unimproved epochs.", args.patience
                )
                break

    best_model_path = args.output_dir / "best_model.pth"
    if best_model_path.is_file():
        best_state = torch.load(best_model_path, map_location=device)
        logger.info("Loaded best model from %s", best_model_path)
    else:
        logger.info("Best-model file not found; evaluating the current model state.")
    model.load_state_dict(best_state)
    evaluation_split = "test" if "test" in loaders else "val"
    metrics = evaluate_and_save(
        model,
        loaders[evaluation_split],
        criterion,
        device,
        use_amp,
        datasets[evaluation_split].classes,
        args.output_dir,
        evaluation_split,
    )
    logger.info("Evaluation split: %s", evaluation_split)
    logger.info("Evaluation metrics: %s", json.dumps(metrics))
    logger.info("CSV evaluation results saved in %s", args.output_dir)


if __name__ == "__main__":
    main()
