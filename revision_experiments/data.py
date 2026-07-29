"""Dataset preparation and loading for the reviewer-requested experiments."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    image: str
    label: int
    source: str
    synthetic: bool = False


def parse_label(value: str) -> int:
    """Return 1 for defective and 0 for good, accepting Chinese/English labels."""
    text = value.strip().lower().replace("[", "").replace("]", "")
    good_tokens = ("无缺陷", "good", "normal", "negative", "ok")
    defect_tokens = ("有缺陷", "defective", "defect", "positive", "ng")
    if any(token in text for token in good_tokens):
        return 0
    if any(token in text for token in defect_tokens):
        return 1
    if text in {"0", "1"}:
        return int(text)
    raise ValueError(f"Unrecognised class label: {value!r}")


def _read_csv(csv_path: Path, image_dir: Path, synthetic: bool) -> list[Record]:
    records = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("image") or row.get("path") or "").strip()
            if not name:
                continue
            path = Path(name)
            if not path.is_absolute():
                path = image_dir / path
            source = re.split(r"_aug(?:mented)?_", Path(name).stem, maxsplit=1)[0]
            records.append(
                Record(
                    # Keep relative paths relative so the generated manifests can
                    # be copied from Windows to a Linux training server.
                    image=path.as_posix(),
                    label=parse_label(row["label"]),
                    source=source,
                    synthetic=synthetic,
                )
            )
    return records


def _write_manifest(path: Path, records: Iterable[Record]) -> None:
    rows = list(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image", "label", "source", "synthetic"]
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def prepare_manifests(
    output_dir: Path,
    train_csv: Path,
    real_test_csv: Path,
    image_dir: Path,
    augmentation_csv: Path | None,
    augmentation_dir: Path | None,
    val_fraction: float,
    seed: int,
) -> dict[str, int]:
    """Build train/val/real-test manifests without crossing source groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = _read_csv(train_csv, image_dir, synthetic=False)
    real_test = _read_csv(real_test_csv, image_dir, synthetic=False)
    test_by_label = {
        label: [row for row in real_test if row.label == label]
        for label in (0, 1)
    }
    balanced_count = min(len(rows) for rows in test_by_label.values())
    rng = random.Random(seed)
    real_test_balanced = sorted(
        [
            row
            for label in (0, 1)
            for row in rng.sample(test_by_label[label], balanced_count)
        ],
        key=lambda row: row.image,
    )

    paths = [row.image for row in originals]
    labels = [row.label for row in originals]
    train_paths, val_paths = train_test_split(
        paths,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_path_set = set(train_paths)
    train_original = [row for row in originals if row.image in train_path_set]
    val = [row for row in originals if row.image not in train_path_set]
    train_sources = {row.source for row in train_original}

    augmented: list[Record] = []
    if augmentation_csv and augmentation_csv.exists():
        if augmentation_dir is None:
            raise ValueError("augmentation_dir is required with augmentation_csv")
        candidates = _read_csv(augmentation_csv, augmentation_dir, synthetic=True)
        # augmented.csv also contains copies of original rows; only generated rows
        # are admitted, and only when their source image belongs to the train fold.
        augmented = [
            row
            for row in candidates
            if "_aug" in Path(row.image).stem.lower() and row.source in train_sources
        ]

    real_paths = {row.image for row in real_test}
    overlap = real_paths & {row.image for row in train_original + val + augmented}
    if overlap:
        raise RuntimeError(f"Real-test leakage detected for {len(overlap)} paths")

    manifests = {
        "train_original.csv": train_original,
        "train_augmented.csv": augmented,
        "val.csv": val,
        "real_test.csv": real_test,
        "real_test_balanced.csv": real_test_balanced,
    }
    for name, rows in manifests.items():
        missing = [row.image for row in rows if not Path(row.image).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{name}: {len(missing)} images are missing; first: {missing[0]}"
            )
        _write_manifest(output_dir / name, rows)

    summary = {name.removesuffix(".csv"): len(rows) for name, rows in manifests.items()}
    summary["seed"] = seed
    summary["val_fraction"] = val_fraction
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def read_manifest(path: Path) -> list[Record]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Record(
                    image=row["image"],
                    label=int(row["label"]),
                    source=row.get("source", Path(row["image"]).stem),
                    synthetic=row.get("synthetic", "False").lower() == "true",
                )
            )
    return rows


class ManifestDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform):
        if not records:
            raise ValueError("Dataset manifest is empty")
        self.records = list(records)
        self.transform = transform
        self.targets = [row.label for row in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        with Image.open(row.image) as image:
            image = image.convert("RGB")
        return self.transform(image), row.label, index
