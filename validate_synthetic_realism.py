#!/usr/bin/env python
"""Quantify realism and distribution consistency of synthetic tube defects.

The real reference set must be real-only and independent from augmentation
generation. Results are supportive diagnostics; they do not replace evaluation
of the trained detector on the untouched real test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg, ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from revision_experiments.data import read_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit synthetic-defect realism against independent real defects."
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path("data/reviewer_revision/real_test.csv"),
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=Path("data/reviewer_revision/train_augmented.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/reviewer_revision/synthetic_realism"),
    )
    parser.add_argument("--max-samples-per-domain", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    return parser.parse_args()


class FeatureDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), str(self.paths[index])


def stratified_paths(manifest: Path, synthetic: bool, limit: int, seed: int):
    records = [
        row
        for row in read_manifest(manifest)
        if row.label == 1 and row.synthetic is synthetic
    ]
    if not records:
        kind = "synthetic" if synthetic else "real"
        raise ValueError(f"No {kind} defective samples found in {manifest}")
    rng = random.Random(seed)
    selected = rng.sample(records, min(limit, len(records)))
    return [Path(row.image) for row in selected]


def extract_features(paths, model, transform, batch_size, workers, device):
    loader = DataLoader(
        FeatureDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="ResNet50 features", leave=False):
            images = images.to(device, non_blocking=True)
            features.append(model(images).flatten(1).cpu().numpy())
    return np.concatenate(features)


def polynomial_mmd(real, synthetic):
    dimension = real.shape[1]
    k_xx = (real @ real.T / dimension + 1.0) ** 3
    k_yy = (synthetic @ synthetic.T / dimension + 1.0) ** 3
    k_xy = (real @ synthetic.T / dimension + 1.0) ** 3
    n, m = len(real), len(synthetic)
    return float(
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )


def kid(real, synthetic, subsets, subset_size, seed):
    rng = np.random.default_rng(seed)
    size = min(subset_size, len(real), len(synthetic))
    values = []
    for _ in range(subsets):
        x = real[rng.choice(len(real), size=size, replace=False)]
        y = synthetic[rng.choice(len(synthetic), size=size, replace=False)]
        values.append(polynomial_mmd(x, y))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def rbf_mmd(real, synthetic):
    combined = np.concatenate([real, synthetic])
    distances = pairwise_distances(combined, metric="euclidean")
    nonzero = distances[distances > 0]
    bandwidth = float(np.median(nonzero))
    gamma = 1.0 / (2.0 * bandwidth**2)
    k_xx = np.exp(-gamma * pairwise_distances(real, squared=True))
    k_yy = np.exp(-gamma * pairwise_distances(synthetic, squared=True))
    k_xy = np.exp(-gamma * pairwise_distances(real, synthetic, squared=True))
    n, m = len(real), len(synthetic)
    value = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )
    return float(value), bandwidth


def frechet_feature_distance(real, synthetic, seed):
    combined = np.concatenate([real, synthetic])
    components = min(128, len(combined) - 2, combined.shape[1])
    pca = PCA(n_components=components, whiten=False, random_state=seed)
    projected = pca.fit_transform(combined)
    real_p, synthetic_p = projected[: len(real)], projected[len(real) :]
    mean_delta = real_p.mean(0) - synthetic_p.mean(0)
    cov_real = np.cov(real_p, rowvar=False)
    cov_synthetic = np.cov(synthetic_p, rowvar=False)
    covariance_mean = linalg.sqrtm(cov_real @ cov_synthetic)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    distance = (
        mean_delta @ mean_delta
        + np.trace(cov_real + cov_synthetic - 2.0 * covariance_mean)
    )
    return float(max(distance, 0.0)), components, float(
        pca.explained_variance_ratio_.sum()
    )


def distribution_prdc(real, synthetic, nearest_k):
    if min(len(real), len(synthetic)) <= nearest_k:
        raise ValueError("The number of samples must exceed --knn-k")
    real_real = pairwise_distances(real)
    synthetic_synthetic = pairwise_distances(synthetic)
    real_synthetic = pairwise_distances(real, synthetic)
    real_radius = np.partition(real_real, nearest_k, axis=1)[:, nearest_k]
    synthetic_radius = np.partition(
        synthetic_synthetic, nearest_k, axis=1
    )[:, nearest_k]
    precision = (real_synthetic <= real_radius[:, None]).any(axis=0).mean()
    recall = (real_synthetic <= synthetic_radius[None, :]).any(axis=1).mean()
    density = (real_synthetic <= real_radius[:, None]).sum(axis=0).mean() / nearest_k
    coverage = (real_synthetic.min(axis=1) <= real_radius).mean()
    return {
        "feature_precision": float(precision),
        "feature_recall": float(recall),
        "feature_density": float(density),
        "feature_coverage": float(coverage),
    }


def classifier_two_sample_test(real, synthetic, seed):
    features = np.concatenate([real, synthetic])
    labels = np.concatenate(
        [np.zeros(len(real), dtype=int), np.ones(len(synthetic), dtype=int)]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    components = min(64, len(features) - 2, features.shape[1])
    classifier = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=seed),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    probabilities = cross_val_predict(
        classifier, features, labels, cv=folds, method="predict_proba"
    )[:, 1]
    fold_auc = []
    for _, test_indices in folds.split(features, labels):
        fold_auc.append(roc_auc_score(labels[test_indices], probabilities[test_indices]))
    auc = roc_auc_score(labels, probabilities)
    return float(max(auc, 1.0 - auc)), float(np.std(fold_auc, ddof=1))


def physical_descriptors(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    gradient = np.hypot(grad_x, grad_y)
    return {
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "laplacian_variance": float(ndimage.laplace(gray).var()),
        "gradient_mean": float(gradient.mean()),
        "edge_density": float((gradient > 0.20).mean()),
        "specular_ratio": float((rgb.max(axis=2) > 0.95).mean()),
    }


def compare_physical_descriptors(real_paths, synthetic_paths):
    real_rows = [physical_descriptors(path) for path in tqdm(real_paths, desc="Real optics")]
    synthetic_rows = [
        physical_descriptors(path) for path in tqdm(synthetic_paths, desc="Synthetic optics")
    ]
    output = []
    for name in real_rows[0]:
        real = np.array([row[name] for row in real_rows])
        synthetic = np.array([row[name] for row in synthetic_rows])
        pooled_std = np.sqrt((real.var(ddof=1) + synthetic.var(ddof=1)) / 2.0)
        output.append(
            {
                "descriptor": name,
                "real_mean": float(real.mean()),
                "real_std": float(real.std(ddof=1)),
                "synthetic_mean": float(synthetic.mean()),
                "synthetic_std": float(synthetic.std(ddof=1)),
                "standardized_mean_difference": float(
                    (synthetic.mean() - real.mean()) / max(pooled_std, 1e-12)
                ),
                "normalized_wasserstein": float(
                    stats.wasserstein_distance(real, synthetic)
                    / max(pooled_std, 1e-12)
                ),
                "ks_statistic": float(stats.ks_2samp(real, synthetic).statistic),
                "ks_pvalue": float(stats.ks_2samp(real, synthetic).pvalue),
            }
        )
    return output


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    limit = args.max_samples_per_domain
    real_paths = stratified_paths(args.real_manifest, False, limit, args.seed)
    synthetic_paths = stratified_paths(
        args.synthetic_manifest, True, limit, args.seed
    )
    paired_count = min(len(real_paths), len(synthetic_paths))
    real_paths, synthetic_paths = real_paths[:paired_count], synthetic_paths[:paired_count]

    weights = ResNet50_Weights.IMAGENET1K_V2
    feature_model = resnet50(weights=weights)
    feature_model.fc = torch.nn.Identity()
    feature_model = feature_model.to(device).eval()
    real_features = extract_features(
        real_paths,
        feature_model,
        weights.transforms(),
        args.batch_size,
        args.workers,
        device,
    )
    synthetic_features = extract_features(
        synthetic_paths,
        feature_model,
        weights.transforms(),
        args.batch_size,
        args.workers,
        device,
    )
    normalized_real = real_features / np.maximum(
        np.linalg.norm(real_features, axis=1, keepdims=True), 1e-12
    )
    normalized_synthetic = synthetic_features / np.maximum(
        np.linalg.norm(synthetic_features, axis=1, keepdims=True), 1e-12
    )

    frd, pca_components, explained_variance = frechet_feature_distance(
        real_features, synthetic_features, args.seed
    )
    kid_mean, kid_std = kid(
        real_features,
        synthetic_features,
        args.kid_subsets,
        args.kid_subset_size,
        args.seed,
    )
    mmd, bandwidth = rbf_mmd(normalized_real, normalized_synthetic)
    c2st_auc, c2st_std = classifier_two_sample_test(
        real_features, synthetic_features, args.seed
    )
    summary = {
        "real_reference": str(args.real_manifest),
        "synthetic_source": str(args.synthetic_manifest),
        "real_defective_n": len(real_paths),
        "synthetic_defective_n": len(synthetic_paths),
        "feature_extractor": "ResNet50 ImageNet-1K V2",
        "frechet_resnet_distance_pca": frd,
        "frechet_pca_components": pca_components,
        "frechet_pca_explained_variance": explained_variance,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "rbf_mmd": mmd,
        "rbf_median_bandwidth": bandwidth,
        "c2st_roc_auc": c2st_auc,
        "c2st_fold_std": c2st_std,
        **distribution_prdc(normalized_real, normalized_synthetic, args.knn_k),
        "seed": args.seed,
    }
    physical_rows = compare_physical_descriptors(real_paths, synthetic_paths)
    write_rows(
        args.output_dir / "feature_distribution_metrics.csv",
        [{"metric": key, "value": value} for key, value in summary.items()],
    )
    write_rows(args.output_dir / "physical_descriptor_comparison.csv", physical_rows)
    write_rows(
        args.output_dir / "audited_samples.csv",
        [
            {"domain": domain, "image": path.as_posix()}
            for domain, paths in (("real", real_paths), ("synthetic", synthetic_paths))
            for path in paths
        ],
    )
    with (args.output_dir / "synthetic_realism_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
