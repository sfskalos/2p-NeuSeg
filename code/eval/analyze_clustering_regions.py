"""Reviewer-requested clustering stability and regional-weight sensitivity.

The analysis deliberately ignores the pre-existing ``cluster`` column because the
user-supplied feature_boxplots CSV contains a degenerate one-member cluster.  The
12 manuscript-specified features are standardized and clustered from scratch.
Bootstrap resampling is performed at image level to preserve within-image
dependence between cells.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Some Windows Anaconda builds expose an MKL DLL whose version query returns
# null, causing threadpoolctl to fail before KMeans starts.  This only disables
# sklearn's optional oversubscription inspection; it does not alter clustering.
import sklearn.cluster._kmeans as _sk_kmeans
from contextlib import nullcontext

_sk_kmeans.threadpool_info = lambda: []
_sk_kmeans.threadpool_limits = lambda *args, **kwargs: nullcontext()


FEATURES = [
    "area",
    "perimeter",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "circularity",
    "curvature_std",
    "intensity_mean",
    "intensity_std",
    "intensity_skew",
    "intensity_kurtosis",
    "optical_contrast",
]
SEEDS = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2026, 31415, 27182, 12345, 54321, 20261022]
WEIGHT_SCHEMES = {
    "0-1-2": np.array([0.0, 1.0, 2.0]),
    "1-2-4": np.array([1.0, 2.0, 4.0]),
    "1-3-5": np.array([1.0, 3.0, 5.0]),
}


def q025(x: pd.Series) -> float:
    return float(x.quantile(0.025))


def q975(x: pd.Series) -> float:
    return float(x.quantile(0.975))


def summarize(df: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group, sort=True):
        row = {group: key, "n": len(part)}
        for metric in metrics:
            s = part[metric].astype(float)
            row.update(
                {
                    f"{metric}_mean": s.mean(),
                    f"{metric}_sd": s.std(ddof=1),
                    f"{metric}_median": s.median(),
                    f"{metric}_q025": q025(s),
                    f"{metric}_q975": q975(s),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_seed_stability(xz: np.ndarray, out_dir: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict] = []
    labels_by_k: dict[int, list[np.ndarray]] = {}
    rng = np.random.default_rng(20261022)
    sil_idx = np.sort(rng.choice(len(xz), size=min(3000, len(xz)), replace=False))
    for k in range(2, 6):
        labels_by_k[k] = []
        for seed in SEEDS:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
            labels = km.fit_predict(xz)
            labels_by_k[k].append(labels)
            rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "wcss": float(km.inertia_),
                    "silhouette": float(silhouette_score(xz[sil_idx], labels[sil_idx])),
                }
            )
    seed_runs = pd.DataFrame(rows)
    seed_runs.to_csv(out_dir / "clustering_seed_runs.csv", index=False)

    pair_rows = []
    for k, label_runs in labels_by_k.items():
        for i, j in combinations(range(len(SEEDS)), 2):
            pair_rows.append(
                {
                    "k": k,
                    "seed_a": SEEDS[i],
                    "seed_b": SEEDS[j],
                    "ari": adjusted_rand_score(label_runs[i], label_runs[j]),
                    "nmi": normalized_mutual_info_score(label_runs[i], label_runs[j]),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "clustering_seed_pairwise.csv", index=False)
    summary = summarize(seed_runs, "k", ["wcss", "silhouette"]).merge(
        summarize(pair_df, "k", ["ari", "nmi"]), on=["k"], suffixes=("_runs", "_pairs")
    )
    summary.to_csv(out_dir / "clustering_seed_summary.csv", index=False)
    return summary, {k: v[SEEDS.index(20261022)] for k, v in labels_by_k.items()}


def run_group_bootstrap(
    frame: pd.DataFrame,
    full_x: np.ndarray,
    reference_labels: dict[int, np.ndarray],
    out_dir: Path,
    n_boot: int,
) -> pd.DataFrame:
    images = frame["image"].astype(str).to_numpy()
    unique_images = np.unique(images)
    image_rows = {name: np.flatnonzero(images == name) for name in unique_images}
    rng = np.random.default_rng(20261022)
    rows = []
    for k in range(2, 6):
        ref = reference_labels[k]
        for boot in range(n_boot):
            sampled = rng.choice(unique_images, size=len(unique_images), replace=True)
            idx = np.concatenate([image_rows[name] for name in sampled])
            scaler = StandardScaler().fit(full_x[idx])
            xb = scaler.transform(full_x[idx])
            km = KMeans(n_clusters=k, random_state=SEEDS[boot % len(SEEDS)], n_init=10, max_iter=500)
            km.fit(xb)
            labels_full = km.predict(scaler.transform(full_x))
            rows.append(
                {
                    "k": k,
                    "bootstrap": boot,
                    "n_sampled_images": len(sampled),
                    "n_sampled_cells": len(idx),
                    "ari_vs_full": adjusted_rand_score(ref, labels_full),
                    "nmi_vs_full": normalized_mutual_info_score(ref, labels_full),
                    "wcss_bootstrap": float(km.inertia_),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "clustering_group_bootstrap_runs.csv", index=False)
    summarize(result, "k", ["ari_vs_full", "nmi_vs_full", "wcss_bootstrap"]).to_csv(
        out_dir / "clustering_group_bootstrap_summary.csv", index=False
    )
    return result


def add_centroids(frame: pd.DataFrame, mat_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    audit = []
    for image, part in frame.groupby("image", sort=False):
        stem = Path(str(image)).stem
        path = mat_dir / f"{stem}.mat"
        if not path.exists():
            audit.append({"image": image, "status": "missing_mat", "n_rows": len(part)})
            continue
        data = loadmat(path)
        inst_map = np.asarray(data["inst_map"])
        n_ok = 0
        for idx, row in part.iterrows():
            cid = int(row["cell_id"])
            yy, xx = np.where(inst_map == cid)
            if len(xx) == 0:
                continue
            records.append({"row_index": idx, "x": float(xx.mean()), "y": float(yy.mean()), "image_height": inst_map.shape[0], "image_width": inst_map.shape[1]})
            n_ok += 1
        audit.append({"image": image, "status": "ok", "n_rows": len(part), "n_centroids": n_ok})
    coords = pd.DataFrame(records).set_index("row_index")
    joined = frame.join(coords, how="inner")
    return joined, pd.DataFrame(audit)


def severity_order(frame: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Order clusters using manuscript-consistent injury cues.

    Greater injury is defined by lower circularity and greater boundary
    irregularity/optical contrast.  This only assigns biological names to the
    label-invariant K-means solution; it does not affect ARI/NMI.
    """
    cues = frame[["circularity", "curvature_std", "optical_contrast"]].astype(float)
    z = StandardScaler().fit_transform(cues)
    score = -z[:, 0] + z[:, 1] + z[:, 2]
    tmp = pd.DataFrame({"cluster_raw": labels, "severity_score": score})
    means = tmp.groupby("cluster_raw")["severity_score"].mean().sort_values()
    mapping = {int(raw): rank for rank, raw in enumerate(means.index)}
    ordered = np.array([mapping[int(v)] for v in labels], dtype=np.int32)
    detail = means.rename("mean_severity_score").reset_index()
    detail["ordered_subtype"] = detail["cluster_raw"].map(mapping)
    detail["name"] = detail["ordered_subtype"].map({0: "mild", 1: "moderate", 2: "severe"})
    return ordered, detail


def make_windows(frame: pd.DataFrame, grid_size: int, min_cells: int) -> pd.DataFrame:
    f = frame.copy()
    f["grid_x"] = np.minimum((f["x"] / f["image_width"] * grid_size).astype(int), grid_size - 1)
    f["grid_y"] = np.minimum((f["y"] / f["image_height"] * grid_size).astype(int), grid_size - 1)
    rows = []
    for (image, gy, gx), part in f.groupby(["image", "grid_y", "grid_x"], sort=True):
        n = len(part)
        if n < min_cells:
            continue
        counts = np.bincount(part["subtype_ordered"].to_numpy(dtype=int), minlength=3)
        p = counts / n
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(3.0))
        rows.append(
            {
                "image": image,
                "grid_y": gy,
                "grid_x": gx,
                "n_cells": n,
                "p_mild": p[0],
                "p_moderate": p[1],
                "p_severe": p[2],
                "density": n / ((512.0 / grid_size) ** 2),
                "entropy": entropy,
                "x_norm": (gx + 0.5) / grid_size,
                "y_norm": (gy + 0.5) / grid_size,
            }
        )
    return pd.DataFrame(rows)


def best_mapped_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    cm = confusion_matrix(a, b)
    # Only three regional labels, so enumerating permutations is exact.
    import itertools

    return max(sum(cm[i, p[i]] for i in range(len(p))) for p in itertools.permutations(range(cm.shape[0]))) / len(a)


def regional_sensitivity(windows: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    props = windows[["p_mild", "p_moderate", "p_severe"]].to_numpy()
    labels = {}
    run_rows = []
    for name, weights in WEIGHT_SCHEMES.items():
        injury = props @ weights
        raw = np.column_stack(
            [props, injury, windows["density"], windows["entropy"], windows["x_norm"], windows["y_norm"]]
        )
        xz = StandardScaler().fit_transform(raw)
        scheme_runs = []
        for seed in SEEDS:
            km = KMeans(n_clusters=3, random_state=seed, n_init=20, max_iter=500)
            lab = km.fit_predict(xz)
            severity = pd.Series(injury).groupby(lab).mean().sort_values()
            mapping = {int(cluster): rank for rank, cluster in enumerate(severity.index)}
            ordered = np.array([mapping[int(v)] for v in lab], dtype=np.int32)
            scheme_runs.append(ordered)
            run_rows.append(
                {
                    "weight_scheme": name,
                    "seed": seed,
                    "wcss": float(km.inertia_),
                    "silhouette": float(silhouette_score(xz, lab)),
                }
            )
        labels[name] = scheme_runs[SEEDS.index(20261022)]
    pair_rows = []
    for a, b in combinations(WEIGHT_SCHEMES, 2):
        pair_rows.append(
            {
                "scheme_a": a,
                "scheme_b": b,
                "ari": adjusted_rand_score(labels[a], labels[b]),
                "nmi": normalized_mutual_info_score(labels[a], labels[b]),
                "label_mapped_agreement": best_mapped_accuracy(labels[a], labels[b]),
            }
        )
    assignment = windows[["image", "grid_y", "grid_x", "n_cells"]].copy()
    for name, lab in labels.items():
        assignment[f"region_{name}"] = lab
    assignment.to_csv(out_dir / "regional_window_assignments.csv", index=False)
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(out_dir / "regional_weight_seed_runs.csv", index=False)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "regional_weight_pairwise_consistency.csv", index=False)
    summarize(run_df, "weight_scheme", ["wcss", "silhouette"]).to_csv(
        out_dir / "regional_weight_seed_summary.csv", index=False
    )
    return run_df, pair_df


def save_plots(seed_summary: pd.DataFrame, boot: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.errorbar(seed_summary["k"], seed_summary["wcss_mean"], yerr=seed_summary["wcss_sd"], marker="o", color="#2F5597")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("WCSS (mean ± SD)", color="#2F5597")
    ax2 = ax1.twinx()
    ax2.errorbar(seed_summary["k"], seed_summary["silhouette_mean"], yerr=seed_summary["silhouette_sd"], marker="s", color="#C55A11")
    ax2.set_ylabel("Silhouette (mean ± SD)", color="#C55A11")
    fig.tight_layout()
    fig.savefig(out_dir / "k_selection_seed_stability.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for k in range(2, 6):
        part = boot[boot["k"] == k]
        axes[0].boxplot(part["ari_vs_full"], positions=[k], widths=0.55)
        axes[1].boxplot(part["nmi_vs_full"], positions=[k], widths=0.55)
    axes[0].set_ylabel("ARI vs full-data solution")
    axes[1].set_ylabel("NMI vs full-data solution")
    for ax in axes:
        ax.set_xlabel("k")
        ax.set_xticks(range(2, 6))
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "group_bootstrap_stability.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--min-cells", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.features)
    missing = sorted(set(FEATURES + ["image", "cell_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    valid = x.notna().all(axis=1)
    dropped = int((~valid).sum())
    frame = frame.loc[valid].reset_index(drop=True)
    x = x.loc[valid].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(x)
    xz = scaler.transform(x)

    seed_summary, ref_labels = run_seed_stability(xz, args.out_dir)
    boot = run_group_bootstrap(frame, x, ref_labels, args.out_dir, args.bootstrap)
    save_plots(seed_summary, boot, args.out_dir)

    frame["cluster_k3_raw"] = ref_labels[3]
    ordered, order_detail = severity_order(frame, ref_labels[3])
    frame["subtype_ordered"] = ordered
    order_detail.to_csv(args.out_dir / "cluster_severity_order.csv", index=False)
    frame.to_csv(args.out_dir / "cells_reclustered_k3.csv", index=False)

    located, centroid_audit = add_centroids(frame, args.mat_dir)
    centroid_audit.to_csv(args.out_dir / "centroid_mapping_audit.csv", index=False)
    windows = make_windows(located, args.grid_size, args.min_cells)
    windows.to_csv(args.out_dir / "regional_windows_base.csv", index=False)
    region_runs, region_pairs = regional_sensitivity(windows, args.out_dir)

    manifest = {
        "feature_file": str(args.features),
        "mat_dir": str(args.mat_dir),
        "features": FEATURES,
        "n_cells_input": int(len(valid)),
        "n_cells_analyzed": int(len(frame)),
        "n_cells_dropped_nonfinite": dropped,
        "n_images": int(frame["image"].nunique()),
        "seeds": SEEDS,
        "bootstrap_unit": "image",
        "bootstrap_replicates": args.bootstrap,
        "grid_size": f"{args.grid_size}x{args.grid_size}",
        "minimum_cells_per_window": args.min_cells,
        "n_regional_windows": int(len(windows)),
        "regional_weight_schemes": {k: v.tolist() for k, v in WEIGHT_SCHEMES.items()},
        "regional_feature_vector": ["p_mild", "p_moderate", "p_severe", "injury_index", "density", "entropy", "x_norm", "y_norm"],
        "severity_order_rule": "ascending mean(-z(circularity)+z(curvature_std)+z(optical_contrast))",
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nSeed stability summary:\n", seed_summary.to_string(index=False))
    print("\nRegional weight consistency:\n", region_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
