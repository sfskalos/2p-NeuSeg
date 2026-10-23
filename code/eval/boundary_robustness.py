"""Fixed-model sensitivity to controlled GT contour perturbations.

This is an evaluation robustness analysis: model predictions remain fixed while
reference instance contours are dilated, eroded, or locally jittered.  Jitter is
implemented as smooth noise added to each instance's signed distance field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.metrics import f1_score

from evaluate_12px import aji, binary_dice, pair_coordinates, read_instances


def resolve_dilated(original: np.ndarray, union: np.ndarray) -> np.ndarray:
    _, indices = distance_transform_edt(original == 0, return_indices=True)
    nearest = original[indices[0], indices[1]]
    return np.where(union, nearest, 0).astype(np.int32)


def morph_labels(inst_map: np.ndarray, radius: int, operation: str) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = np.zeros_like(inst_map, dtype=np.int32)
    if operation == "dilate":
        union = cv2.dilate((inst_map > 0).astype(np.uint8), kernel) > 0
        return resolve_dilated(inst_map, union)
    for inst_id in np.unique(inst_map)[1:]:
        mask = cv2.erode((inst_map == inst_id).astype(np.uint8), kernel) > 0
        out[mask] = int(inst_id)
    return out


def jitter_labels(inst_map: np.ndarray, radius: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.zeros_like(inst_map, dtype=np.int32)
    best_score = np.full(inst_map.shape, -np.inf, dtype=np.float32)
    for inst_id in np.unique(inst_map)[1:]:
        mask = inst_map == inst_id
        inside = distance_transform_edt(mask)
        outside = distance_transform_edt(~mask)
        signed = inside - outside
        noise = gaussian_filter(rng.normal(size=inst_map.shape).astype(np.float32), sigma=max(1.0, radius * 1.5))
        noise /= noise.std() + 1e-8
        score = signed + noise * radius
        perturbed = score > 0
        take = perturbed & (score > best_score)
        candidates[take] = int(inst_id)
        best_score[take] = score[take]
    return candidates


def instances_from_perturbed(inst_map: np.ndarray, original_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids, types = [], []
    for inst_id in np.unique(inst_map)[1:]:
        yy, xx = np.where(inst_map == inst_id)
        if len(xx) == 0:
            continue
        centroids.append([xx.mean(), yy.mean()])
        types.append(int(original_types[int(inst_id) - 1]))
    return np.asarray(centroids, dtype=np.float32).reshape(-1, 2), np.asarray(types, dtype=np.int32)


def summarize_global(records: list[dict], radius_match: float) -> dict:
    yt = np.concatenate([r["paired_true"] for r in records])
    yp = np.concatenate([r["paired_pred"] for r in records])
    ut = np.concatenate([r["unpaired_true"] for r in records])
    up = np.concatenate([r["unpaired_pred"] for r in records])
    labels = sorted(int(x) for x in np.unique(np.concatenate([yt, yp, ut, up])) if x > 0)
    det_f1 = 2 * len(yt) / (2 * len(yt) + len(ut) + len(up))
    return {
        "matching_radius_px": radius_match,
        "n_true": int(len(yt) + len(ut)),
        "n_pred": int(len(yp) + len(up)),
        "n_matched": int(len(yt)),
        "detection_f1": float(det_f1),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": {str(c): float(v) for c, v in zip(labels, f1_score(yt, yp, labels=labels, average=None, zero_division=0))},
        "dice_mean": float(np.mean([r["dice"] for r in records])),
        "aji_mean": float(np.mean([r["aji"] for r in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-radius", type=float, default=12.0)
    parser.add_argument("--jitter-replicates", type=int, default=30)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(set(p.stem for p in args.gt_dir.glob("*.mat")) & set(p.stem for p in args.pred_dir.glob("*.mat")))
    conditions = [("baseline", 0, 0)]
    conditions += [(op, radius, 0) for op in ("dilate", "erode") for radius in (1, 2, 3)]
    conditions += [("jitter", radius, replicate) for radius in (1, 2, 3) for replicate in range(args.jitter_replicates)]
    summary_rows, image_rows = [], []
    for operation, radius, replicate in conditions:
        records = []
        for image_index, stem in enumerate(stems):
            gt_map, _, gt_types = read_instances(args.gt_dir / f"{stem}.mat")
            pred_map, pred_centroids, pred_types = read_instances(args.pred_dir / f"{stem}.mat")
            if operation == "baseline":
                perturbed = gt_map
            elif operation in ("dilate", "erode"):
                perturbed = morph_labels(gt_map, radius, operation)
            else:
                rng = np.random.default_rng(20261022 + 100000 * radius + 1000 * replicate + image_index)
                perturbed = jitter_labels(gt_map, radius, rng)
            gt_centroids, perturbed_types = instances_from_perturbed(perturbed, gt_types)
            pairs, un_t, un_p = pair_coordinates(gt_centroids, pred_centroids, args.matching_radius)
            record = {
                "paired_true": perturbed_types[pairs[:, 0]] if len(pairs) else np.empty(0, dtype=int),
                "paired_pred": pred_types[pairs[:, 1]] if len(pairs) else np.empty(0, dtype=int),
                "unpaired_true": perturbed_types[un_t],
                "unpaired_pred": pred_types[un_p],
                "dice": binary_dice(perturbed, pred_map),
                "aji": aji(perturbed, pred_map),
            }
            records.append(record)
            image_rows.append(
                {
                    "operation": operation,
                    "radius_px": radius,
                    "replicate": replicate,
                    "image": stem,
                    "dice": record["dice"],
                    "aji": record["aji"],
                    "n_true": len(gt_centroids),
                    "n_pred": len(pred_centroids),
                    "n_matched": len(pairs),
                }
            )
        global_result = summarize_global(records, args.matching_radius)
        global_result.update({"operation": operation, "radius_px": radius, "replicate": replicate, "n_images": len(stems)})
        summary_rows.append(global_result)
        print(operation, radius, replicate, global_result["dice_mean"], global_result["aji_mean"], global_result["macro_f1"])
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "boundary_robustness_global.csv", index=False)
    pd.DataFrame(image_rows).to_csv(args.out_dir / "boundary_robustness_per_image.csv", index=False)
    aggregate = summary.groupby(["operation", "radius_px"], as_index=False).agg(
        n_replicates=("replicate", "count"),
        dice_mean=("dice_mean", "mean"), dice_sd=("dice_mean", "std"),
        aji_mean=("aji_mean", "mean"), aji_sd=("aji_mean", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        micro_f1_mean=("micro_f1", "mean"), micro_f1_sd=("micro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_sd=("weighted_f1", "std"),
        detection_f1_mean=("detection_f1", "mean"), detection_f1_sd=("detection_f1", "std"),
    )
    aggregate.to_csv(args.out_dir / "boundary_robustness_summary.csv", index=False)
    manifest = {
        "interpretation": "fixed-model evaluation sensitivity; no retraining",
        "jitter_method": "independent smooth Gaussian noise added to each instance signed-distance field",
        "jitter_replicates": args.jitter_replicates,
        "matching_radius_px": args.matching_radius,
        "n_images": len(stems),
        "gt_dir": str(args.gt_dir),
        "pred_dir": str(args.pred_dir),
    }
    (args.out_dir / "boundary_robustness_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
