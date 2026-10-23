"""Uniform 12-pixel centroid-matching evaluation for one or more methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score


def pair_coordinates(true: np.ndarray, pred: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(true) == 0 or len(pred) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(true)), np.arange(len(pred))
    d = np.sqrt(((true[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2))
    ti, pi = linear_sum_assignment(d)
    keep = d[ti, pi] <= radius
    pairs = np.column_stack([ti[keep], pi[keep]]).astype(int)
    unmatched_true = np.setdiff1d(np.arange(len(true)), pairs[:, 0] if len(pairs) else [], assume_unique=True)
    unmatched_pred = np.setdiff1d(np.arange(len(pred)), pairs[:, 1] if len(pairs) else [], assume_unique=True)
    return pairs, unmatched_true, unmatched_pred


def remap_label(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(x)[1:], start=1):
        out[x == old_id] = new_id
    return out


def binary_dice(true: np.ndarray, pred: np.ndarray) -> float:
    a, b = true > 0, pred > 0
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 1.0


def aji(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = remap_label(true), remap_label(pred)
    true_ids, pred_ids = np.unique(true)[1:], np.unique(pred)[1:]
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    intersections = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    unions = np.zeros_like(intersections)
    pred_areas = np.array([(pred == j).sum() for j in pred_ids])
    true_areas = np.array([(true == i).sum() for i in true_ids])
    for ai, i in enumerate(true_ids):
        mask = true == i
        overlap = np.unique(pred[mask])
        for j in overlap[overlap > 0]:
            bj = int(j - 1)
            inter = np.logical_and(mask, pred == j).sum()
            intersections[ai, bj] = inter
            unions[ai, bj] = true_areas[ai] + pred_areas[bj] - inter
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    best_pred = iou.argmax(axis=1)
    best_iou = iou[np.arange(len(true_ids)), best_pred]
    matched = best_iou > 0
    used_pred = np.unique(best_pred[matched])
    overall_inter = intersections[np.arange(len(true_ids))[matched], best_pred[matched]].sum()
    overall_union = unions[np.arange(len(true_ids))[matched], best_pred[matched]].sum()
    overall_union += true_areas[~matched].sum()
    overall_union += pred_areas[np.setdiff1d(np.arange(len(pred_ids)), used_pred)].sum()
    return float(overall_inter / overall_union) if overall_union else 1.0


def read_instances(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = sio.loadmat(path)
    inst_map = np.asarray(data["inst_map"], dtype=np.int32)
    if "inst_centroid" in data and "inst_type" in data:
        centroids = np.asarray(data["inst_centroid"], dtype=np.float32).reshape(-1, 2)
        types = np.asarray(data["inst_type"], dtype=np.int32).reshape(-1)
        return inst_map, centroids, types
    centroids, types = [], []
    type_map = np.asarray(data.get("type_map", np.zeros_like(inst_map)), dtype=np.int32)
    for inst_id in np.unique(inst_map)[1:]:
        yy, xx = np.where(inst_map == inst_id)
        centroids.append([xx.mean(), yy.mean()])
        vals, counts = np.unique(type_map[inst_map == inst_id], return_counts=True)
        valid = vals > 0
        types.append(vals[valid][np.argmax(counts[valid])] if valid.any() else 0)
    return inst_map, np.asarray(centroids, dtype=np.float32).reshape(-1, 2), np.asarray(types, dtype=np.int32)


def evaluate(method: str, pred_dir: Path, gt_dir: Path, radius: float) -> tuple[dict, pd.DataFrame]:
    pred_files = {p.stem: p for p in pred_dir.glob("*.mat")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.mat")}
    stems = sorted(set(pred_files) & set(gt_files))
    if not stems:
        raise ValueError(f"No matching MAT stems for {method}: {pred_dir}")
    per_image = []
    paired_true_all, paired_pred_all, un_true_all, un_pred_all = [], [], [], []
    for stem in stems:
        tm, tc, tt = read_instances(gt_files[stem])
        pm, pc, pt = read_instances(pred_files[stem])
        pairs, un_t, un_p = pair_coordinates(tc, pc, radius)
        ptt = tt[pairs[:, 0]] if len(pairs) else np.empty(0, dtype=int)
        ppt = pt[pairs[:, 1]] if len(pairs) else np.empty(0, dtype=int)
        paired_true_all.append(ptt)
        paired_pred_all.append(ppt)
        un_true_all.append(tt[un_t])
        un_pred_all.append(pt[un_p])
        det_f1 = 2 * len(pairs) / (2 * len(pairs) + len(un_t) + len(un_p)) if (len(tc) + len(pc)) else 1.0
        per_image.append(
            {
                "method": method,
                "image": stem,
                "n_true": len(tc),
                "n_pred": len(pc),
                "n_matched": len(pairs),
                "n_unmatched_true": len(un_t),
                "n_unmatched_pred": len(un_p),
                "detection_f1": det_f1,
                "dice": binary_dice(tm, pm),
                "aji": aji(tm, pm),
            }
        )
    yt = np.concatenate(paired_true_all)
    yp = np.concatenate(paired_pred_all)
    ut = np.concatenate(un_true_all)
    up = np.concatenate(un_pred_all)
    labels = sorted(int(v) for v in np.unique(np.concatenate([yt, yp, ut, up])) if v > 0)
    tp_det, fn_det, fp_det = len(yt), len(ut), len(up)
    det_f1 = 2 * tp_det / (2 * tp_det + fp_det + fn_det)
    matched_per_class = f1_score(yt, yp, labels=labels, average=None, zero_division=0)

    # Detection-aware class scores: wrong-class matches count once as FP for the
    # predicted class and once as FN for the true class; unmatched instances also count.
    aware = {}
    for class_id in labels:
        tp = int(((yt == class_id) & (yp == class_id)).sum())
        fp = int(((yt != class_id) & (yp == class_id)).sum() + (up == class_id).sum())
        fn = int(((yt == class_id) & (yp != class_id)).sum() + (ut == class_id).sum())
        aware[class_id] = {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0}
    supports = {class_id: int((yt == class_id).sum() + (ut == class_id).sum()) for class_id in labels}
    aware_f1 = np.array([aware[c]["f1"] for c in labels])
    support_vec = np.array([supports[c] for c in labels])
    total_tp = sum(v["tp"] for v in aware.values())
    total_fp = sum(v["fp"] for v in aware.values())
    total_fn = sum(v["fn"] for v in aware.values())

    image_df = pd.DataFrame(per_image)
    result = {
        "method": method,
        "matching_radius_px": radius,
        "n_images": len(stems),
        "n_true": int(tp_det + fn_det),
        "n_pred": int(tp_det + fp_det),
        "n_matched": int(tp_det),
        "detection_f1": float(det_f1),
        "classification_matched_only": {
            "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
            "per_class_f1": {str(c): float(v) for c, v in zip(labels, matched_per_class)},
            "support": {str(c): int((yt == c).sum()) for c in labels},
        },
        "classification_detection_aware": {
            "macro_f1": float(aware_f1.mean()),
            "micro_f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn)),
            "weighted_f1": float(np.average(aware_f1, weights=support_vec)),
            "per_class": {str(k): v for k, v in aware.items()},
            "support": {str(k): v for k, v in supports.items()},
        },
        "segmentation_image_mean": {
            "dice_mean": float(image_df["dice"].mean()),
            "dice_sd": float(image_df["dice"].std(ddof=1)),
            "aji_mean": float(image_df["aji"].mean()),
            "aji_sd": float(image_df["aji"].std(ddof=1)),
        },
    }
    return result, image_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, help="NAME=prediction_directory")
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results, all_images = [], []
    for spec in args.method:
        if "=" not in spec:
            raise ValueError("--method must be NAME=directory")
        name, directory = spec.split("=", 1)
        result, images = evaluate(name, Path(directory), args.gt_dir, args.radius)
        all_results.append(result)
        all_images.append(images)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    (args.out_dir / "metrics_12px.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat(all_images, ignore_index=True).to_csv(args.out_dir / "metrics_12px_per_image.csv", index=False)


if __name__ == "__main__":
    main()
