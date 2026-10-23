# -*- coding: utf-8 -*-
"""Unified 12-px evaluation with detection P/R and bootstrap CIs.

Produces the reviewer-requested metric set for every method that has per-instance
MAT predictions available locally, then builds the corrected comparison table by
combining available results with the published per-class F1 values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

CLASS_NAMES = {1: "normal", 2: "injured"}


def pair_coordinates(true, pred, radius):
    if len(true) == 0 or len(pred) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(true)), np.arange(len(pred))
    d = np.sqrt(((true[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2))
    ti, pi = linear_sum_assignment(d)
    keep = d[ti, pi] <= radius
    pairs = np.column_stack([ti[keep], pi[keep]]).astype(int)
    un_t = np.setdiff1d(np.arange(len(true)), pairs[:, 0] if len(pairs) else [], assume_unique=True)
    un_p = np.setdiff1d(np.arange(len(pred)), pairs[:, 1] if len(pairs) else [], assume_unique=True)
    return pairs, un_t, un_p


def remap_label(x):
    out = np.zeros_like(x, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(x)[1:], start=1):
        out[x == old_id] = new_id
    return out


def binary_dice(true, pred):
    a, b = true > 0, pred > 0
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 1.0


def aji(true, pred):
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


def read_instances(path):
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
        types.append(int(vals[valid][np.argmax(counts[valid])]) if valid.any() else 0)
    return inst_map, np.asarray(centroids, dtype=np.float32).reshape(-1, 2), np.asarray(types, dtype=np.int32)


def metrics_from_pairs(yt, yp, ut, up):
    labels = [1, 2]
    out = {"n_true": int(len(yt) + len(ut)), "n_pred": int(len(yp) + len(up)), "n_matched": int(len(yt)),
           "n_unmatched_true": int(len(ut)), "n_unmatched_pred": int(len(up))}
    tp, fp, fn = len(yt), len(up), len(ut)
    out["detection_precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    out["detection_recall"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    out["detection_f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    out["overall_matched_instance_f1"] = out["detection_f1"]
    out["classification_matched_only"] = {
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": {f"{c}_{CLASS_NAMES.get(c, c)}": float(v)
                         for c, v in zip(labels, f1_score(yt, yp, labels=labels, average=None, zero_division=0))},
        "support": {f"{c}_{CLASS_NAMES.get(c, c)}": int((yt == c).sum()) for c in labels},
    }
    aware, supports = {}, {}
    for c in labels:
        tp_c = int(((yt == c) & (yp == c)).sum())
        fp_c = int(((yt != c) & (yp == c)).sum() + (up == c).sum())
        fn_c = int(((yt == c) & (yp != c)).sum() + (ut == c).sum())
        aware[f"{c}_{CLASS_NAMES.get(c, c)}"] = {
            "precision": float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) else 0.0,
            "recall": float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) else 0.0,
            "f1": float(2 * tp_c / (2 * tp_c + fp_c + fn_c)) if (2 * tp_c + fp_c + fn_c) else 0.0,
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
        }
        supports[f"{c}_{CLASS_NAMES.get(c, c)}"] = int((yt == c).sum() + (ut == c).sum())
    aware_f1 = np.array([v["f1"] for v in aware.values()])
    support_vec = np.array([supports[k] for k in aware])
    total_tp = sum(v["tp"] for v in aware.values())
    total_fp = sum(v["fp"] for v in aware.values())
    total_fn = sum(v["fn"] for v in aware.values())
    out["classification_detection_aware"] = {
        "macro_f1": float(aware_f1.mean()),
        "micro_f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn)),
        "weighted_f1": float(np.average(aware_f1, weights=support_vec)),
        "per_class": aware, "support": supports,
    }
    return out


def evaluate(method, pred_dir, gt_dir, radius):
    pred_files = {p.stem: p for p in Path(pred_dir).glob("*.mat")}
    gt_files = {p.stem: p for p in Path(gt_dir).glob("*.mat")}
    stems = sorted(set(pred_files) & set(gt_files))
    if not stems:
        raise ValueError(f"No matching MAT stems for {method}")
    per_image, arrays = [], {"yt": [], "yp": [], "ut": [], "up": [], "dice": [], "aji": []}
    for stem in stems:
        tm, tc, tt = read_instances(gt_files[stem])
        pm, pc, pt = read_instances(pred_files[stem])
        pairs, un_t, un_p = pair_coordinates(tc, pc, radius)
        yt = tt[pairs[:, 0]] if len(pairs) else np.empty(0, int)
        yp = pt[pairs[:, 1]] if len(pairs) else np.empty(0, int)
        ut, up = tt[un_t], pt[un_p]
        dice, aji_v = binary_dice(tm, pm), aji(tm, pm)
        row = metrics_from_pairs(yt, yp, ut, up)
        row.update({"image": stem, "dice": dice, "aji": aji_v})
        per_image.append(row)
        arrays["yt"].append(yt); arrays["yp"].append(yp); arrays["ut"].append(ut); arrays["up"].append(up)
        arrays["dice"].append(dice); arrays["aji"].append(aji_v)
    result = metrics_from_pairs(*(np.concatenate(arrays[k]) for k in ("yt", "yp", "ut", "up")))
    result.update({
        "method": method, "matching_radius_px": radius, "n_images": len(stems),
        "segmentation_image_mean": {
            "dice_mean": float(np.mean(arrays["dice"])), "dice_sd": float(np.std(arrays["dice"], ddof=1)),
            "aji_mean": float(np.mean(arrays["aji"])), "aji_sd": float(np.std(arrays["aji"], ddof=1)),
        },
    })
    # bootstrap over images
    rng = np.random.default_rng(20261022)
    n = len(stems)
    boot = {"dice": [], "aji": [], "detection_f1": [], "macro_f1_matched": [], "macro_f1_aware": []}
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boot["dice"].append(float(np.mean([arrays["dice"][i] for i in idx])))
        boot["aji"].append(float(np.mean([arrays["aji"][i] for i in idx])))
        yt = np.concatenate([arrays["yt"][i] for i in idx]); yp = np.concatenate([arrays["yp"][i] for i in idx])
        ut = np.concatenate([arrays["ut"][i] for i in idx]); up = np.concatenate([arrays["up"][i] for i in idx])
        m = metrics_from_pairs(yt, yp, ut, up)
        boot["detection_f1"].append(m["detection_f1"])
        boot["macro_f1_matched"].append(m["classification_matched_only"]["macro_f1"])
        boot["macro_f1_aware"].append(m["classification_detection_aware"]["macro_f1"])
    result["bootstrap_95ci"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()
    }
    return result, pd.DataFrame(per_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, help="NAME=prediction_directory")
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, frames = [], []
    for spec in args.method:
        name, directory = spec.split("=", 1)
        res, img = evaluate(name, Path(directory), args.gt_dir, args.radius)
        results.append(res); frames.append(img)
        print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])
    (args.out_dir / "unified_metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(args.out_dir / "unified_metrics_per_image.csv", index=False)


if __name__ == "__main__":
    main()
