"""Reproducible multi-seed OGS/GIF ablation and OGS-loss comparison.

The script uses the supplied model package, but intentionally excludes its newly
added NP decoder from the objective and derives foreground from the TP branch.
This preserves the original paper's HV+TP decoding logic while retaining the
package's explicit GIF/OG switches. Images stay on the original 0--255 scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if not hasattr(np, "bool"):
    np.bool = np.bool_


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PatchDataset(Dataset):
    def __init__(self, files: list[Path], target_fn, train: bool):
        self.files = files
        self.target_fn = target_fn
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[index])
        image = data[..., :3].astype(np.uint8)
        inst = data[..., 3].astype(np.int32)
        type_map = data[..., 4].astype(np.int64)
        if self.train:
            # Shape transforms are applied identically to image and labels.
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy(); inst = np.rot90(inst, k).copy(); type_map = np.rot90(type_map, k).copy()
            if np.random.rand() < 0.5:
                image = image[:, ::-1].copy(); inst = inst[:, ::-1].copy(); type_map = type_map[:, ::-1].copy()
            if np.random.rand() < 0.5:
                image = image[::-1].copy(); inst = inst[::-1].copy(); type_map = type_map[::-1].copy()
            # Mild intensity perturbation, matching the manuscript description.
            gain = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-12, 12)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        targets = self.target_fn(image.copy(), inst.copy(), [384, 384])
        return {
            "img": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "inst_map": torch.from_numpy(inst.copy()).long(),
            "tp_map": torch.from_numpy(type_map.copy()).long(),
            "hv_map": torch.from_numpy(targets["hv_map"].copy()).float(),
            "og_map": torch.from_numpy(targets["og_map"].copy()).float(),
        }


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    return (1 - (2 * inter + 1e-3) / (denom + 1e-3)).sum()


def boundary_loss(pred: torch.Tensor, target_255: torch.Tensor, mode: str, original_head: bool) -> torch.Tensor:
    target = (target_255 > 0).float()
    if original_head and mode == "mse":
        return F.mse_loss(pred.squeeze(1), target_255)
    logits = pred.squeeze(1)
    prob = torch.sigmoid(logits)
    if mode == "mse":
        return F.mse_loss(prob, target)
    if mode == "weighted_mse":
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = torch.clamp(negative / (positive + 1), 1.0, 20.0)
        weights = torch.where(target > 0, pos_weight, 1.0)
        return (weights * (prob - target).square()).mean()
    if mode == "focal_bce":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target > 0, prob, 1 - prob)
        alpha_t = torch.where(target > 0, 0.75, 0.25)
        return (alpha_t * (1 - pt).square() * bce).mean()
    if mode == "dice_tversky":
        dims = (1, 2)
        tp = (prob * target).sum(dims)
        fp = (prob * (1 - target)).sum(dims)
        fn = ((1 - prob) * target).sum(dims)
        tversky = (tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5)
        return (1 - tversky).mean()
    raise ValueError(mode)


def compute_loss(output: dict, batch: dict, msge_fn, use_og: bool, og_mode: str, original_head: bool) -> tuple[torch.Tensor, dict]:
    hv_pred = output["hv"].permute(0, 2, 3, 1).contiguous().float()
    hv_true = batch["hv_map"].float()
    focus = (batch["inst_map"] > 0).long()
    terms = {
        "hv_mse": F.mse_loss(hv_pred, hv_true),
        "hv_msge": msge_fn(hv_true, hv_pred, focus),
        "tp_ce": F.cross_entropy(output["tp"].float(), batch["tp_map"]),
        "tp_dice": multiclass_dice_loss(output["tp"].float(), batch["tp_map"]),
    }
    if use_og:
        terms[f"og_{og_mode}"] = boundary_loss(output["og"].float(), batch["og_map"].float(), og_mode, original_head)
    return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


def remap_label(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int32)
    for new, old in enumerate(np.unique(x)[1:], start=1):
        out[x == old] = new
    return out


def centroids_types(inst: np.ndarray, type_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs, ts = [], []
    for iid in np.unique(inst)[1:]:
        yy, xx = np.where(inst == iid)
        if not len(xx):
            continue
        cs.append([xx.mean(), yy.mean()])
        vals, counts = np.unique(type_map[inst == iid], return_counts=True)
        mask = vals > 0
        ts.append(int(vals[mask][np.argmax(counts[mask])]) if mask.any() else 0)
    return np.asarray(cs, np.float32).reshape(-1, 2), np.asarray(ts, np.int32)


def pair_coordinates(a: np.ndarray, b: np.ndarray, radius: float = 12.0):
    from scipy.optimize import linear_sum_assignment
    if not len(a) or not len(b):
        return np.empty((0, 2), int), np.arange(len(a)), np.arange(len(b))
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(2))
    i, j = linear_sum_assignment(d)
    keep = d[i, j] <= radius
    pairs = np.column_stack([i[keep], j[keep]])
    return pairs, np.setdiff1d(np.arange(len(a)), pairs[:, 0]), np.setdiff1d(np.arange(len(b)), pairs[:, 1])


def validate(model, loader, process_fn, dice_fn, aji_fn, device) -> dict:
    """Validate with the official inference post-processing (models.MyNet.post_proc).

    The earlier version used run_ablation.post_proc_func_from_nphv_tp, which
    assigns almost every pixel to an instance and produced Dice ~0.126 while the
    same weights reach Dice ~0.80 with the official decoder.  Model selection is
    therefore based on the same pipeline that is used for the reported test
    metrics.
    """
    model.eval()
    dices, ajis = [], []
    yt_all, yp_all, ut_all, up_all = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(images)
            prob = torch.softmax(output["tp"].float(), dim=1)
            types_px = prob.argmax(1).cpu().numpy()
            hv = output["hv"].permute(0, 2, 3, 1).float().cpu().numpy()
            for n in range(len(images)):
                pred_map = np.concatenate([hv[n], types_px[n][..., None]], axis=-1)
                inst_raw, info = process_fn(pred_map, nr_types=3, return_centroids=True)
                true_inst = remap_label(batch["inst_map"][n].numpy())
                true_type_map = batch["tp_map"][n].numpy()
                pred_inst = np.zeros_like(inst_raw, dtype=np.int32)
                pred_type_map = np.zeros_like(inst_raw, dtype=np.int32)
                new_id = 0
                for old_id in sorted(info):
                    mask = inst_raw == old_id
                    if not mask.any():
                        continue
                    new_id += 1
                    pred_inst[mask] = new_id
                    pred_type_map[mask] = int(info[old_id]["type"])
                pc, pt = centroids_types(pred_inst, pred_type_map)
                tc, tt = centroids_types(true_inst, true_type_map)
                pairs, ut, up = pair_coordinates(tc, pc, 12.0)
                yt_all.append(tt[pairs[:, 0]] if len(pairs) else np.empty(0, int))
                yp_all.append(pt[pairs[:, 1]] if len(pairs) else np.empty(0, int))
                ut_all.append(tt[ut]); up_all.append(pt[up])
                dices.append(float(dice_fn(true_inst, pred_inst)))
                ajis.append(float(aji_fn(true_inst, pred_inst)))
    from sklearn.metrics import f1_score
    yt, yp, ut, up = map(np.concatenate, (yt_all, yp_all, ut_all, up_all))
    labels = [1, 2]
    det = 2 * len(yt) / (2 * len(yt) + len(ut) + len(up))
    return {
        "dice": float(np.mean(dices)), "aji": float(np.mean(ajis)),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        "class1_f1": float(f1_score(yt, yp, labels=labels, average=None, zero_division=0)[0]),
        "class2_f1": float(f1_score(yt, yp, labels=labels, average=None, zero_division=0)[1]),
        "detection_f1": float(det), "n_true": int(len(yt) + len(ut)), "n_pred": int(len(yp) + len(up)), "n_matched": int(len(yt)),
    }


def configuration_list(experiment: str):
    if experiment == "ablation":
        return [
            {"name": "GIF0_OGS0", "use_git": False, "use_og": False, "og_mode": "mse", "original_head": True},
            {"name": "GIF0_OGS1", "use_git": False, "use_og": True, "og_mode": "mse", "original_head": True},
            {"name": "GIF1_OGS0", "use_git": True, "use_og": False, "og_mode": "mse", "original_head": True},
            {"name": "GIF1_OGS1", "use_git": True, "use_og": True, "og_mode": "mse", "original_head": True},
        ]
    return [
        {"name": f"OGS_{mode}", "use_git": True, "use_og": True, "og_mode": mode, "original_head": False}
        for mode in ("mse", "weighted_mse", "focal_bce", "dice_tversky")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=["ablation", "loss"], required=True)
    parser.add_argument("--seeds", default="10,20,30")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only", default="", help="Optional exact configuration name")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.package_root)
    sys.path.insert(0, str(args.package_root))
    from models.MyNet.net_desc import mynet
    from models.MyNet.targets import gen_targets
    from models.MyNet.utils import msge_loss
    from metrics.stats_utils import get_dice_1, get_fast_aji
    from models.MyNet.post_proc import process as official_process

    train_files = sorted(args.train_dir.glob("*.npy"))
    # Only the non-transformed base patch is used for validation, preventing
    # four offline augmentations of one field from being treated as replicates.
    valid_files = sorted(args.valid_dir.glob("*_000.npy"))
    if not valid_files:
        valid_files = sorted(args.valid_dir.glob("*.npy"))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    configs = configuration_list(args.experiment)
    if args.only:
        configs = [x for x in configs if x["name"] == args.only]
    manifest = vars(args).copy()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in manifest.items()}
    manifest.update({"n_train": len(train_files), "n_valid_independent": len(valid_files), "configs": configs})
    (args.out_dir / f"manifest_{args.experiment}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_best = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config in configs:
        for seed in seeds:
            run_dir = args.out_dir / args.experiment / config["name"] / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "best_metrics.json"
            if result_path.exists():
                all_best.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            seed_everything(seed)
            model = mynet(nr_types=3, use_git=config["use_git"], use_og=config["use_og"]).to(device)
            if not config["original_head"] and model.texture_encoder is not None:
                model.texture_encoder.conv_out.relu = torch.nn.Identity()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            train_loader = DataLoader(PatchDataset(train_files, gen_targets, True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            valid_loader = DataLoader(PatchDataset(valid_files, gen_targets, False), batch_size=max(1, args.batch_size * 2), shuffle=False, num_workers=args.num_workers)
            best_score, stale, history = -np.inf, 0, []
            for epoch in range(1, args.epochs + 1):
                tick = time.time(); model.train(); optimizer.zero_grad(set_to_none=True); losses = []
                for step, batch in enumerate(train_loader, start=1):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        output = model(batch["img"])
                        loss, _ = compute_loss(output, batch, msge_loss, config["use_og"], config["og_mode"], config["original_head"])
                        loss = loss / args.accumulation
                    scaler.scale(loss).backward()
                    if step % args.accumulation == 0 or step == len(train_loader):
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach()) * args.accumulation)
                scheduler.step()
                metrics = validate(model, valid_loader, official_process, get_dice_1, get_fast_aji, device)
                metrics.update({"experiment": args.experiment, "configuration": config["name"], "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.time() - tick})
                history.append(metrics)
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                score = metrics["aji"]
                print(json.dumps(metrics), flush=True)
                if score > best_score:
                    best_score, stale = score, 0
                    torch.save({"model": model.state_dict(), "config": config, "seed": seed, "epoch": epoch, "metrics": metrics}, run_dir / "best_model.tar")
                    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            best = json.loads(result_path.read_text(encoding="utf-8")); all_best.append(best)
    best_df = pd.DataFrame(all_best)
    best_df.to_csv(args.out_dir / f"{args.experiment}_best_by_seed.csv", index=False)
    numeric = ["dice", "aji", "macro_f1", "micro_f1", "weighted_f1", "class1_f1", "class2_f1", "detection_f1"]
    if len(best_df):
        summary = best_df.groupby("configuration")[numeric].agg(["mean", "std"])
        summary.to_csv(args.out_dir / f"{args.experiment}_mean_sd.csv")


if __name__ == "__main__":
    main()
