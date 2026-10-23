# -*- coding: utf-8 -*-
"""Generic MAT inference for MyNet checkpoints (full model or ablation runs)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("desc", ckpt.get("model"))
    if state is None:
        raise KeyError(f"no 'desc'/'model' state dict in {path}")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    config = ckpt.get("config", {}) or {}
    return state, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(args.package_root)
    sys.path.insert(0, str(args.package_root))
    from models.MyNet.net_desc import mynet
    from models.MyNet.post_proc import process

    state, config = load_checkpoint(args.checkpoint, args.device)
    use_git = config.get("use_git")
    use_og = config.get("use_og")
    model = mynet(nr_types=3, use_git=use_git, use_og=use_og)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.eval().to(device)

    image_lookup = {p.stem.lower(): p for p in args.image_dir.iterdir() if p.is_file()}
    stems = sorted(p.stem for p in args.gt_dir.glob("*.mat"))
    audit = []
    for number, stem in enumerate(stems, start=1):
        start = time.time()
        path = image_lookup.get(stem.lower())
        if path is None:
            audit.append({"stem": stem, "status": "missing_image"})
            continue
        encoded = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            audit.append({"stem": stem, "status": "unreadable_image", "path": str(path)})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                    enabled=device.type == "cuda"):
            output = model(tensor)
            hv = output["hv"][0].permute(1, 2, 0).float().cpu().numpy()[:h, :w]
            tp = output["tp"].argmax(dim=1)[0].float().cpu().numpy()[:h, :w, None]
        pred_map = np.concatenate([hv, tp], axis=-1)
        inst_map_raw, info = process(pred_map, nr_types=3, return_centroids=True)

        inst_map = np.zeros_like(inst_map_raw, dtype=np.int32)
        centroids, types, probs = [], [], []
        new_id = 0
        for old_id in sorted(info):
            mask = inst_map_raw == old_id
            if not mask.any():
                continue
            new_id += 1
            inst_map[mask] = new_id
            centroids.append(np.asarray(info[old_id]["centroid"], dtype=np.float32))
            types.append([int(info[old_id]["type"])])
            probs.append([float(info[old_id]["type_prob"])])
        type_map = np.zeros_like(inst_map, dtype=np.int32)
        for inst_id, type_id in enumerate(np.asarray(types, dtype=np.int32).reshape(-1), start=1):
            type_map[inst_map == inst_id] = type_id
        sio.savemat(
            args.output_dir / f"{stem}.mat",
            {
                "inst_map": inst_map,
                "type_map": type_map,
                "inst_centroid": np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
                "inst_type": np.asarray(types, dtype=np.int32).reshape(-1, 1),
                "inst_type_prob": np.asarray(probs, dtype=np.float32).reshape(-1, 1),
            },
            do_compression=True,
        )
        audit.append({"stem": stem, "status": "ok", "n_instances": new_id, "seconds": time.time() - start})
        print(f"[{number}/{len(stems)}] {stem}: {new_id} instances", flush=True)

    manifest = {
        "package_root": str(args.package_root),
        "checkpoint": str(args.checkpoint),
        "config": config,
        "image_dir": str(args.image_dir),
        "gt_dir": str(args.gt_dir),
        "output_dir": str(args.output_dir),
        "n_requested": len(stems),
        "n_completed": sum(row["status"] == "ok" for row in audit),
        "audit": audit,
    }
    (args.output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
