# -*- coding: utf-8 -*-
"""
消融实验配置 - GIT替代方法对比（实例级指标版本，线程/worker 安全版）
==============================================================

你遇到的错误：
  Can't spawn new thread / DataLoader worker exited unexpectedly
本脚本已做：
  1) 限制 OpenMP/MKL/OpenBLAS/NumExpr 线程为 1
  2) 禁止 OpenCV 多线程
  3) torch intra/inter-op 线程设为 1
  4) DataLoader num_workers 默认降到很小（可命令行调整）
  5) mp sharing_strategy=file_system（减少资源共享导致的线程/FD问题）
"""

# ========= 必须最先：限制各种线程 =========
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ========= 再 import 其他库 =========
from scipy.optimize import linear_sum_assignment  # noqa: F401
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

if not hasattr(np, "bool"):
    np.bool = np.bool_

# 建议：减少 torch multiprocessing 共享资源的线程/FD压力
import torch.multiprocessing as tmp
try:
    tmp.set_sharing_strategy("file_system")
except Exception:
    pass

# torch 线程限制
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# OpenCV 线程限制（必须在 import cv2 后立刻设置）
import cv2
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.utils import check_manual_seed
from models.MyNet.targets import gen_targets  # noqa: F401

from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    remap_label,
)

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Model
# =========================================================
def create_ablation_model(fusion_method="GIT", nr_types=3, use_git=None, use_og=True):
    from models.MyNet.net_desc import mynet
    model = mynet(
        channel=256,
        M=[32, 32, 32],
        N=[8, 16, 32],
        nr_types=nr_types,
        fusion_method=fusion_method,
        use_git=use_git,
        use_og=use_og,
    )
    return model


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.randint(0, 2**32, (1,))[0].cpu().item() + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


# =========================================================
# Safe helpers (避免 empty argmax / 空实例崩溃)
# =========================================================
def safe_remap_label(inst_map: np.ndarray) -> np.ndarray:
    inst_map = inst_map.astype(np.int32, copy=False)
    if inst_map.size == 0 or inst_map.max() == 0:
        return inst_map
    return remap_label(inst_map, by_size=False)


def safe_instance_metrics(gt_inst: np.ndarray, pr_inst: np.ndarray):
    gt_inst = gt_inst.astype(np.int32, copy=False)
    pr_inst = pr_inst.astype(np.int32, copy=False)
    gt_has = (gt_inst.size > 0) and (gt_inst.max() > 0)
    pr_has = (pr_inst.size > 0) and (pr_inst.max() > 0)

    if (not gt_has) and (not pr_has):
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    if (not gt_has) and pr_has:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if gt_has and (not pr_has):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        pq_info = get_fast_pq(gt_inst, pr_inst, match_iou=0.5)[0]  # [dq,sq,pq]
        dsc = float(get_dice_1(gt_inst, pr_inst))
        aji = float(get_fast_aji(gt_inst, pr_inst))
        aji_plus = float(get_fast_aji_plus(gt_inst, pr_inst))
        dq, sq, pq = float(pq_info[0]), float(pq_info[1]), float(pq_info[2])
        return dsc, aji, aji_plus, dq, sq, pq
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


# =========================================================
# Loss
# =========================================================
def compute_loss(output, target, use_og=True):
    import torch.nn.functional as F
    from models.MyNet.utils import mse_loss, msge_loss, dice_loss

    losses = {}

    def ensure_nhwc(x):
        x = x.float()
        if x.dim() == 4:
            if x.shape[1] <= 8 and x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
                return x.permute(0, 2, 3, 1).contiguous()
        return x

    def ensure_nhw(x):
        x = x.float()
        if x.dim() == 4:
            if x.shape[1] == 1:
                return x.squeeze(1)
            if x.shape[-1] == 1:
                return x.squeeze(-1)
        return x

    np_pred = output["np"].float()
    if np_pred.dim() == 4 and np_pred.shape[-1] <= 10:
        np_pred = np_pred.permute(0, 3, 1, 2).contiguous()

    np_target_long = ensure_nhw(target["np_map"]).long()
    losses["np_ce"] = F.cross_entropy(np_pred, np_target_long).float()

    np_target_onehot = F.one_hot(np_target_long, num_classes=np_pred.shape[1]).permute(0, 3, 1, 2).float()
    losses["np_dice"] = dice_loss(np_target_onehot, torch.softmax(np_pred, dim=1)).float()

    hv_pred = ensure_nhwc(output["hv"])
    hv_target = ensure_nhwc(target["hv_map"])
    np_target = np_target_long

    losses["hv_mse"] = mse_loss(hv_target, hv_pred).float()
    losses["hv_msge"] = msge_loss(hv_target, hv_pred, np_target).float()

    tp_pred = output["tp"].float()
    if tp_pred.dim() == 4 and tp_pred.shape[-1] <= 10:
        tp_pred = tp_pred.permute(0, 3, 1, 2).contiguous()

    tp_target_long = ensure_nhw(target["tp_map"]).long()
    losses["tp_ce"] = F.cross_entropy(tp_pred, tp_target_long).float()

    nr_classes = tp_pred.shape[1]
    tp_target_onehot = F.one_hot(tp_target_long, num_classes=nr_classes).permute(0, 3, 1, 2).float()
    losses["tp_dice"] = dice_loss(tp_target_onehot, torch.softmax(tp_pred, dim=1)).float()

    if use_og:
        og_pred = ensure_nhw(output["og"])
        og_target = ensure_nhw(target["og_map"])
        losses["og_mse"] = F.mse_loss(og_pred.float(), og_target.float()).float()

    return losses


def ablation_tag(use_git, use_og):
    git_tag = "GIT1" if use_git else "GIT0"
    og_tag = "OG1" if use_og else "OG0"
    return f"{git_tag}_{og_tag}"


# =========================================================
# Post-process: np/hv/tp -> inst_map
# =========================================================
def _safe_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def post_proc_func_from_nphv_tp(
    np_map: np.ndarray,
    hv_map: np.ndarray,
    tp_map_or_predcls: np.ndarray = None,
    hv_grad_thresh: float = 0.35,
    dt_fg_thresh: float = 0.45,
    min_inst_area: int = 10,
):
    from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing
    from scipy.ndimage import label as ndi_label

    if np_map.dtype != np.bool_:
        fg = (np_map > 0.5)
    else:
        fg = np_map.copy()

    fg = binary_opening(fg, structure=np.ones((3, 3), np.uint8))
    fg = binary_closing(fg, structure=np.ones((3, 3), np.uint8))

    if fg.sum() == 0:
        h, w = fg.shape
        return np.zeros((h, w), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    hvx = hv_map[..., 0].astype(np.float32)
    hvy = hv_map[..., 1].astype(np.float32)
    gx = cv2.Sobel(hvx, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hvy, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    dt = distance_transform_edt(fg)
    dt_norm = dt / (dt.max() + 1e-8) if dt.max() > 0 else dt
    sure_fg = (dt_norm > dt_fg_thresh) & fg

    markers, n_mark = ndi_label(sure_fg.astype(np.uint8))
    if n_mark == 0:
        markers, n_mark = ndi_label(fg.astype(np.uint8))
        if n_mark == 0:
            h, w = fg.shape
            return np.zeros((h, w), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    g = grad.copy()
    g = g / (g.max() + 1e-8) if g.max() > 0 else g
    elev = 1.0 - g
    img3 = cv2.cvtColor(_safe_uint8(elev), cv2.COLOR_GRAY2BGR)

    mk = markers.astype(np.int32)
    mk[~fg] = 0
    cv2.watershed(img3, mk)

    inst = mk.copy()
    inst[inst < 0] = 0
    inst = inst.astype(np.int32)

    if inst.max() == 0:
        h, w = inst.shape
        return np.zeros((h, w), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    inst = safe_remap_label(inst)

    if min_inst_area > 1:
        new_inst = np.zeros_like(inst, dtype=np.int32)
        new_id = 0
        for iid in np.unique(inst):
            if iid == 0:
                continue
            area = int((inst == iid).sum())
            if area < min_inst_area:
                continue
            new_id += 1
            new_inst[inst == iid] = new_id
        inst = new_inst

    if inst.max() == 0:
        h, w = inst.shape
        return np.zeros((h, w), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    ids = [x for x in np.unique(inst) if x != 0]
    inst_type = np.zeros((len(ids),), dtype=np.int32)
    inst_prob = np.zeros((len(ids),), dtype=np.float32)

    if tp_map_or_predcls is None:
        inst_prob[:] = 1.0
    else:
        cls_map = tp_map_or_predcls.astype(np.int32, copy=False)
        for j, iid in enumerate(ids):
            vals = cls_map[inst == iid]
            if vals.size == 0:
                inst_type[j] = 0
                inst_prob[j] = 0.0
                continue
            uniq, cnt = np.unique(vals, return_counts=True)
            if cnt.size == 0:
                inst_type[j] = 0
                inst_prob[j] = 0.0
                continue
            k = int(uniq[int(np.argmax(cnt))])
            inst_type[j] = k
            inst_prob[j] = float(np.max(cnt) / (vals.size + 1e-8))

    return inst, inst_type, inst_prob


# =========================================================
# Trainer
# =========================================================
class AblationTrainer:
    def __init__(
        self,
        fusion_method="GIT",
        nr_types=3,
        train_workers=2,
        valid_workers=2,
        test_workers=2,
        use_git=None,
        use_og=True,
        train_dir=None,
        valid_dir=None,
        test_dir=None,
        data_root=None,
        valid_ratio=0.2,
    ):
        self.fusion_method = fusion_method
        self.use_git = (fusion_method != "NoFusion") if use_git is None else bool(use_git)
        self.use_og = bool(use_og)
        self.nr_types = nr_types
        self.train_workers = int(train_workers)
        self.valid_workers = int(valid_workers)
        self.test_workers = int(test_workers)
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.data_root = data_root
        self.valid_ratio = float(valid_ratio)
        self.experiment_tag = ablation_tag(self.use_git, self.use_og)
        self.nr_gpus = torch.cuda.device_count()
        print(f"检测到GPU数量: {self.nr_gpus}")

    def get_dataloader(self, batch_size, run_mode, target_gen, nr_procs=0):
        file_list = []
        if run_mode == "train":
            custom_dir = self.train_dir
            data_dir_list = ["/home/LCZ/datasets/train"]
        elif run_mode == "valid":
            custom_dir = self.valid_dir
            data_dir_list = ["/home/LCZ/datasets/valid"]
        elif run_mode == "test":
            custom_dir = self.test_dir
            data_dir_list = ["/home/LCZ/datasets/test"]
        else:
            raise ValueError(f"Unknown run mode: {run_mode}")

        if custom_dir is not None:
            file_list.extend(glob.glob(os.path.join(custom_dir, "*.npy")))
        elif self.data_root is not None and run_mode in ("train", "valid"):
            all_files = sorted(glob.glob(os.path.join(self.data_root, "*.npy")))
            if len(all_files) <= 1:
                file_list = all_files
            else:
                split_index = int(round(len(all_files) * (1.0 - self.valid_ratio)))
                split_index = max(1, min(split_index, len(all_files) - 1))
                file_list = all_files[:split_index] if run_mode == "train" else all_files[split_index:]
        else:
            for dir_path in data_dir_list:
                file_list.extend(glob.glob(f"{dir_path}/*.npy"))
        file_list.sort()
        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npy found for `{run_mode}`. Please set the corresponding directory.")
        print(f"数据集 {run_mode}: {len(file_list)}")

        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=True,
            setup_augmentor=(nr_procs == 0),
            target_gen=(target_gen, {}),
            input_shape=[384, 384],
            mask_shape=[384, 384],
        )

        # ✅ 线程/资源安全：减少 worker 并关闭持久化
        dataloader = DataLoader(
            input_dataset,
            num_workers=int(nr_procs),
            batch_size=batch_size,
            shuffle=(run_mode == "train"),
            drop_last=(run_mode == "train"),
            worker_init_fn=worker_init_fn if int(nr_procs) > 0 else None,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2 if int(nr_procs) > 0 else None,
        )
        return dataloader

    def debug_batch(self, dataloader, n_batches=1, title="DATASET"):
        print("\n" + "=" * 90)
        print(f"[DEBUG] {title} batch inspection")
        print("=" * 90)

        for b, batch_data in enumerate(dataloader):
            print(f"\n--- Batch {b} ---")
            print("BATCH KEYS:", list(batch_data.keys()))
            for k, v in batch_data.items():
                if torch.is_tensor(v):
                    x = v
                    shape = tuple(x.shape)
                    dtype = x.dtype
                    flat = x.detach().flatten()
                    if flat.numel() > 0:
                        sample = flat[: min(200000, flat.numel())].float().cpu()
                        mn = float(sample.min().item())
                        mx = float(sample.max().item())
                        print(f"  {k:<8} shape={shape} dtype={dtype} range=[{mn:.3f},{mx:.3f}]")
            if b + 1 >= n_batches:
                break
        print("=" * 90 + "\n")

    def debug_model_output(self, model, dataloader, n_batches=1, title="MODEL OUTPUT"):
        print("\n" + "=" * 90)
        print(f"[DEBUG] {title} inspection")
        print("=" * 90)

        model.eval()
        with torch.no_grad():
            for b, batch_data in enumerate(dataloader):
                img = batch_data["img"].to(TORCH_DEVICE)
                if img.dim() == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)

                out = model(img)

                print(f"\n--- Batch {b} ---")
                print("OUTPUT KEYS:", list(out.keys()))
                for k, v in out.items():
                    if torch.is_tensor(v):
                        x = v.detach()
                        shape = tuple(x.shape)
                        dtype = x.dtype
                        flat = x.flatten()
                        if flat.numel() > 0:
                            sample = flat[: min(200000, flat.numel())].float().cpu()
                            mn = float(sample.min().item())
                            mx = float(sample.max().item())
                            print(f"  {k:<8} shape={shape} dtype={dtype} range=[{mn:.3f},{mx:.3f}]")

                if "tp" in out and torch.is_tensor(out["tp"]) and out["tp"].dim() == 4:
                    tp = out["tp"]
                    print("\n[DEBUG] tp layout hint:")
                    print("  tp.shape =", tuple(tp.shape))
                    if tp.shape[1] <= 10:
                        print("  likely NCHW (C=dim1)")
                    elif tp.shape[-1] <= 10:
                        print("  likely NHWC (C=last)")

                if b + 1 >= n_batches:
                    break

        print("=" * 90 + "\n")

    def validate(self, model, dataloader, print_img_stats=False):
        model.eval()
        per_img_stats = []

        with torch.no_grad():
            for batch_data in dataloader:
                batch_gpu = {k: v.to(TORCH_DEVICE) for k, v in batch_data.items()}
                img = batch_gpu["img"]

                if img.dim() == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)

                out = model(img)

                N = img.shape[0]
                np_logits = out["np"]
                tp_logits = out["tp"]
                hv_pred = out["hv"]

                if np_logits.dim() == 4 and np_logits.shape[1] <= 10:
                    np_prob = torch.softmax(np_logits, dim=1)
                    pred_np = torch.argmax(np_prob, dim=1)
                elif np_logits.dim() == 4 and np_logits.shape[-1] <= 10:
                    np_prob = torch.softmax(np_logits, dim=-1)
                    pred_np = torch.argmax(np_prob, dim=-1)
                else:
                    pred_np = torch.zeros((N, img.shape[2], img.shape[3]), dtype=torch.long, device=img.device)

                # hv -> NHWC numpy
                if hv_pred.dim() == 4 and hv_pred.shape[1] == 2:
                    hv_pred_nhwc = hv_pred.permute(0, 2, 3, 1).contiguous()
                else:
                    hv_pred_nhwc = hv_pred
                hv_pred_np = hv_pred_nhwc.detach().cpu().numpy().astype(np.float32)

                # pred cls from tp
                if tp_logits.dim() == 4 and tp_logits.shape[1] <= 10:
                    prob = torch.softmax(tp_logits, dim=1)
                    pred_cls = torch.argmax(prob, dim=1)  # (N,H,W)
                elif tp_logits.dim() == 4 and tp_logits.shape[-1] <= 10:
                    prob = torch.softmax(tp_logits, dim=-1)
                    pred_cls = torch.argmax(prob, dim=-1)
                else:
                    pred_cls = torch.zeros((N, img.shape[2], img.shape[3]), dtype=torch.long, device="cpu")

                pred_cls_np = pred_cls.detach().cpu().numpy().astype(np.int32)
                pred_np_map = pred_np.detach().cpu().numpy().astype(np.uint8)

                gt_np = batch_gpu["np_map"]
                if gt_np.dim() == 4 and gt_np.shape[1] == 1:
                    gt_np = gt_np.squeeze(1)
                gt_np_np = (gt_np.detach().cpu().numpy() > 0).astype(np.uint8)

                gt_hv = batch_gpu["hv_map"]
                if gt_hv.dim() == 4 and gt_hv.shape[-1] == 2:
                    gt_hv_nhwc = gt_hv
                elif gt_hv.dim() == 4 and gt_hv.shape[1] == 2:
                    gt_hv_nhwc = gt_hv.permute(0, 2, 3, 1).contiguous()
                else:
                    gt_hv_nhwc = gt_hv
                gt_hv_np = gt_hv_nhwc.detach().cpu().numpy().astype(np.float32)

                gt_tp = batch_gpu["tp_map"]
                if gt_tp.dim() == 4 and gt_tp.shape[1] == 1:
                    gt_tp = gt_tp.squeeze(1)
                gt_tp_np = gt_tp.detach().cpu().numpy().astype(np.int32)

                for i in range(N):
                    gt_inst, _, _ = post_proc_func_from_nphv_tp(gt_np_np[i], gt_hv_np[i], gt_tp_np[i])
                    pr_inst, _, _ = post_proc_func_from_nphv_tp(pred_np_map[i], hv_pred_np[i], pred_cls_np[i])

                    gt_inst = safe_remap_label(gt_inst)
                    pr_inst = safe_remap_label(pr_inst)

                    dsc, aji, aji_plus, dq, sq, pq = safe_instance_metrics(gt_inst, pr_inst)
                    per_img_stats.append({"DSC": dsc, "AJI": aji, "AJI+": aji_plus, "DQ": dq, "SQ": sq, "PQ": pq})

        if len(per_img_stats) == 0:
            return {"DSC": 0.0, "AJI": 0.0, "AJI+": 0.0, "PQ": 0.0, "DQ": 0.0, "SQ": 0.0}

        keys = ["DSC", "AJI", "AJI+", "PQ", "DQ", "SQ"]
        avg = {k: float(np.mean([d[k] for d in per_img_stats])) for k in keys}

        if print_img_stats:
            print("\n[Per-image VALID stats] (first 30 shown)")
            for idx, d in enumerate(per_img_stats[:30]):
                print(
                    f"  img#{idx:04d}  DSC={d['DSC']:.4f} AJI={d['AJI']:.4f} AJI+={d['AJI+']:.4f} "
                    f"PQ={d['PQ']:.4f} (DQ={d['DQ']:.4f}, SQ={d['SQ']:.4f})"
                )

        return avg

    def evaluate_checkpoint(self, checkpoint_path, print_img_stats=False):
        model = create_ablation_model(
            self.fusion_method,
            nr_types=self.nr_types,
            use_git=self.use_git,
            use_og=self.use_og,
        ).to(TORCH_DEVICE)

        checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

        test_loader = self.get_dataloader(16, "test", gen_targets, nr_procs=self.test_workers)
        test_metrics = self.validate(model, test_loader, print_img_stats=print_img_stats)

        print("\n" + "=" * 60)
        print(f"Test checkpoint: {checkpoint_path}")
        print(
            f"Test - DSC: {test_metrics['DSC']:.4f}, AJI: {test_metrics['AJI']:.4f}, "
            f"AJI+: {test_metrics['AJI+']:.4f}, PQ: {test_metrics['PQ']:.4f}"
        )
        print(f"       DQ: {test_metrics['DQ']:.4f}, SQ: {test_metrics['SQ']:.4f}")
        print("=" * 60)
        return test_metrics

    def train(self, nr_epochs=50, debug_once=True, print_img_stats=False):
        print(f"\n{'='*60}")
        print(f"开始消融实验: {self.fusion_method}")
        print(f"{'='*60}\n")

        check_manual_seed(10)
        log_dir = f"logs/Ablation2x2_{self.experiment_tag}"
        rm_n_mkdir(log_dir)

        model = create_ablation_model(
            self.fusion_method,
            nr_types=self.nr_types,
            use_git=self.use_git,
            use_og=self.use_og,
        ).to(TORCH_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)

        loader_dict = {
            "train": self.get_dataloader(8, "train", gen_targets, nr_procs=self.train_workers),
            "valid": self.get_dataloader(16, "valid", gen_targets, nr_procs=self.valid_workers),
        }

        if debug_once:
            self.debug_batch(loader_dict["train"], n_batches=1, title="TRAIN DATASET")
            self.debug_batch(loader_dict["valid"], n_batches=1, title="VALID DATASET")
            self.debug_model_output(model, loader_dict["valid"], n_batches=1, title="MODEL OUTPUT (VALID)")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})\n")

        best_metric = -1.0
        best_pack = None

        for epoch in range(nr_epochs):
            print(f"Epoch {epoch+1}/{nr_epochs}")

            model.train()
            train_loss = 0.0
            n_steps = 0

            for batch_data in loader_dict["train"]:
                batch_gpu = {k: v.to(TORCH_DEVICE) for k, v in batch_data.items()}
                img = batch_gpu["img"]
                if img.dim() == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)

                optimizer.zero_grad()
                out = model(img)
                loss_dict = compute_loss(out, batch_gpu, use_og=self.use_og)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                n_steps += 1

            train_loss = train_loss / max(1, n_steps)
            print(f"  Train Loss: {train_loss:.4f}")

            val_metrics = self.validate(model, loader_dict["valid"], print_img_stats=print_img_stats)
            print(
                f"  Val - DSC: {val_metrics['DSC']:.4f}, AJI: {val_metrics['AJI']:.4f}, AJI+: {val_metrics['AJI+']:.4f}, "
                f"PQ: {val_metrics['PQ']:.4f}"
            )
            print(f"        DQ: {val_metrics['DQ']:.4f}, SQ: {val_metrics['SQ']:.4f}")

            pick_metric = val_metrics["PQ"]
            if pick_metric > best_metric:
                best_metric = pick_metric
                best_pack = val_metrics
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "fusion_method": self.fusion_method,
                        "use_git": self.use_git,
                        "use_og": self.use_og,
                    },
                    f"{log_dir}/best_model.tar",
                )
                print("  [保存最佳模型] (by PQ)")

            scheduler.step()

        print(f"\n{'='*60}")
        print(f"消融实验 {self.fusion_method} 完成!")
        if best_pack is not None:
            print(
                f"最佳(PQ)={best_metric:.4f}  "
                f"DSC={best_pack['DSC']:.4f} AJI={best_pack['AJI']:.4f} AJI+={best_pack['AJI+']:.4f} "
                f"DQ={best_pack['DQ']:.4f} SQ={best_pack['SQ']:.4f}"
            )
        else:
            print("未获得有效验证结果")
        print(f"{'='*60}\n")

        return best_metric


def run_all_ablation_experiments(epochs=10, print_img_stats=False, train_workers=2, valid_workers=2):
    print("\n" + "=" * 60)
    print("消融实验: GIT替代方法对比")
    print("=" * 60 + "\n")

    methods = ["GIT", "Concat", "SE", "CBAM", "NoFusion", "Identity", "WeightSum"]
    results = {}

    for method in methods:
        try:
            trainer = AblationTrainer(
                fusion_method=method,
                nr_types=3,
                train_workers=train_workers,
                valid_workers=valid_workers,
            )
            best_score = trainer.train(
                nr_epochs=epochs,
                debug_once=(method == methods[0]),
                print_img_stats=print_img_stats,
            )
            results[method] = best_score
        except Exception as e:
            print(f"实验 {method} 失败: {e}")
            results[method] = 0.0

    print("\n" + "=" * 60)
    print("消融实验结果对比（按 PQ best）")
    print("=" * 60)
    print(f"{'方法':<15} {'BestPQ':<10} {'相对于GIT':<10}")
    print("-" * 40)

    git_score = results.get("GIT", 0.0)
    for method, sc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        rel_change = f"{sc - git_score:+.4f}" if method != "GIT" else "基线"
        print(f"{method:<15} {sc:<10.4f} {rel_change:<10}")

    print("=" * 60)
    return results


def run_all_2x2_ablation_experiments(
    epochs=10,
    print_img_stats=False,
    train_workers=2,
    valid_workers=2,
    train_dir=None,
    valid_dir=None,
    data_root=None,
    valid_ratio=0.2,
):
    print("\n" + "=" * 60)
    print("2x2 ablation: GIT and OG")
    print("=" * 60 + "\n")

    settings = [
        {"use_git": False, "use_og": False},
        {"use_git": False, "use_og": True},
        {"use_git": True, "use_og": False},
        {"use_git": True, "use_og": True},
    ]
    results = {}

    for index, setting in enumerate(settings):
        tag = ablation_tag(setting["use_git"], setting["use_og"])
        try:
            trainer = AblationTrainer(
                fusion_method="GIT" if setting["use_git"] else "NoFusion",
                nr_types=3,
                train_workers=train_workers,
                valid_workers=valid_workers,
                use_git=setting["use_git"],
                use_og=setting["use_og"],
                train_dir=train_dir,
                valid_dir=valid_dir,
                data_root=data_root,
                valid_ratio=valid_ratio,
            )
            best_score = trainer.train(
                nr_epochs=epochs,
                debug_once=(index == 0),
                print_img_stats=print_img_stats,
            )
            results[tag] = best_score
        except Exception as e:
            print(f"Experiment {tag} failed: {e}")
            results[tag] = 0.0

    print("\n" + "=" * 60)
    print("2x2 ablation summary (best PQ)")
    print("=" * 60)
    print(f"{'Combination':<15} {'BestPQ':<10} {'vs GIT1_OG1':<12}")
    print("-" * 44)

    full_score = results.get("GIT1_OG1", 0.0)
    for combination, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        rel_change = f"{score - full_score:+.4f}" if combination != "GIT1_OG1" else "baseline"
        print(f"{combination:<15} {score:<10.4f} {rel_change:<12}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验: GIT替代方法（实例指标版，线程安全）")
    parser.add_argument("--method", type=str, default="GIT",
                        choices=["GIT", "Concat", "SE", "CBAM", "NoFusion", "Identity", "WeightSum",
                                 "SelfAttention", "Res2NetStyle"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no_debug", action="store_true")
    parser.add_argument("--print_img_stats", action="store_true")
    parser.add_argument("--use_git", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_og", type=int, choices=[0, 1], default=1)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--valid_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=str, default=None)

    # ✅ 新增：控制 DataLoader worker
    parser.add_argument("--train_workers", type=int, default=2, help="train dataloader num_workers")
    parser.add_argument("--valid_workers", type=int, default=2, help="valid dataloader num_workers")
    parser.add_argument("--test_workers", type=int, default=2, help="test dataloader num_workers")

    args = parser.parse_args()

    if args.all:
        run_all_2x2_ablation_experiments(
            epochs=args.epochs if args.epochs > 0 else 10,
            print_img_stats=args.print_img_stats,
            train_workers=args.train_workers,
            valid_workers=args.valid_workers,
            train_dir=args.train_dir,
            valid_dir=args.valid_dir,
            data_root=args.data_root,
            valid_ratio=args.valid_ratio,
        )
    else:
        trainer = AblationTrainer(
            fusion_method=args.method,
            nr_types=3,
            train_workers=args.train_workers,
            valid_workers=args.valid_workers,
            test_workers=args.test_workers,
            use_git=bool(args.use_git),
            use_og=bool(args.use_og),
            train_dir=args.train_dir,
            valid_dir=args.valid_dir,
            test_dir=args.test_dir,
            data_root=args.data_root,
            valid_ratio=args.valid_ratio,
        )
        if args.test_only:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required when using --test_only")
            if args.test_dir is None:
                raise ValueError("--test_dir is required when using --test_only")
            trainer.evaluate_checkpoint(
                checkpoint_path=args.checkpoint,
                print_img_stats=args.print_img_stats,
            )
        else:
            trainer.train(
                nr_epochs=args.epochs,
                debug_once=(not args.no_debug),
                print_img_stats=args.print_img_stats,
            )
