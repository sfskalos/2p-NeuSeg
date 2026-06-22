#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/LCZ/mynetZXIAO"
IMAGE_DIR="/home/LCZ/datasets/images"
GT_DIR="/home/LCZ/datasets/labels/mat"
OUTPUT_ROOT="/home/LCZ/datasets/ablation_eval"
GPU_ID="0"
MODEL_MODE="MyNet"

mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local tag="$1"
  local fusion_method="$2"
  local use_git="$3"
  local use_og="$4"
  local ckpt="${ROOT_DIR}/logs/${tag}/best_model.tar"
  local out_dir="${OUTPUT_ROOT}/${tag}"

  echo "============================================================"
  echo "Running inference for ${tag}"
  echo "Checkpoint: ${ckpt}"
  echo "Output: ${out_dir}"
  echo "============================================================"

  mkdir -p "${out_dir}"

  python "${ROOT_DIR}/run_infer.py" \
    --gpu="${GPU_ID}" \
    --model_path "${ckpt}" \
    --model_mode="${MODEL_MODE}" \
    --fusion_method="${fusion_method}" \
    --use_git="${use_git}" \
    --use_og="${use_og}" \
    tile \
    --input_dir="${IMAGE_DIR}" \
    --output_dir="${out_dir}"

  echo "Instance metrics for ${tag}" | tee "${out_dir}/instance_metrics.txt"
  python "${ROOT_DIR}/compute_stats.py" \
    --mode instance \
    --pred_dir "${out_dir}/mat/" \
    --true_dir "${GT_DIR}/" | tee -a "${out_dir}/instance_metrics.txt"

  echo "Type metrics for ${tag}" | tee "${out_dir}/type_metrics.txt"
  python "${ROOT_DIR}/compute_stats.py" \
    --mode type \
    --pred_dir "${out_dir}/mat/" \
    --true_dir "${GT_DIR}/" | tee -a "${out_dir}/type_metrics.txt"
}

run_one "Ablation2x2_GIT0_OG0" "NoFusion" "0" "0"
run_one "Ablation2x2_GIT0_OG1" "NoFusion" "0" "1"
run_one "Ablation2x2_GIT1_OG0" "GIT" "1" "0"
run_one "Ablation2x2_GIT1_OG1" "GIT" "1" "1"

echo "All ablation evaluations finished."
