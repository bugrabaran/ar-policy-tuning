#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

GPT_CKPT="${GPT_CKPT:-$REPO_DIR/pretrained_models/c2i_L_256.pt}"
VQ_CKPT="${VQ_CKPT:-$REPO_DIR/pretrained_models/vq_ds16_c2i.pt}"
FID_STATS_NPZ="${FID_STATS_NPZ:-$REPO_DIR/imagenet_train_incv3_299_original.npz}"
FID_INIT_GEN_NPZ="${FID_INIT_GEN_NPZ:-$REPO_DIR/pretrained_gen_stats.npz}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/grpo_runs/run1}"
SAMPLES_DIR="${SAMPLES_DIR:-$OUT_DIR/samples}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-12}"
MAX_ITERATIONS="${MAX_ITERATIONS:-600}"
NUM_GENERATIONS="${NUM_GENERATIONS:-12}"
INNER_EPOCHS="${INNER_EPOCHS:-2}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SAVE_SAMPLES_EVERY="${SAVE_SAMPLES_EVERY:-100}"
NUM_SAMPLE_IMAGES="${NUM_SAMPLE_IMAGES:-8}"

W_CLIP="${W_CLIP:-1.0}"
W_HPS="${W_HPS:-1.0}"
W_FID="${W_FID:-1.0}"
FID_BETA="${FID_BETA:-0.5}"

ENT_COEF_BASE="${ENT_COEF_BASE:-2.2e-3}"
ENT_MIN="${ENT_MIN:-7e-5}"
ENT_MAX="${ENT_MAX:-4e-3}"
ENT_TARGET_FRAC="${ENT_TARGET_FRAC:-0.78}"
ENT_DEADBAND="${ENT_DEADBAND:-0.015}"
ENT_K="${ENT_K:-3.0}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
DECAY_START_FRAC="${DECAY_START_FRAC:-0.85}"

for required_file in "$GPT_CKPT" "$VQ_CKPT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

TRAIN_ARGS=(
  --gpt-model GPT-L
  --gpt-ckpt "$GPT_CKPT"
  --gpt-type c2i
  --cls-token-num 1
  --precision bf16
  --vq-model VQ-16
  --vq-ckpt "$VQ_CKPT"
  --codebook-size 16384
  --codebook-embed-dim 8
  --image-size 256
  --downsample-size 16
  --cfg-scale 1.5
  --per-proc-batch-size "$PER_PROC_BATCH_SIZE"
  --global-seed 0
  --top-k 0
  --temperature 1.0
  --top-p 1.0
  --G "$NUM_GENERATIONS"
  --K "$INNER_EPOCHS"
  --eps 0.2
  --beta 3.0
  --ema-beta 0.999
  --eos-id -1
  --lr 1e-5
  --ent-coef-base "$ENT_COEF_BASE"
  --ent-min "$ENT_MIN"
  --ent-max "$ENT_MAX"
  --ent-target-frac "$ENT_TARGET_FRAC"
  --ent-deadband "$ENT_DEADBAND"
  --ent-k "$ENT_K"
  --warmup-frac "$WARMUP_FRAC"
  --decay-start-frac "$DECAY_START_FRAC"
  --w-clip "$W_CLIP"
  --w-hps "$W_HPS"
  --w-fid "$W_FID"
  --fid-beta "$FID_BETA"
  --fid-no-debias
  --max-iterations "$MAX_ITERATIONS"
  --output-dir "$OUT_DIR"
  --save-every "$SAVE_EVERY"
  --resume "$OUT_DIR"
  --save-samples-every "$SAVE_SAMPLES_EVERY"
  --samples-dir "$SAMPLES_DIR"
  --num-sample-images "$NUM_SAMPLE_IMAGES"
)

if [[ "$W_FID" != "0" && "$W_FID" != "0.0" ]]; then
  if [[ ! -f "$FID_STATS_NPZ" ]]; then
    echo "FID statistics not found: $FID_STATS_NPZ" >&2
    echo "Set W_FID=0 to run without the distribution-level reward." >&2
    exit 1
  fi
  TRAIN_ARGS+=(--fid-stats-npz "$FID_STATS_NPZ")
  if [[ -n "$FID_INIT_GEN_NPZ" ]]; then
    if [[ ! -f "$FID_INIT_GEN_NPZ" ]]; then
      echo "Initial generated statistics not found: $FID_INIT_GEN_NPZ" >&2
      exit 1
    fi
    TRAIN_ARGS+=(--fid-init-gen-npz "$FID_INIT_GEN_NPZ")
  fi
fi

mkdir -p "$OUT_DIR" "$SAMPLES_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.cache/huggingface}"

if [[ "$NNODES" == "1" ]]; then
  LAUNCH_ARGS=(--standalone --nproc_per_node="$NPROC_PER_NODE")
else
  LAUNCH_ARGS=(
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
  )
fi

python -m torch.distributed.run "${LAUNCH_ARGS[@]}" grpo.py "${TRAIN_ARGS[@]}" "$@"
