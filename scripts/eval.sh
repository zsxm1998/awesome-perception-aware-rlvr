#!/usr/bin/env bash
# One-click evaluation of a model or an EasyR1 training checkpoint.
#
#   bash scripts/eval.sh <model_or_ckpt> [--suite papo] [--all-steps] [runner options]
#
# <model_or_ckpt>: a Hugging Face id (Qwen/Qwen2.5-VL-3B-Instruct), a merged HF directory,
# .../global_step_N/actor, .../global_step_N, or a run checkpoint root (the latest step is
# evaluated; --all-steps evaluates every step). FSDP shards are merged into
# <actor>/huggingface with scripts/model_merger.py (shards are kept); runs finalized by
# scripts/finalize_run.py hold the merged weights in <actor> itself.
# Results: eval/results/<run_name>/<step>/ (summary.csv, metrics/, predictions/) and one row
# per run in eval/results/summary.csv. All visible GPUs are used (set CUDA_VISIBLE_DEVICES
# or pass --gpus 0,1). Any other option goes to eval/run_all_benchmarks.py, e.g.
#   bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct --suite papo                    # base model
#   bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo   # latest step
#   bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo --all-steps
#   bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo/global_step_200 --suite papo
#   bash scripts/eval.sh checkpoints/CGPO-Reproduce/qwen3_vl_8b_cgpo --suite cgpo
#   bash scripts/eval.sh Qwen/Qwen2.5-VL-3B-Instruct --benchmarks geo3k,pope --limit 32  # smoke test
# Prepare the data first: bash scripts/prepare_eval_data.sh <suite or benchmarks>.
set -eo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
exec "$PYTHON" "$ROOT_DIR/eval/evaluate.py" "$@"
