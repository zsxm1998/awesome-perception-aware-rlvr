#!/usr/bin/env bash
# One-click evaluation data preparation (writes data/eval/<benchmark>/). Examples:
#   bash scripts/prepare_eval_data.sh papo              # every benchmark of a suite
#   bash scripts/prepare_eval_data.sh geo3k pope vstar  # individual benchmarks
#   bash scripts/prepare_eval_data.sh all               # every non-optional benchmark
#   bash scripts/prepare_eval_data.sh --list            # benchmarks, sources and suites
# The data root defaults to $EVAL_DATA_ROOT, else $DATA_ROOT/eval, else <repo>/data/eval
# (or pass --data-root DIR). Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is
# slow or blocked. Re-running is safe: prepared benchmarks are skipped (use --force to redo).
set -eo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
exec "$PYTHON" "$ROOT_DIR/eval/prepare/prepare.py" "$@"
