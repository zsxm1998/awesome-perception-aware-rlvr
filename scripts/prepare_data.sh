#!/usr/bin/env bash
# One-click training data preparation. Examples:
#   bash scripts/prepare_data.sh papo            # data needed by reproduction/papo
#   bash scripts/prepare_data.sh comparison      # data needed by comparison/
#   bash scripts/prepare_data.sh all             # everything
#   bash scripts/prepare_data.sh --list          # show datasets and method groups
# Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is slow or blocked.
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 "$ROOT_DIR/scripts/data/prepare_train_data.py" "$@"
