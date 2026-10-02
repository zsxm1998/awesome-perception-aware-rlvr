#!/usr/bin/env bash
# Thin shim so every method-level common.sh can `source "$THIS_DIR/../common.sh"`.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/launcher.sh"
