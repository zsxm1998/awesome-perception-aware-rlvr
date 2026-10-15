#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "vapo"
# ViRL39K row for row with VAPO's visual claims (the 2,289 multi-image problems have none and get the GRPO reward).
# K = 20 claim probes per correct response, beta = 1.5; reward 0.8 accuracy + 0.1 format + 0.1 perception, as the
# released VAPO reward (the other arms: 0.9 accuracy + 0.1 format). The probe question ends without the released
# trailing space, after which Qwen3-VL-4B answers the probes with "1"/"0" (see the VAPO README).
ALGO_ARGS=(
    "data.train_files=$DATA_ROOT/virl39k_claims/train.parquet"
    "algorithm.claim_probe_count=20"
    "algorithm.claim_probe_late_emphasis=1.5"
    'algorithm.claim_probe_question="\n<anchor>{claim} Is this claim correct? Answer (Yes/No):"'
    "worker.reward.reward_function_kwargs={\"format_weight\":0.1,\"perception_weight\":0.1}"
    "worker.rollout.mm_processor_cache_gb=4"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
