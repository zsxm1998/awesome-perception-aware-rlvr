#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "noisyrollout"
# 4 of the 8 rollouts per prompt from noised images, with the strength of NoisyRollout's MMK12-7B script
# (alpha_0 = 450, lambda = 60) and the annealing midpoint at 10% of the steps (gamma = 0.1 t_max, about step 20)
# instead of its t_max / 3: see "NoisyRollout's noise schedule" in this directory's README.
ALGO_ARGS=(
    "algorithm.rollout_image_transform=vp_diffusion"
    "algorithm.rollout_image_transform_kwargs={\"noise_t_init\":450,\"noise_gamma\":60,\"noise_t_mid\":0.1,\"noise_t_max\":1000,\"pixel_rounding\":\"floor\"}"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
