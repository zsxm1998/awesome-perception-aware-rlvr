#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "cgpo"
ALGO_ARGS=(
    "data.format_prompt=$ROOT_DIR/examples/format_prompt/xml_grounded_reasoning_v2.jinja"
    "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/xml_grounded_reasoning.py:compute_score"
    "worker.reward.reward_function_kwargs={\"format_weight\":0.1}"
    "algorithm.corrupt_image=cgpo_flat"
    "algorithm.corrupt_image_kwargs={\"fill_type\":\"local_mean\"}"
    "algorithm.corrupt_image_position=response"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.top_perception_quantile=0.3"
    "algorithm.perception_thr_granularity=micro_batch"
    "algorithm.top_entropy_quantile=0.3"
    "algorithm.entropy_thr_granularity=micro_batch"
    "algorithm.advantage_scaling_method=cgpo"
    "algorithm.cgpo_response_scaling_coef=0.1"
    "algorithm.use_grounding_consistency_reward=true"
    "algorithm.grounding_consistency_reward_weight=0.1"
    "algorithm.grounding_consistency_aggregation=response"  # the paper's R_gc: mean over the response's regions
    "worker.rollout.mm_processor_cache_gb=4"  # GCR detects several regions per image: process each image once
    "algorithm.include_region_tokens_in_perception_mask=true"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
