#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# NoisyRollout's training scripts (training_scripts/*.sh and config.yaml of the official code).
# The first argument picks the training set: geo3k (Geometry3K) or k12 (xyliu6/k12-freeform train).
launch_noisyrollout() {
    pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
    local dataset=$1
    shift
    local train_files val_files
    case "$dataset" in
        geo3k)
            train_files="$DATA_ROOT/geometry3k/train.parquet"
            val_files="$DATA_ROOT/geometry3k/test.parquet"
            ;;
        k12)
            train_files="$DATA_ROOT/k12/train.parquet"
            val_files="$DATA_ROOT/k12/test.parquet"
            ;;
        *)
            echo "the dataset must be geo3k or k12, got '$dataset'" >&2
            return 1
            ;;
    esac
    METHOD_COMMON_ARGS=(
        "data.train_files=$train_files"
        "data.val_files=$val_files"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=null"
        "data.system_prompt=$ROOT_DIR/examples/system_prompt/noisyrollout.txt"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=262144"
        "data.max_pixels=1000000"
        "data.max_prompt_length=2048"
        "data.max_response_length=2048"
        "data.rollout_batch_size=512"
        "data.mini_rollout_batch_size=null"
        "data.val_batch_size=1024"
        "data.seed=1"
        "algorithm.adv_estimator=grpo"
        "algorithm.use_kl_loss=false"
        "algorithm.disable_kl=true"
        "algorithm.invariant_entropy_coef=-0.001"
        "algorithm.entropy_loss_type=full"
        "algorithm.log_entropy=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=true"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.weight_decay=1e-2"
        "worker.actor.global_batch_size=128"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.micro_batch_size_per_device_for_update=2"
        "worker.actor.micro_batch_size_per_device_for_experience=4"
        "worker.actor.offload.offload_params=true"
        "worker.actor.offload.offload_optimizer=true"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=4"
        "worker.rollout.gpu_memory_utilization=0.35"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=0.0"
        "worker.rollout.val_override_config.top_p=1.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/math.py:compute_score"
        "trainer.project_name=NoisyRollout-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.total_epochs=15"
        "trainer.val_freq=5"
        "trainer.save_freq=20"
        "trainer.save_limit=1"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
    )
    launch_training "$@"
}

# NoisyRollout's annealed noise (Eq. 3): alpha_0, lambda, and gamma as a fraction of t_max.
noisyrollout_args() {
    local alpha0=$1 lambda=$2 gamma_fraction=$3
    printf '%s\n' \
        "algorithm.rollout_image_transform=vp_diffusion" \
        "algorithm.rollout_image_transform_kwargs={\"noise_t_init\":$alpha0,\"noise_gamma\":$lambda,\"noise_t_mid\":$gamma_fraction,\"noise_t_max\":1000,\"pixel_rounding\":\"floor\"}"
}
