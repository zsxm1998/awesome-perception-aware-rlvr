#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# VA-OPD (arXiv 2605.21924) setting: Qwen3-VL-2B-Instruct student, 16 prompts x 4 rollouts per step and one
# update per step, 5 epochs, rollouts at T=1.0 / top-p 1.0. VA_OPD_DATA selects the training set (geo3k:
# Geometry3K train, about 655 steps; virl39k: ViRL39K, about 12,150 steps). Every checkpoint is kept (weights
# only) for the paper's best-checkpoint selection, see README.md.
va_opd_common_args() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
    pr_default VA_OPD_DATA "geo3k"
    local train_files val_files save_freq
    case "$VA_OPD_DATA" in
        geo3k)
            train_files="$DATA_ROOT/geometry3k/train.parquet"
            val_files="$DATA_ROOT/geometry3k/test.parquet"
            save_freq=50
            ;;
        virl39k)
            train_files="$DATA_ROOT/virl39k/train.parquet"
            val_files="$DATA_ROOT/mmk12/test.parquet"
            save_freq=500
            ;;
        *)
            echo "[va_opd] VA_OPD_DATA must be geo3k or virl39k, got '$VA_OPD_DATA'" >&2
            exit 1
            ;;
    esac
    METHOD_COMMON_ARGS=(
        "data.train_files=$train_files"
        "data.val_files=$val_files"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=$ROOT_DIR/examples/format_prompt/math.jinja"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=262144"
        "data.max_pixels=4194304"
        "data.max_prompt_length=2048"
        "data.max_response_length=2048"
        "data.rollout_batch_size=16"
        "data.val_batch_size=512"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.weight_decay=1e-2"
        "worker.actor.optim.strategy=adamw"
        "worker.actor.max_grad_norm=1.0"
        "worker.actor.global_batch_size=16"
        "worker.actor.micro_batch_size_per_device_for_update=4"
        "worker.actor.micro_batch_size_per_device_for_experience=8"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.rollout.n=4"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.5"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=8"
        "worker.rollout.val_override_config.temperature=1.0"
        "worker.rollout.val_override_config.top_p=1.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/math.py:compute_score"
        "trainer.project_name=VA-OPD-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.total_epochs=5"
        "trainer.val_freq=$save_freq"
        "trainer.save_freq=$save_freq"
        "trainer.save_limit=-1"
        "trainer.save_model_only=true"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
    )
}

# Standard OPD and VA-OPD: the full-vocabulary reverse KL to the teacher as the only loss, no reference KL.
launch_va_opd_distillation() {
    pr_default TEACHER_PATH "Qwen/Qwen3-VL-8B-Instruct"
    va_opd_common_args
    METHOD_COMMON_ARGS+=(
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "worker.teacher.source=model"
        "worker.teacher.model.model_path=$TEACHER_PATH"
        "algorithm.distill_loss_coef=1.0"
        "algorithm.policy_loss_coef=0.0"
        "algorithm.distill_divergence=reverse_kl"
    )
    launch_training "$@"
}

# The RL baselines of Table 1 on the same data, rollouts and optimizer (the paper gives no RL settings):
# GRPO with this repository's default low-variance KL loss to the reference model.
launch_va_opd_rl() {
    va_opd_common_args
    METHOD_COMMON_ARGS+=(
        "algorithm.adv_estimator=grpo"
        "algorithm.use_kl_loss=true"
        "algorithm.kl_penalty=low_var_kl"
        "algorithm.kl_coef=0.01"
        "worker.actor.clip_ratio_high=0.2"
    )
    launch_training "$@"
}
