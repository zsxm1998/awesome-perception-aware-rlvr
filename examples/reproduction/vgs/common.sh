#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# VGS (arXiv 2606.00564) setting: the authors' split of Vision-SR1-47K (45,246 train / 2,382 validation),
# the VGS system prompt (<reason></reason> + \boxed{}) with the problem as the user message, a 0/1
# correctness reward, AdamW with lr 1e-6 constant and weight decay 1e-2, the vision tower trained, 8 GPUs.
# Three stages share it: the GRPO teacher (Table 3), the distilled students (Table 4) and the GRPO
# combinations of Table 2.
vgs_common_args() {
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/vision_sr1/train.parquet"
        "data.val_files=$DATA_ROOT/vision_sr1_val/val.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=null"
        "data.system_prompt=$ROOT_DIR/examples/system_prompt/vgs.txt"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=262144"
        "data.max_pixels=4194304"
        "data.max_response_length=2048"
        "data.val_batch_size=512"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.weight_decay=1e-2"
        "worker.actor.optim.strategy=adamw"
        "worker.actor.max_grad_norm=1.0"
        "worker.actor.global_batch_size=128"
        # Dynamic batching packs up to micro-batch size x (max prompt + max response) tokens per micro-batch;
        # with the long prompt limits of VGS, micro-batches of 2 / 4 hold 30k-37k / 59k-74k tokens.
        "worker.actor.micro_batch_size_per_device_for_update=2"
        "worker.actor.micro_batch_size_per_device_for_experience=4"
        "worker.rollout.temperature=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.5"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=1.0"
        "worker.rollout.val_override_config.top_p=1.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/vgs.py:compute_score"
        "trainer.project_name=VGS-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.save_limit=1"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
    )
}

# Stage 1 (Table 3): the teacher, Qwen3-VL-8B-Instruct trained with GRPO for 2 epochs: 512 prompts x 8 rollouts
# per step, update batch 128 prompts (4 updates per step), top-p 0.99, no KL, clip 0.2 / 0.2.
launch_vgs_teacher_grpo() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
    vgs_common_args
    METHOD_COMMON_ARGS+=(
        "data.max_prompt_length=12800"
        "data.rollout_batch_size=512"
        "data.mini_rollout_batch_size=128"
        "algorithm.adv_estimator=grpo"
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.rollout.n=8"
        "worker.rollout.top_p=0.99"
        "worker.rollout.max_num_batched_tokens=14848"
        "trainer.total_epochs=2"
        "trainer.val_freq=44"
        "trainer.save_freq=44"
    )
    launch_training "$@"
}

# The finalized teacher of stage 1 (scripts/finalize_run.py turns its last step, global_step_176 with the
# full training split, into a Hugging Face model directory).
default_vgs_teacher() {
    pr_default TEACHER_PATH "$ROOT_DIR/checkpoints/VGS-Reproduce/qwen3_vl_8b_grpo_teacher/global_step_176/actor"
}

# Stage 2 (Table 4): a student distilled from the teacher: 128 prompts x 4 rollouts and one update per step,
# 1 epoch (353 steps), top-p 1.0, max prompt 16,384, the full-vocabulary reverse KL averaged over the tokens
# (TRL's default), no reference KL; evaluated every 40 steps.
launch_vgs_student() {
    default_vgs_teacher
    vgs_common_args
    METHOD_COMMON_ARGS+=(
        "data.max_prompt_length=16384"
        "data.rollout_batch_size=128"
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "worker.teacher.source=model"
        "worker.teacher.model.model_path=$TEACHER_PATH"
        "algorithm.distill_loss_coef=1.0"
        "algorithm.policy_loss_coef=0.0"
        "algorithm.distill_divergence=reverse_kl"
        "worker.actor.loss_avg_mode=token"
        "worker.rollout.n=4"
        "worker.rollout.top_p=1.0"
        "worker.rollout.max_num_batched_tokens=18432"
        "trainer.total_epochs=1"
        "trainer.val_freq=40"
        "trainer.save_freq=40"
    )
    launch_training "$@"
}

# Table 2: GRPO and its combinations with distillation, (1 - alpha) GRPO + alpha distillation with alpha 0.3
# (Eq. 22, without eta), on the 2B student: 512 prompts x 8 rollouts per step, update batch 128 prompts,
# top-p 1.0 (the distillation needs untruncated sampling), no KL, clip 0.2 / 0.2, 1 epoch (88 steps).
launch_vgs_grpo_combination() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
    vgs_common_args
    METHOD_COMMON_ARGS+=(
        "data.max_prompt_length=16384"
        "data.rollout_batch_size=512"
        "data.mini_rollout_batch_size=128"
        "algorithm.adv_estimator=grpo"
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.loss_avg_mode=token"
        "worker.rollout.n=8"
        "worker.rollout.top_p=1.0"
        "worker.rollout.max_num_batched_tokens=18432"
        "trainer.total_epochs=1"
        "trainer.val_freq=10"
        "trainer.save_freq=10"
    )
    launch_training "$@"
}
