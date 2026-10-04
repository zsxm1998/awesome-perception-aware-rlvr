#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# Shared recipe of the on-policy distillation (OPD) comparison: a Qwen3-VL-2B student distilled from
# Qwen3-VL-8B-Instruct on the data, prompt, optimizer and lengths of the RL comparison
# (../qwen3_vl_4b/common.sh). Each update uses one fresh rollout batch (128 prompts x 8), the same
# number of samples per update and of updates as the RL recipe's 384 x 8 with three updates.
# There is no reference KL and no entropy term; the task reward is computed for logging only.
launch_opd_comparison() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
    pr_default TEACHER_PATH "Qwen/Qwen3-VL-8B-Instruct"
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/virl39k/train.parquet"
        "data.val_files=$DATA_ROOT/mmk12/test.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=$ROOT_DIR/examples/format_prompt/math_perception.jinja"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=200704"
        "data.max_pixels=1003520"
        "data.max_prompt_length=4096"
        "data.max_response_length=2048"
        "data.rollout_batch_size=128"
        "data.val_batch_size=512"
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "algorithm.log_entropy=true"
        "worker.teacher.source=model"
        "worker.teacher.model.model_path=$TEACHER_PATH"
        "worker.actor.use_torch_compile=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.strategy=adamw_bf16"
        "worker.actor.global_batch_size=128"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.loss_avg_mode=token"
        "worker.actor.micro_batch_size_per_device_for_update=4"
        "worker.actor.micro_batch_size_per_device_for_experience=8"
        "worker.actor.fsdp.torch_dtype=bf16"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.rollout.n=8"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.6"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=8"
        "worker.rollout.val_override_config.temperature=1.0"
        "worker.rollout.val_override_config.top_p=0.99"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/math.py:compute_score"
        "trainer.project_name=Comparison-OPD-Qwen3-VL-2B"
        "trainer.n_gpus_per_node=4"
        "trainer.total_epochs=2"
        "trainer.val_freq=15"
        "trainer.save_freq=15"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
        "trainer.save_limit=1"
    )
    launch_training "$@"
}
