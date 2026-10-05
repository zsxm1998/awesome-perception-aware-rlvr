#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# The paper's recipe (Sec. 5, App. A.7: GRPO, clip 0.2, KL 1e-2, 2 epochs); the rest follows the released script
# qwen2_5_vl_7b_train36k_vapo.sh and examples/config.yaml of the official code. See README.md.
launch_vapo() {
    pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/vapo/train.parquet"
        "data.val_files=$DATA_ROOT/vapo_val/val.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=$ROOT_DIR/examples/format_prompt/math_perception.jinja"
        "data.filter_overlong_prompts=true"
        "data.max_prompt_length=4096"
        "data.max_response_length=2048"
        "data.rollout_batch_size=384"
        "data.mini_rollout_batch_size=null"
        "data.val_batch_size=1024"
        "data.min_pixels=262144"
        "data.max_pixels=4194304"
        "data.seed=1"
        "algorithm.adv_estimator=grpo"
        "algorithm.disable_kl=false"
        "algorithm.use_kl_loss=true"
        "algorithm.kl_penalty=low_var_kl"
        "algorithm.kl_coef=1e-2"
        "algorithm.online_filtering=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=5e-6"
        "worker.actor.optim.weight_decay=1e-2"
        "worker.actor.optim.strategy=adamw"
        "worker.actor.optim.lr_warmup_ratio=0.0"
        "worker.actor.max_grad_norm=1.0"
        "worker.actor.global_batch_size=128"
        "worker.actor.micro_batch_size_per_device_for_update=4"
        "worker.actor.micro_batch_size_per_device_for_experience=16"
        "worker.actor.padding_free=true"
        "worker.actor.dynamic_batching=true"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.fsdp.torch_dtype=bf16"
        "worker.actor.offload.offload_params=true"
        "worker.actor.offload.offload_optimizer=true"
        "worker.rollout.n=5"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=0.99"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.6"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=0.0"
        "worker.rollout.val_override_config.top_p=1.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/math.py:compute_score"
        "trainer.project_name=VAPO-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.total_epochs=2"
        "trainer.val_freq=3"
        "trainer.save_freq=5"
        "trainer.save_limit=1"
        "trainer.val_generations_to_log=3"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
    )
    launch_training "$@"
}

# VAPO on top of the recipe: K = 20 claim probes per correct response, beta = 1.5, gamma = 0.1. The perception
# reward replaces 0.1 of the accuracy weight, as the released reward: 0.8 accuracy + 0.1 format + 0.1 perception.
# The processor cache lets the probes of a prompt (n x K requests on one image) reuse its processed image.
VAPO_ARGS=(
    "algorithm.claim_probe_count=20"
    "algorithm.claim_probe_late_emphasis=1.5"
    "worker.reward.reward_function_kwargs={\"format_weight\":0.1,\"perception_weight\":0.1}"
    "worker.rollout.mm_processor_cache_gb=4"
)
