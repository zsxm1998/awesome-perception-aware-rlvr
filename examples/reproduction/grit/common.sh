#!/usr/bin/env bash
# GRIT: Teaching MLLMs to Think with Images (NeurIPS 2025), reproduced with EasyR1.
#
# Official recipe (github.com/UCSB-AI/GRIT, scripts/train_base_config.sh): 20 training samples
# (10 VSR + 10 TallyQA), GRPO with 4 rollouts, lr 2e-6 cosine, KL beta 0.01, symmetric clip
# 0.28, max completion 1000 tokens, max_pixels 256*28*28, 200 steps on 8 GPUs.
# The answer reward uses a rule-based match instead of the paper's GPT-4o judge
# (examples/reward_function/grit.py:compute_score_official). Each step uses all 20 prompts
# (80 rollouts). Validation uses the GRIT VSR and
# TallyQA test sets, as in the official scripts (which validate on the test sets).
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

launch_grit() {
    pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/grit/train.parquet"
        "data.val_files=$DATA_ROOT/grit/test.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=$ROOT_DIR/examples/format_prompt/grit.jinja"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=3136"
        "data.max_pixels=200704"
        "data.max_prompt_length=1024"
        "data.max_response_length=1024"
        "data.rollout_batch_size=20"
        "data.val_batch_size=256"
        "algorithm.adv_estimator=grpo"
        "algorithm.use_kl_loss=true"
        "algorithm.kl_penalty=low_var_kl"
        "algorithm.kl_coef=0.01"
        "worker.actor.use_torch_compile=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=2e-6"
        "worker.actor.optim.lr_scheduler_type=cosine"
        "worker.actor.optim.strategy=adamw_bf16"
        "worker.actor.global_batch_size=20"
        "worker.actor.clip_ratio_low=0.28"
        "worker.actor.clip_ratio_high=0.28"
        "worker.actor.micro_batch_size_per_device_for_update=2"
        "worker.actor.micro_batch_size_per_device_for_experience=8"
        "worker.actor.fsdp.torch_dtype=bf16"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.ref.fsdp.torch_dtype=bf16"
        "worker.rollout.n=4"
        "worker.rollout.temperature=0.9"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.6"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=0.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/grit.py:compute_score_official"
        "trainer.project_name=GRIT-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.max_steps=200"
        "trainer.total_epochs=200"
        "trainer.val_freq=50"
        "trainer.save_freq=50"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
        "trainer.save_limit=1"
    )
    launch_training "$@"
}
