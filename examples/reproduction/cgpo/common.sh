#!/usr/bin/env bash
# CGPO (Counterfactual Grounding Policy Optimization), natural-image reproduction.
#
# The paper trains pathology models (SFT on public + in-house data, then RLVR on PathVQA
# and in-house VQA). The in-house data cannot be released, so this directory reproduces
# the RLVR stage on public natural-image data (ViRL39K train / MMK12 val, the same data
# as PAPO) with the paper's backbones and RL hyper-parameters:
#   lr 1e-6, 384 prompts x 8 rollouts per step (generation batch 3072),
#   1024 responses per update (global_batch_size 128 prompts), 200 steps,
#   KL beta 0.04, lambda 0.1, rho_r = rho_p = 0.3, gamma 0.5, reward = 0.1 format + 0.9 accuracy.
# GRPO and CGPO share the evidence-grounded output format and reward; only the
# policy-optimization algorithm differs.
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

launch_cgpo() {
    pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/virl39k/train.parquet"
        "data.val_files=$DATA_ROOT/mmk12/test.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=$ROOT_DIR/examples/format_prompt/xml_grounded_reasoning.jinja"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=200704"
        "data.max_pixels=1003520"
        "data.max_prompt_length=4096"
        "data.max_response_length=2048"
        "data.rollout_batch_size=384"
        "data.mini_rollout_batch_size=128"
        "data.val_batch_size=512"
        "algorithm.adv_estimator=grpo"
        "algorithm.use_kl_loss=true"
        "algorithm.kl_penalty=low_var_kl"
        "algorithm.kl_coef=0.04"
        "algorithm.log_entropy=true"
        "worker.actor.use_torch_compile=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.strategy=adamw_bf16"
        "worker.actor.global_batch_size=128"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.micro_batch_size_per_device_for_update=4"
        "worker.actor.micro_batch_size_per_device_for_experience=8"
        "worker.actor.fsdp.torch_dtype=bf16"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.ref.fsdp.torch_dtype=bf16"
        "worker.rollout.n=8"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=0.99"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.6"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=8"
        "worker.rollout.val_override_config.temperature=1.0"
        "worker.rollout.val_override_config.top_p=0.99"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/xml_grounded_reasoning.py:compute_score"
        "worker.reward.reward_function_kwargs={\"format_weight\":0.1}"
        "trainer.project_name=CGPO-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.max_steps=200"
        "trainer.total_epochs=2"
        "trainer.val_freq=10"
        "trainer.save_freq=20"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
        "trainer.save_limit=1"
    )
    launch_training "$@"
}

# Algorithm arguments of CGPO (Sec. 3 of the paper).
CGPO_ALGO_ARGS=(
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
    "algorithm.include_region_tokens_in_perception_mask=true"
    "algorithm.use_grounding_consistency_reward=true"
    "algorithm.grounding_consistency_reward_weight=0.5"
)
