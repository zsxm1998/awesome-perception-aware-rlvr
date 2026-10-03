#!/usr/bin/env bash
# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning (ICLR 2026).
#
# Multi-turn agentic RL with an image zoom-in tool on the official DeepEyes-Datasets-47k
# (V*-derived 22k + ArxivQA charts 14k + ThinkLite-VL 11k). Hyper-parameters follow the paper
# and the released 7B script: GRPO, 16 rollouts, 256 prompts per step with one update per step,
# lr 1e-6, no KL, 80 iterations, 20480 trajectory tokens (10240 per turn), at most 6 tool calls,
# reward 0.8 acc + 0.2 format + 1.2 tool (tool only when correct; ThinkLite: 1.2 acc + 0.4 format).
#
# Backbones: qwen2_5_vl_7b_* use the paper's Qwen2.5-VL-7B-Instruct, which grounds in absolute pixel
# coordinates of the resized image it sees (QWEN25_VL_ARGS); qwen3_vl_8b_* use Qwen3-VL-8B-Instruct
# with its native 0-1000 coordinates and tool-call chat template.
#
# Differences from the official release (see examples/reproduction/deepeyes/README.md):
#   * answer judging: set DEEPEYES_JUDGE_BASE_URL / DEEPEYES_JUDGE_MODEL to an OpenAI-compatible
#     judge (the paper uses Qwen2.5-72B-Instruct served by vLLM); without it a rule-based
#     matcher is used;
#   * ThinkLite samples go through the same tool-enabled rollout (the official env disables the
#     tool for them) but keep the math reward without tool bonus.
# The official setup recommends >= 32 GPUs for the 7B model; on one 8-GPU node expect long steps.
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

launch_deepeyes() {
    pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
    METHOD_COMMON_ARGS=(
        "data.train_files=$DATA_ROOT/deepeyes/train.parquet"
        "data.val_files=$DATA_ROOT/deepeyes/val.parquet"
        "data.prompt_key=problem"
        "data.answer_key=answer"
        "data.image_key=images"
        "data.video_key=videos"
        "data.format_prompt=null"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=200704"
        "data.max_pixels=1003520"
        "data.max_prompt_length=8192"
        "data.max_response_length=20480"
        "data.rollout_batch_size=256"
        "data.mini_rollout_batch_size=256"
        "data.val_batch_size=-1"
        "algorithm.adv_estimator=grpo"
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "algorithm.online_filtering=false"
        "worker.actor.use_torch_compile=false"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.strategy=adamw_bf16"
        "worker.actor.global_batch_size=256"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.micro_batch_size_per_device_for_update=1"
        "worker.actor.micro_batch_size_per_device_for_experience=1"
        "worker.actor.fsdp.torch_dtype=bf16"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.ref.fsdp.torch_dtype=bf16"
        "worker.rollout.n=16"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.6"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.max_num_batched_tokens=32768"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=0.0"
        "trainer.project_name=DeepEyes-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.max_steps=80"
        "trainer.total_epochs=1"
        "trainer.val_freq=-1"
        "trainer.val_before_train=false"
        "trainer.save_freq=10"
        "trainer.save_limit=1"
        "trainer.find_last_checkpoint=true"
    )
    launch_training "$@"
}

# Agentic rollout with the zoom-in tool (DeepEyes).
DEEPEYES_AGENT_ARGS=(
    "data.system_prompt=$ROOT_DIR/examples/system_prompt/deepeyes.txt"
    "worker.rollout.interaction_mode=agentic"
    "worker.rollout.agent_max_tool_calls=6"
    "worker.rollout.agent_max_tokens_per_turn=10240"
    "worker.rollout.agent_max_batch_images=256"
    "worker.rollout.agent_observation_min_pixels=3136"
    "worker.rollout.limit_images=16"
    "worker.rollout.mm_processor_cache_gb=4"
    "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/deepeyes.py:compute_score_official"
)

# Qwen2-VL / Qwen2.5-VL: the model writes bbox_2d in absolute pixels of the image it sees (after the
# min/max_pixels resize and the processor's 28-pixel rounding); the tool maps them back to the source
# image before cropping. Their stock chat template has no tool calling, hence the template override.
# Append after DEEPEYES_AGENT_ARGS (later arguments win).
QWEN25_VL_ARGS=(
    "data.system_prompt=$ROOT_DIR/examples/system_prompt/deepeyes_pixel.txt"
    "data.override_chat_template=$ROOT_DIR/examples/chat_template/qwen2_5_vl_tool_call.jinja"
    "worker.rollout.agent_bbox_format=pixel"
)

# "RL with text-only CoT" baseline of the paper: same data and reward, no tool.
TEXT_ONLY_ARGS=(
    "data.system_prompt=$ROOT_DIR/examples/system_prompt/deepeyes_text_only.txt"
    "data.max_response_length=10240"
    "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/deepeyes.py:compute_score_text_only"
)
