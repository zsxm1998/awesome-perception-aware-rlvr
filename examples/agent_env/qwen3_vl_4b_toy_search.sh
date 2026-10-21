#!/usr/bin/env bash
# GRPO of Qwen3-VL-4B on the toy visual-search environment over HTTP (examples/agent_env/toy_search.py), the
# reward being the environment's episode reward. Start the server and write the tasks first:
#   python examples/agent_env/toy_search.py tasks --out data/agent_env/toy_search
#   python examples/agent_env/toy_search.py serve --port 8000 &
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/launcher.sh"

pr_default MODEL_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_4b_grpo_toy_search"
pr_default AGENT_ENV_URL "http://127.0.0.1:8000"
METHOD_COMMON_ARGS=(
    "data.train_files=$DATA_ROOT/agent_env/toy_search/train.parquet"
    "data.val_files=$DATA_ROOT/agent_env/toy_search/val.parquet"
    "data.prompt_key=prompt"
    "data.answer_key=answer"
    "data.format_prompt=null"
    "data.min_pixels=200704"
    "data.max_pixels=1048576"  # the 1024x1024 canvas is not downscaled
    "data.max_prompt_length=4096"
    "data.max_response_length=8192"
    "data.rollout_batch_size=64"
    "data.mini_rollout_batch_size=64"
    "data.val_batch_size=-1"
    "algorithm.adv_estimator=grpo"
    "algorithm.disable_kl=true"
    "algorithm.use_kl_loss=false"
    "worker.actor.global_batch_size=64"
    "worker.actor.micro_batch_size_per_device_for_update=1"
    "worker.actor.micro_batch_size_per_device_for_experience=1"
    "worker.actor.model.trust_remote_code=true"
    "worker.actor.offload.offload_params=false"
    "worker.actor.offload.offload_optimizer=false"
    "worker.rollout.n=8"
    "worker.rollout.tensor_parallel_size=1"
    "worker.rollout.gpu_memory_utilization=0.6"
    "worker.rollout.max_num_batched_tokens=16384"
    "worker.rollout.interaction_mode=agentic"
    "worker.rollout.agent_environment=remote_http"
    "worker.rollout.agent_environment_kwargs.url=$AGENT_ENV_URL"
    "worker.rollout.agent_max_tool_calls=4"
    "worker.rollout.agent_max_tokens_per_turn=2048"
    "worker.rollout.limit_images=6"
    "worker.rollout.mm_processor_cache_gb=4"
    "worker.rollout.val_override_config.n=1"
    "worker.rollout.val_override_config.temperature=0.0"
    "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/environment.py:compute_score"
    "trainer.project_name=agent_env"
    "trainer.n_gpus_per_node=8"
    "trainer.max_steps=60"
    "trainer.val_freq=10"
    "trainer.save_freq=-1"
)
launch_training "$@"
