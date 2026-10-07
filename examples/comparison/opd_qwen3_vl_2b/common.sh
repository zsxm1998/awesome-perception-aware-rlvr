#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# Shared recipe of the on-policy distillation (OPD) comparison: a Qwen3-VL-2B student distilled from the
# GRPO run of the RL comparison (Qwen3-VL-4B-Instruct after ../qwen3_vl_4b/grpo.sh, at its best validation
# step) on the data, prompt, optimizer and lengths of that comparison (../qwen3_vl_4b/common.sh). Each update
# uses one fresh rollout batch (128 prompts x 8), the same number of samples per update and of updates as the
# RL recipe's 384 x 8 with three updates. There is no reference KL and no entropy term; the task reward is
# computed for logging only.

# Hugging Face directory of the GRPO run's best validation step (checkpoint_tracker.json), finalized
# (scripts/finalize_run.py keeps it) or merged (actor/huggingface), whichever holds complete weights.
grpo_teacher_path() {
    local run=${GRPO_TEACHER_RUN:-$ROOT_DIR/checkpoints/Comparison-Qwen3-VL-4B/grpo}
    python3 - "$run" "${DRY_RUN:-0}" "$ROOT_DIR/scripts" <<'PY'
import json
import sys
from pathlib import Path

run, dry_run = Path(sys.argv[1]), sys.argv[2] == "1"
sys.path.insert(0, sys.argv[3])
from finalize_run import inspect_checkpoint

try:
    step = json.loads((run / "checkpoint_tracker.json").read_text())["best_global_step"]
except (OSError, KeyError, ValueError):
    if dry_run:  # the configuration check does not load the teacher
        print(run / "global_step_BEST" / "actor")
        sys.exit(0)
    sys.exit(f"no GRPO teacher: {run} has no checkpoint_tracker.json; run examples/comparison/qwen3_vl_4b/grpo.sh "
             "first, or set TEACHER_PATH")
actor = run / f"global_step_{step}" / "actor"
reasons = []
for candidate in (actor, actor / "huggingface"):
    check = inspect_checkpoint([candidate])  # config.json and every safetensors file the index names
    if check.ok:
        print(candidate)
        sys.exit(0)
    reasons.append(f"{candidate}: {check.detail}")
if dry_run:
    print(actor)
    sys.exit(0)
sys.exit(f"no complete Hugging Face weights for the GRPO teacher (best validation step {step}): "
         + "; ".join(reasons) + f". Run `python3 scripts/finalize_run.py {run}` (or scripts/model_merger.py "
         f"--local_dir {actor}), or set TEACHER_PATH")
PY
}

launch_opd_comparison() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
    # The teacher the method and the command line choose (the last setting wins, as in the config merge). A
    # self-distilling method (VCSD: worker.teacher.source=ema) or source=none has no external teacher; otherwise
    # a teacher path not given (or null) is TEACHER_PATH, by default the GRPO run's best step.
    local arg source="" path="" teacher_args=()
    for arg in "${ALGO_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"; do
        case "$arg" in
            worker.teacher.source=*) source=${arg#*=} ;;
            worker.teacher.model.model_path=*) path=${arg#*=} ;;
        esac
    done
    if [[ $source != ema && $source != none ]]; then
        teacher_args=("worker.teacher.source=model")
        if [[ -z $path || $path == null ]]; then
            if [[ -z "${TEACHER_PATH:-}" ]]; then
                TEACHER_PATH=$(grpo_teacher_path)
            fi
            teacher_args+=("worker.teacher.model.model_path=$TEACHER_PATH")
        fi
    fi
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
    # after the method's own settings, so that the default fills a path the method left null
    launch_training "${teacher_args[@]}" "$@"
}
