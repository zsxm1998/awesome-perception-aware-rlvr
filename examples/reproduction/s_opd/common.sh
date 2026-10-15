#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../common.sh"

# S-OPD (arXiv 2609.39120) setting (Table 6): the system prompt "You are a helpful assistant." and the think /
# \boxed{} instruction after the question, images within the Qwen3-VL processor's default bounds, one rollout per
# prompt at T=1.0 / top-p 1.0 and one update per step, AdamW with lr 1e-6 and weight decay 1e-2, gradient clipping
# at 1.0, PPO clipping 0.2 / 0.2, the vision tower trained, no entropy term, 8 GPUs.
# $1 selects the data: virl39k (the main experiments: ViRL39K, 192 prompts per step, prompt / response 4,096,
# 1 epoch, MMK12 test for monitoring only) or geo3k (the component analysis: Geometry3K train, 128 prompts per
# step, prompt 1,024 / response 2,048, 20 epochs, its test split for monitoring).
s_opd_common_args() {
    local data=$1 train_files val_files prompt_length response_length batch epochs freq
    case "$data" in
        virl39k)
            train_files="$DATA_ROOT/virl39k/train.parquet"
            val_files="$DATA_ROOT/mmk12/test.parquet"
            prompt_length=4096 response_length=4096 batch=192 epochs=1 freq=50
            ;;
        geo3k)
            train_files="$DATA_ROOT/geometry3k/train.parquet"
            val_files="$DATA_ROOT/geometry3k/test.parquet"
            prompt_length=1024 response_length=2048 batch=128 epochs=20 freq=20
            ;;
        *)
            echo "[s_opd] the data must be virl39k or geo3k, got '$data'" >&2
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
        "data.system_prompt=$ROOT_DIR/examples/system_prompt/helpful_assistant.txt"
        "data.filter_overlong_prompts=true"
        "data.min_pixels=65536"
        "data.max_pixels=16777216"
        "data.max_prompt_length=$prompt_length"
        "data.max_response_length=$response_length"
        "data.rollout_batch_size=$batch"
        "data.val_batch_size=512"
        "worker.actor.model.enable_gradient_checkpointing=true"
        "worker.actor.model.trust_remote_code=true"
        "worker.actor.model.freeze_vision_tower=false"
        "worker.actor.optim.lr=1e-6"
        "worker.actor.optim.weight_decay=1e-2"
        "worker.actor.optim.strategy=adamw"
        "worker.actor.max_grad_norm=1.0"
        "worker.actor.global_batch_size=$batch"
        "worker.actor.clip_ratio_low=0.2"
        "worker.actor.clip_ratio_high=0.2"
        "worker.actor.loss_avg_mode=token"
        "worker.actor.micro_batch_size_per_device_for_update=4"
        "worker.actor.micro_batch_size_per_device_for_experience=8"
        "worker.actor.offload.offload_params=false"
        "worker.actor.offload.offload_optimizer=false"
        "worker.rollout.temperature=1.0"
        "worker.rollout.top_p=1.0"
        "worker.rollout.tensor_parallel_size=1"
        "worker.rollout.gpu_memory_utilization=0.5"
        "worker.rollout.enable_chunked_prefill=false"
        "worker.rollout.enforce_eager=true"
        "worker.rollout.val_override_config.n=1"
        "worker.rollout.val_override_config.temperature=1.0"
        "worker.rollout.val_override_config.top_p=1.0"
        "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/math.py:compute_score"
        "trainer.project_name=S-OPD-Reproduce"
        "trainer.n_gpus_per_node=8"
        "trainer.total_epochs=$epochs"
        "trainer.val_freq=$freq"
        "trainer.save_freq=$freq"
        "trainer.save_limit=1"
        "trainer.val_generations_to_log=5"
        "trainer.val_before_train=true"
        "trainer.find_last_checkpoint=true"
    )
}

# The finalized last step of qwen3_vl_8b_grpo_teacher.sh (or TEACHER_PATH): its Hugging Face directory, finalized
# (scripts/finalize_run.py keeps the last step) or merged (actor/huggingface), whichever holds complete weights.
s_opd_grpo_teacher_path() {
    local run=${GRPO_TEACHER_RUN:-$ROOT_DIR/checkpoints/S-OPD-Reproduce/qwen3_vl_8b_grpo_teacher}
    python3 - "$run" "${DRY_RUN:-0}" "$ROOT_DIR/scripts" <<'PY'
import json
import sys
from pathlib import Path

run, dry_run = Path(sys.argv[1]), sys.argv[2] == "1"
sys.path.insert(0, sys.argv[3])
from finalize_run import inspect_checkpoint

try:
    step = json.loads((run / "checkpoint_tracker.json").read_text())["last_global_step"]
except (OSError, KeyError, ValueError):
    if dry_run:  # the configuration check does not load the teacher
        print(run / "global_step_LAST" / "actor")
        sys.exit(0)
    sys.exit(f"no GRPO teacher: {run} has no checkpoint_tracker.json; run "
             "examples/reproduction/s_opd/qwen3_vl_8b_grpo_teacher.sh first, or set TEACHER_PATH")
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
sys.exit(f"no complete Hugging Face weights for the GRPO teacher (last step {step}): " + "; ".join(reasons)
         + f". Run `python3 scripts/finalize_run.py {run}` (or scripts/model_merger.py --local_dir {actor}), "
         "or set TEACHER_PATH")
PY
}

# S-OPD's two terms (Eq. 7), as ALGO_ARGS for launch_s_opd; a weight of 0 leaves a term out (the Geometry3K
# component analysis).
# - TPC, weight $1: the k3 between the student's log-probs on the image and on a masked copy (blocks of the model's
#   patch size, 16 px for Qwen3-VL, each blackened with probability 0.6 on the original image before resizing),
#   weighted per token by the teacher's log-prob drop max(log q(y_t | image) - log q(y_t | masked), 0) on the same
#   copy, averaged over the tokens with a drop, and maximized.
# - PA, weight $2: the k3 to a copy with Gaussian noise (std 0.2 on [0, 1] pixels, before resizing), minimized.
# One copy of each per prompt and step (one rollout per prompt). The incremental view needs `corrupt_image`, so the
# PA-only run also scores the masked copy, which no loss uses.
s_opd_terms() {
    local tpc=$1 pa=$2
    ALGO_ARGS=(
        "algorithm.corrupt_image=random_patch"
        "algorithm.corrupt_image_kwargs={\"black_prob\":0.6,\"mask_before_resize\":true}"
        "algorithm.corrupt_image_position=prompt"
    )
    if [[ $tpc != 0 ]]; then
        ALGO_ARGS+=(
            "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
            "algorithm.visual_sensitivity_reference=current"
            "algorithm.visual_sensitivity_loss_coef=$tpc"
            "algorithm.visual_sensitivity_loss_weighting=teacher"
        )
    fi
    if [[ $pa != 0 ]]; then
        ALGO_ARGS+=(
            "algorithm.incremental_image_transform=rgb_gaussian_noise"
            "algorithm.incremental_image_kwargs={\"std\":0.2,\"noise_before_resize\":true}"
            "algorithm.visual_robustness_loss_coef=$pa"
        )
    fi
}

# OPD from sampled tokens (S-OPD's base objective, the k1 estimator with the policy gradient): advantage
# log q(y_t) - log pi_old(y_t), clamped to +-10, with the PPO loss and no reference KL. $1 selects the data (above),
# $2 the default teacher: `8b` (Qwen3-VL-8B-Instruct) or `grpo` (the 8B GRPO run). The teacher the script and the
# command line choose wins (the last setting, as in the config merge): with worker.teacher.source=ema or none there
# is no external teacher; otherwise a teacher path not given (or null) is TEACHER_PATH, else the default teacher.
launch_s_opd() {
    local data=$1 teacher=$2 arg source="" path="" teacher_args=()
    shift 2
    if [[ $teacher != 8b && $teacher != grpo ]]; then
        echo "[s_opd] the teacher must be 8b or grpo, got '$teacher'" >&2
        exit 1
    fi
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
                if [[ $teacher == 8b ]]; then
                    TEACHER_PATH="Qwen/Qwen3-VL-8B-Instruct"
                else
                    TEACHER_PATH=$(s_opd_grpo_teacher_path)
                fi
            fi
            teacher_args+=("worker.teacher.model.model_path=$TEACHER_PATH")
        fi
    fi
    s_opd_common_args "$data"
    METHOD_COMMON_ARGS+=(
        "algorithm.disable_kl=true"
        "algorithm.use_kl_loss=false"
        "worker.rollout.n=1"
        "algorithm.adv_estimator=teacher_log_ratio"
        "algorithm.teacher_log_ratio_clip=10"
    )
    # after the script's own settings, so that the default fills a path the script left null
    launch_training "${teacher_args[@]}" "$@"
}

# The GRPO teacher (Table 6, "ViRL39K GRPO"): Qwen3-VL-8B-Instruct, 192 prompts x 5 rollouts per step, one update
# per step, response 4,096, 1 epoch. The paper gives no other GRPO setting: the rest is the setting above and this
# repository's default low-variance KL loss to the reference model (coefficient 0.01, examples/config.yaml).
launch_s_opd_grpo_teacher() {
    pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
    s_opd_common_args virl39k
    METHOD_COMMON_ARGS+=(
        "worker.rollout.n=5"
        "algorithm.adv_estimator=grpo"
        "worker.actor.micro_batch_size_per_device_for_update=2"
        "worker.actor.micro_batch_size_per_device_for_experience=4"
    )
    launch_training "$@"
}
