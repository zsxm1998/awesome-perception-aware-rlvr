# Adding a new method

Every perception-aware method in this repository is expressed as a combination of a few
building blocks that live in `verl/trainer/`. Most new methods only need one or two new
blocks plus a launcher script.

| Building block | Where | Examples |
| --- | --- | --- |
| Auxiliary (counterfactual) view of the image | `perception_reasoning_data.py` (`PerceptionReasoningCorruptionBuilder`) and `verl/utils/perturbations/` | `random_patch` (PAPO/VPPO/DVRP), `no_image` (ToR), `gaussian_noise` (VEPO), `cgpo_flat` (CGPO), `mask_visual_attention` (PGPO), `cross_modal_attention_value_mean` (CFPO) |
| Token-level visual-sensitivity signal | `visual_sensitivity.py` | `sampled_low_var_kl`, `sampled_boxcox`, `full_vocab_*`, `vepo`, `hidden_state_similarity` (PEPO) |
| Token selection / reweighting | `perception_reasoning_loss.py` (step 4) | top-entropy and top-perception masks (ToR/VPPO/CGPO), ToR token weights |
| Advantage scaling | `perception_reasoning_loss.py::_compute_advantage_scaling` | `vppo`, `cgpo`, `pgpo`, `pepo` |
| Auxiliary losses | `perception_reasoning_loss.py` (step 5) | PAPO implicit-perception KL, entropy terms, DVRP robustness KL |
| Extra rewards | `grounding_consistency.py`, `examples/reward_function/` | CGPO grounding consistency reward, GRIT reward, DeepEyes tool reward |
| Rollout mode | `verl/workers/agent/` | DeepEyes multi-turn zoom-in tool use |
| Teacher and distillation losses | `worker.teacher` (`verl/workers/fsdp_workers.py`), `distillation.py` | OPD from sampled tokens (`teacher_log_ratio`), full-distribution OPD, targets (`contrast_sharpened` for VCSD, `visual_gain` for VGS), token weights (`va_opd`), teacher views (`data_image` for Vision-OPD) |

All switches are fields of `AlgorithmConfig` in `verl/trainer/config.py`; see
[algorithm_parameters.md](algorithm_parameters.md) for which method uses which field.

## Step by step

1. **Config.** Add the new fields to `AlgorithmConfig` with a docstring and validate them in
   `AlgorithmConfig.post_init` (allowed values, ranges, incompatible combinations). If the
   method needs an auxiliary forward pass, make sure the `needs_*` helpers in
   `perception_reasoning_data.py` return `True` for it.
2. **Loss config.** Forward the new fields to the actor through
   `build_perception_reasoning_loss_config` (`perception_reasoning_data.py`). The actor only
   sees this dictionary.
3. **Algorithm.** Implement the new piece in the corresponding block:
   - a new image transform: add it to `_transform_multi_modal_data` of the corruption builder (or
     `_build_sample` for transforms that drop or replace the image) and to the
     `corrupt_image` choices in `config.py`;
   - a new advantage scaling: add a branch to `_compute_advantage_scaling` that returns a
     `[batch, response_len]` tensor of multiplicative factors and a metrics dict;
   - a new auxiliary loss: add it in step 5 of `compute_perception_reasoning_policy_loss`;
   - a new distillation target or weighting: add it to `verl/trainer/distillation.py` (a target in
     `_distill_rows`, per-token weights computed on the driver before the update, as `va_opd`) and keep the
     loss averaged by `worker.actor.loss_avg_mode` (see [implementation notes](implementation_notes.md#on-policy-distillation)).
   Log method-internal quantities under `algo/<method>/...`.
4. **Tests.** Add unit tests to `tests/test_perception_reasoning.py` that run the loss on tiny
   synthetic tensors (see the existing PGPO/PEPO tests) and, for transforms, on a small PIL
   image.
5. **Launchers.** Create `examples/reproduction/<method>/common.sh` with the paper's data, model and
   hyper-parameters (copy the closest existing method), a GRPO/DAPO baseline leaf and the method
   leaf. If the method should also take part in the controlled comparison, add
   `examples/comparison/qwen3_vl_4b/<method>.sh` (RL) or `examples/comparison/opd_qwen3_vl_2b/<method>.sh`
   (on-policy distillation) that only sets `ALGO_ARGS`.
6. **Data and evaluation.** If the paper uses new training data, register it in
   `scripts/data/prepare_train_data.py`; if it uses new benchmarks, follow
   [eval/README.md](../eval/README.md#adding-a-new-benchmark) and add a suite for the paper.
7. **Docs.** Add `examples/reproduction/<method>/README.md` (paper, official code, setting, differences
   from the official implementation, expected numbers) and a row to the method table in the
   top-level README.

## Launcher conventions

```bash
#!/usr/bin/env bash
set -euo pipefail
THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"           # defines launch_<method>() and METHOD_COMMON_ARGS

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_mymethod"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.my_new_coef=0.1"
)
EXTRA_ARGS=()
launch_mymethod "$@"
```

Arguments are appended in the order `METHOD_COMMON_ARGS`, `ALGO_ARGS`, `EXTRA_ARGS`, runtime
arguments (`LOGGER`, `NNODES`, `N_GPUS_PER_NODE`) and finally the command line, so later
values win. Any config key can be overridden from the command line, e.g.
`bash examples/reproduction/papo/qwen2_5_vl_7b_grpo_papo.sh trainer.total_epochs=1`.
