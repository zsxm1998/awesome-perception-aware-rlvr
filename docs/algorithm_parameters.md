# Perception-Reasoning Algorithm Parameters

This note only lists which `algorithm.*` parameters are used by each perception-reasoning block. Parameter semantics and validation rules live in [`verl/trainer/config.py`](../verl/trainer/config.py).

## Tokenizer and Processor

- `worker.actor.model.plain_think_tokens` (`auto` | `true` | `false`, default `auto`): tokenize `<think>` / `</think>` as plain text when they are untrained added tokens (Qwen3-VL Instruct); see [`verl/utils/plain_think.py`](../verl/utils/plain_think.py) and the README.
- `worker.actor.model.max_dynamic_patch` (InternVL only, default: the model config's): maximum tiles per image, for the trainer's processor and the vLLM rollout (GRIT InternVL uses 2). Pass the same value to the evaluation (`--max-dynamic-patch`).

## Loss Averaging

- `worker.actor.loss_avg_mode`: `token` (default) weights every response token of a mini-batch equally; `seq` averages over each response's tokens, then weights every response equally across micro-batches and ranks (verl's seq-mean-token-mean). Used by the PEPO GRPO arms and GRIT. See [implementation notes](implementation_notes.md#seq-and-token-loss-averaging).

## Online Filtering (Dynamic Sampling)

Used by the DAPO arms.

- `algorithm.online_filtering`, `algorithm.filter_key`, `algorithm.filter_low`, `algorithm.filter_high`, `trainer.max_try_make_batch`
- `algorithm.filter_criterion`: `mean_range` (default: group mean of `filter_key` in (`filter_low`, `filter_high`)) or `std` (the group's values differ, as ms-swift; PEPO)
- `algorithm.online_filtering_fallback`: `error` (default), `keep_round` (a round that keeps no group is kept whole, as PAPO's code), `first_round` (after the last round, train on the first round unfiltered, as ms-swift; PEPO)

## Shared Entropy Controls

Used by entropy logging, entropy masking, and entropy regularization paths shared across the comparison scripts.

- `algorithm.log_entropy`
- `algorithm.entropy_loss_type`
- `algorithm.invariant_entropy_coef`
- `algorithm.top_entropy_quantile`
- `algorithm.entropy_thr_granularity`: `micro_batch` (default; within each update micro-batch, from the policy being updated; CGPO), `batch` (over every token of the rollout batch, from the rollout policy, before the update; ToR) or `response`
- `algorithm.entropy_top_p`: entropy of the top-p truncated distribution for `entropy_thr_granularity=batch` (ToR: 0.95)

## Decremental / Counterfactual View Controls

Used by PAPO, VPPO, ToR, CGPO, DVRP, PGPO, CFPO and VEPO whenever a decremental / counterfactual auxiliary view is enabled.

- `algorithm.corrupt_image`
- `algorithm.corrupt_image_kwargs`: e.g. `random_patch` takes `patch_size`, `black_prob` and `mask_before_resize` (mask the original image before resizing, as PAPO's code); `gaussian_noise` takes `std`, added to the image processor's normalized pixel values (as VEPO's code)
- `algorithm.corrupt_image_position`: `prompt` (one view per prompt) or `response` (one per response)

## Visual Sensitivity Scoring

Used by perception-token selection and response-level scaling. The default keeps the existing sampled low-var KL behavior; full-vocab options compare original and counterfactual output distributions.

- `algorithm.visual_sensitivity_metric`: sampled-token metrics `sampled_low_var_kl` (k3), `sampled_boxcox`, `sampled_abs_log_ratio` (\|log p − log q\|, ToR), full-vocabulary metrics, `vepo`, or `hidden_state_similarity` (PEPO)
- `algorithm.visual_sensitivity_boxcox_alpha`
- `algorithm.visual_sensitivity_hidden_metric`
- `algorithm.visual_token`
- `algorithm.visual_sensitivity_log_metrics`
- `algorithm.visual_sensitivity_jsd_weight`
- `algorithm.visual_sensitivity_entropy_gate`
- `algorithm.visual_sensitivity_reference`
- `algorithm.visual_sensitivity_loss_coef`
- `algorithm.decremental_entropy_coef`
- `algorithm.top_perception_quantile`
- `algorithm.perception_thr_granularity`: as `entropy_thr_granularity`
- `algorithm.include_region_tokens_in_perception_mask`
- `algorithm.normalize_pg_loss_by_selected_tokens`

## Advantage Scaling

Used by VPPO, CGPO, PGPO and PEPO. `algorithm.response_advantage_scaling_method`
is a legacy alias for the VPPO/CGPO response-level formulas; new configs should
prefer `algorithm.advantage_scaling_method`.

- `algorithm.advantage_scaling_method`
- `algorithm.response_advantage_scaling_method`
- `algorithm.vppo_response_scaling_min`
- `algorithm.cgpo_response_scaling_coef`
- `algorithm.pgpo_token_scaling_threshold` (PGPO tau)
- `algorithm.pgpo_token_scaling_boost` (PGPO beta)
- `algorithm.pgpo_threshold_mode`: `absolute` (default, the paper) or `quantile` (the authors' development code)
- `algorithm.pgpo_low_weight_floor`: lower bound of the weights below the threshold (paper 0, development code 0.1)
- `algorithm.pgpo_mass_normalization`: rescale each response's weights to sum to its length (paper true, development code false)
- `algorithm.advantage_scaling_schedule`
- `algorithm.pepo_gate_alpha`
- `algorithm.pepo_gate_temperature`

## ToR Token Weighting

Used by ToR.

- `algorithm.tor_use_token_weighting`
- `algorithm.tor_rsn_weight`
- `algorithm.tor_prcp_weight`

## DVRP Incremental View

Used by DVRP-style incremental-view training.

- `algorithm.incremental_image_transform`
- `algorithm.incremental_image_kwargs`
- `algorithm.visual_robustness_loss_coef`
- `algorithm.incremental_entropy_coef`
- `algorithm.noise_t_init`
- `algorithm.noise_gamma`
- `algorithm.noise_t_max` (at most 1000): the noised view uses step t of NoisyRollout's 1,000-step forward-diffusion schedule

## Grounding Consistency

Used by CGPO.

- `algorithm.use_grounding_consistency_reward`
- `algorithm.grounding_consistency_reward_weight`
- `algorithm.grounding_consistency_detector`: `self` (default, existing rollout-model detector) or `grounding-dino` (`IDEA-Research/grounding-dino-base`)
- `algorithm.grounding_dino_device`: `worker` (default, one detector per rollout worker GPU), `auto`, `cpu`, or a torch CUDA device such as `cuda:0`
- `algorithm.grounding_dino_batch_size`: mini-batch size for Grounding DINO detector inference, default `4`
  - On CUDA, Grounding DINO is loaded in `float16` and runs under autocast; the detector model is unloaded after each detection pass to release GPU memory.
