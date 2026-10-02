# Perception-Reasoning Algorithm Parameters

This note only lists which `algorithm.*` parameters are used by each perception-reasoning block. Parameter semantics and validation rules live in [`verl/trainer/config.py`](../verl/trainer/config.py).

## Shared Entropy Controls

Used by entropy logging, entropy masking, and entropy regularization paths shared across the comparison scripts.

- `algorithm.log_entropy`
- `algorithm.entropy_loss_type`
- `algorithm.invariant_entropy_coef`
- `algorithm.top_entropy_quantile`
- `algorithm.entropy_thr_granularity`

## Decremental / Counterfactual View Controls

Used by PAPO, VPPO, ToR, CGPO, and DVRP whenever a decremental / counterfactual auxiliary view is enabled.

- `algorithm.corrupt_image`
- `algorithm.corrupt_image_kwargs`
- `algorithm.corrupt_image_position`

## Visual Sensitivity Scoring

Used by perception-token selection and response-level scaling. The default keeps the existing sampled low-var KL behavior; full-vocab options compare original and counterfactual output distributions.

- `algorithm.visual_sensitivity_metric`
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
- `algorithm.perception_thr_granularity`
- `algorithm.include_region_tokens_in_perception_mask`
- `algorithm.normalize_pg_loss_by_selected_tokens`

## Advantage Scaling

Used by VPPO, CGPO, and PGPO. `algorithm.response_advantage_scaling_method`
is a legacy alias for the VPPO/CGPO response-level formulas; new configs should
prefer `algorithm.advantage_scaling_method`.

- `algorithm.advantage_scaling_method`
- `algorithm.response_advantage_scaling_method`
- `algorithm.vppo_response_scaling_min`
- `algorithm.cgpo_response_scaling_coef`
- `algorithm.pgpo_token_scaling_threshold` (PGPO tau)
- `algorithm.pgpo_token_scaling_boost` (PGPO beta)
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
- `algorithm.noise_t_max`

## Grounding Consistency

Used by CGPO.

- `algorithm.use_grounding_consistency_reward`
- `algorithm.grounding_consistency_reward_weight`
- `algorithm.grounding_consistency_detector`: `self` (default, existing rollout-model detector) or `grounding-dino` (`IDEA-Research/grounding-dino-base`)
- `algorithm.grounding_dino_device`: `worker` (default, one detector per rollout worker GPU), `auto`, `cpu`, or a torch CUDA device such as `cuda:0`
- `algorithm.grounding_dino_batch_size`: mini-batch size for Grounding DINO detector inference, default `4`
  - On CUDA, Grounding DINO is loaded in `float16` and runs under autocast; the detector model is unloaded after each detection pass to release GPU memory.
