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
- `algorithm.online_filtering_fallback`: `error` (default), `keep_round` (a round that keeps no group is kept whole, as PAPO's code), `first_round` (after the last round, train on the first round unfiltered, as ms-swift; PEPO; needs `data.mini_rollout_batch_size` unset or at least `data.rollout_batch_size`)

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

- `algorithm.visual_sensitivity_metric`: sampled-token metrics `sampled_low_var_kl` (k3), `sampled_boxcox`, `sampled_abs_log_ratio` (\|log p − log q\|, ToR), `sampled_positive_log_ratio` (max(log p − log q, 0), VA-OPD's visual advantage), full-vocabulary metrics, `vepo`, or `hidden_state_similarity` (PEPO)
- `algorithm.visual_sensitivity_boxcox_alpha`
- `algorithm.visual_sensitivity_hidden_metric`
- `algorithm.visual_sensitivity_hidden_layers`: `all` (default, PEPO) averages the scores of every hidden state the model returns; `last` scores the final-norm output only (VGPO), captured without keeping the other layers
- `algorithm.visual_sensitivity_hidden_pooling`: `pairwise_mean` (default, PEPO) averages a token's similarity to each visual token; `prototype` takes the cosine to the mean visual hidden state (VGPO)
- `algorithm.visual_token`
- `algorithm.visual_sensitivity_log_metrics`
- `algorithm.visual_sensitivity_jsd_weight`
- `algorithm.visual_sensitivity_entropy_gate`
- `algorithm.visual_sensitivity_reference`: `current`, `old`, or `teacher` (the distillation teacher scores both views; VA-OPD, with `distill_weighting`). With `hidden_state_similarity` and `advantage_scaling_method=vgpo`, `old` scores the tokens in the old log-prob forward
- `algorithm.visual_sensitivity_loss_coef`
- `algorithm.decremental_entropy_coef`
- `algorithm.top_perception_quantile`
- `algorithm.perception_thr_granularity`: as `entropy_thr_granularity`
- `algorithm.include_region_tokens_in_perception_mask`
- `algorithm.normalize_pg_loss_by_selected_tokens`

## Advantage Scaling

Used by VPPO, CGPO, PGPO, PEPO and VGPO. `algorithm.response_advantage_scaling_method`
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
- `advantage_scaling_method=vgpo` (VGPO): before the update, the driver turns the old-policy cosines S of every response token into
  factors f_inter · f_intra that multiply the advantages. Scores ρ = S − m; in the last `vgpo_gate_tail_ratio` (γ) of each
  response, the top `vgpo_gate_top_ratio` (κ) of ρ are multiplied by 1 + β t / (T − 1) (`vgpo_compensation_strength`, β);
  per response these compensated scores are min-max normalized to ŵ and f_intra = clamp(1 + ŵ − mean ŵ, 0.1, 2); the sum
  of ŵ is min-max normalized among the responses of the same prompt and f_inter = clamp(1 + ŝ − mean ŝ, 0.9, 2). Requires
  `visual_sensitivity_metric=hidden_state_similarity` and `visual_sensitivity_reference=old`.
- `algorithm.vgpo_compensation_strength` (β, default 0.3), `algorithm.vgpo_gate_tail_ratio` (γ, 0.5), `algorithm.vgpo_gate_top_ratio` (κ, 0.2)
- `algorithm.vgpo_score_offset`: `official` (default; m = min(0, smallest cosine of the rollout batch), what the released code's min-max amounts to) or `paper` (m = −1, Eq. 4)
- `algorithm.vgpo_trajectory_score`: `compensated` (default, the released code: the sum of ŵ) or `raw` (Eq. 9: the sum of the compensated scores before normalization)

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

## Rollouts from Transformed Images

Used by NoisyRollout. Off by default (`null`); the generation step is then unchanged.

- `algorithm.rollout_image_transform`: `vp_diffusion` samples the second half of each prompt's
  `worker.rollout.n` rollouts from a noised copy of its images (one copy per prompt and step, seeded by
  `data.seed`, the step and the prompt's position). All rollouts of a prompt share its advantage group,
  and every log-probability (old, reference, current) is computed on the clean images. Requires an even
  `worker.rollout.n` and `worker.rollout.interaction_mode=one_shot`; `adv_estimator=remax` is rejected. With online
  filtering (DAPO), a prompt is kept or dropped on the accuracy of all its rollouts, the noised ones included,
  as its advantage group; the reproduction and comparison scripts run GRPO without filtering.
- `algorithm.rollout_image_transform_kwargs`: `noise_t_init` (α₀), `noise_gamma` (λ) and `noise_t_mid`
  (γ / t_max, in (0, 1)) of the annealed diffusion step ⌊α₀ · (1 − σ(λ (t − γ) / t_max))⌋, where t_max is
  the number of training steps; `noise_t_max` (default 1000) caps the step; `pixel_rounding` (`floor`,
  the default, truncates to 8 bits as the released NoisyRollout code; `round`).

## Visual-Claim Probes

Used by VAPO. Off by default (`0`); the step then has no extra pass, column or reward term.

- `algorithm.claim_probe_count` (K): after the rollout (and after online filtering), every response with accuracy 1
  whose row has claims is cut at K random punctuation marks of its reasoning (the text before `</think>`, else
  `<answer>`, else `\boxed`); at each cut the policy answers (greedy, one token among the candidates below)
  whether one of the row's claims is correct. The claims come from the training column `visual_claims` (a JSON
  list of `{"claim", "correct"}`; `[]` for rows without claims). The perception score
  R_perc = max(0, (Σ w·s / Σ w − 0.5) / 0.5) of a response, with s the right/wrong answers and
  w = exp(β · cut position / response length) in characters, is passed to the reward function as
  `perception_score`. Requires `worker.reward.reward_function_kwargs.perception_weight > 0` (γ of
  `examples/reward_function/math.py:compute_score`: (1 − format_weight − γ)·accuracy + format_weight·format +
  γ·1[accuracy = 1]·R_perc; a reward function without this argument is refused at start-up). Sets
  `worker.rollout.max_model_len` to prompt + response + 128 tokens when unset. Rejected with agentic rollout, `adv_estimator=remax`, `rollout_image_transform` and the
  grounding consistency reward. With online filtering, `filter_key` must be `accuracy` or `format`: the probes run
  on the kept groups, before the perception score exists (see the
  [VAPO README](../examples/reproduction/vapo/README.md#dynamic-sampling)). The probes are seeded by `data.seed`,
  the step and the response's row; validation runs no probes (its perception metrics stay 0). The training data
  is checked at start-up: a row with 1 to K − 1 claims, or no row with claims, is refused. Implementation:
  `verl/trainer/claim_probes.py`.
- `algorithm.claim_probe_late_emphasis` (β, default 1.5): weight of later cuts in R_perc.
- `algorithm.claim_probe_question` (default: the released `"\n<anchor>{claim} Is this claim correct? Answer (Yes/No): "`):
  the text after the response prefix, `{claim}` replaced by the claim. On the command line, put the value in
  double quotes inside the argument, e.g. `'algorithm.claim_probe_question="\n<anchor>{claim} Is this claim correct? Answer (Yes/No):"'`.
- `algorithm.claim_probe_yes_tokens`, `algorithm.claim_probe_no_tokens` (default: the released
  `["yes", "Yes", "True", "true", "1"]` and `["0", "no", "No", "false", "False"]`): the candidates of the greedy
  answer, each a single token of the tokenizer. Quote every item (`'algorithm.claim_probe_yes_tokens=["Yes"]'`):
  YAML reads an unquoted yes/Yes/True as a boolean, which is refused.

## Grounding Consistency

Used by CGPO.

- `algorithm.use_grounding_consistency_reward`
- `algorithm.grounding_consistency_reward_weight`
- `algorithm.grounding_consistency_aggregation`: how a response's per-region match scores become its reward. `group` (default, the earlier behavior): the frequency-weighted share of the group's re-detected regions that the response grounds consistently, so naming more regions scores higher; `response`: the mean over the response's own regions, a region the detector does not find scoring 0 (the CGPO paper and its released code; the CGPO scripts use it)
- `algorithm.grounding_consistency_area_discount`: multiply a region's match score by 1 − (the fraction of the image its boxes cover, their union), so a box around the whole image earns nothing and a tight box keeps almost all of its score; default off
- `algorithm.grounding_consistency_detector`: `self` (default, existing rollout-model detector) or `grounding-dino` (`IDEA-Research/grounding-dino-base`)
- `algorithm.grounding_dino_device`: `worker` (default, one detector per rollout worker GPU), `auto`, `cpu`, or a torch CUDA device such as `cuda:0`
- `algorithm.grounding_dino_batch_size`: mini-batch size for Grounding DINO detector inference, default `4`
  - On CUDA, Grounding DINO is loaded in `float16` and runs under autocast; the detector model is unloaded after each detection pass to release GPU memory.

## On-Policy Distillation

Used by the OPD comparison ([examples/comparison/opd_qwen3_vl_2b](../examples/comparison/opd_qwen3_vl_2b)) and by
VA-OPD, VGS, VCSD and Vision-OPD. The losses are in [`verl/trainer/distillation.py`](../verl/trainer/distillation.py);
see the [implementation notes](implementation_notes.md#on-policy-distillation).

Teacher (`worker.teacher`, off by default):

- `worker.teacher.source`: `none` (default), `model` (a frozen model, `worker.teacher.model.model_path`; the trainer
  checks at startup that it shares the student's token ids, vocabulary size, visual special tokens, rotary layout
  and image preprocessing) or `ema` (an fp32 exponential moving average of the actor, saved with its checkpoints)
- `worker.teacher.ema_rate` (`ema`, default 0.05)
- `worker.teacher.fsdp.*`, `worker.teacher.offload.offload_params`: as for the reference model

With a teacher, the rollout must sample from the student's own distribution (`temperature=1.0`, `top_p=1.0`,
`top_k=-1`) and never samples the ids at or beyond `len(tokenizer)` (padding rows of the LM head).

Objective from sampled tokens (as verl's and NeMo-RL's OPD):

- `algorithm.adv_estimator=teacher_log_ratio`: advantage log q(y_t) − log π_old(y_t) per token, with the PPO loss
- `algorithm.teacher_log_ratio_clip` (default 10, `null` for none)

Objective on the full distributions (a direct loss, no PPO ratio):

- `algorithm.distill_loss_coef` (0 = off) and `algorithm.policy_loss_coef` (weight of the policy loss next to it;
  0 = distillation only)
- `algorithm.distill_divergence`: `reverse_kl` KL(p‖q), `forward_kl` KL(q‖p), `jsd` with `algorithm.distill_jsd_beta`
  (M = (1 − β) p + β q, β in (0, 1))
- `algorithm.distill_support`: `full` (every id of the tokenizer) or `student_top_k` (the student's top
  `algorithm.distill_top_k` ids plus a tail bucket; Vision-OPD)
- `algorithm.distill_temperature` and `algorithm.distill_temperature_scope` (`all`: the distributions use T and the
  loss is multiplied by T²; `loss_scale_only`: only the T² factor, as VCSD's released code); they apply to every
  target, and the reproductions set a temperature for VCSD only
- `algorithm.distill_chunk_size` (response rows per fp32 chunk, default 256), `algorithm.distill_is_clip`
  (detached truncated importance weight, default none)
- `worker.actor.loss_avg_mode` averages it as the policy loss (`token` or `seq`)

Inputs of the teacher and of the target:

- `algorithm.teacher_view`: `original` (default) or `data_image` (the same prompt with the images of
  `data.teacher_image_key`; Vision-OPD)
- `algorithm.distill_target`: `teacher`, `contrast_sharpened` (VCSD) or `visual_gain` (VGS)
- `algorithm.distill_contrast_view`: `black` (VCSD: a black image of the same size) or `no_image` (VGS: the prompt
  without its images, read by the teacher and the student); with `teacher_view=data_image` or a contrast view, the
  views are forwarded packed together, which needs `worker.actor.ulysses_size=1`
- VCSD: `algorithm.vcsd_alpha`, `algorithm.vcsd_support_beta`, `algorithm.vcsd_anchor_coef`,
  `algorithm.vcsd_keep_token_ids` (`auto`: the generation config's end-of-sequence ids)
- VGS: `algorithm.vgs_steering_coef` (γ), `algorithm.vgs_text_prior_coef` (λ), `algorithm.vgs_vds_quantile`,
  `algorithm.vgs_vds_scope` (`micro_batch`, the default, takes the quantile within each update micro-batch and so
  depends on how a step is split; `global` takes it over the step with an extra teacher pass and does not),
  `algorithm.vgs_loss_scale` (η); the visual dependency score of the gate is the teacher's exact KL(q ‖ q_text)
  on the tokenizer's ids

Per-token weights (VA-OPD):

- `algorithm.distill_weighting=va_opd` with `algorithm.corrupt_image=pixelation`,
  `algorithm.visual_sensitivity_reference=teacher` and `algorithm.visual_sensitivity_metric=sampled_positive_log_ratio`;
  needs one token-mean update per rollout batch
- `algorithm.va_opd_softmax_temperature` (τ), `algorithm.va_opd_high_fraction` (p_v),
  `algorithm.va_opd_high_weight` (λ)

Logged once per update, independently of how it is split into micro-batches and ranks: `distill/loss` (the averaged
distillation term before its coefficient, as `worker.actor.loss_avg_mode` averages it) and `distill/<statistic>`
(means over the response tokens of the update: tokens beyond the tokenizer, the VCSD target's change of the teacher,
the VGS terms and gate; `distill/is_weight_mean` with an importance weight); `teacher/log_ratio_*` for
`teacher_log_ratio` and `teacher/ema_update_applied` for an EMA teacher.
