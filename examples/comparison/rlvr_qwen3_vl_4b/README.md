# RLVR controlled comparison on Qwen3-VL-4B-Instruct

[English](README.md) | [简体中文](README_zh.md)

The perception-aware RLVR papers reproduced in [examples/reproduction/](../../reproduction/) use different
backbones, training sets, baseline definitions and evaluation protocols, and several of them reuse
baseline numbers from other papers. This directory trains every RLVR method except DeepEyes
([why](#why-deepeyes-is-not-included)) under **one** recipe: same model, same data, same GRPO
hyper-parameters and the same evaluation. Each script only sets `ALGO_ARGS`, the switches that define the
method; everything else comes from [common.sh](common.sh). The other comparisons are listed in
[examples/comparison/](../README.md).

> [!IMPORTANT]
> **The results are a reference, not a verdict on the methods.** The methods are our re-implementations,
> each configuration is trained once (two runs of one configuration ended 0.6 points apart in the mean of the
> 3 groups), and the shared setting is not the one each method was designed and tuned for. Read
> [About the results](../README.md#about-the-results) before the tables.

## Shared setting

| | `examples/comparison/rlvr_qwen3_vl_4b/common.sh` |
| --- | --- |
| Model | Qwen3-VL-4B-Instruct; `<think>` / `</think>` are tokenized as plain text, because they are untrained added tokens in the Instruct checkpoint (`worker.actor.model.plain_think_tokens=auto`, see the [README](../../../README.md#-configuration)) |
| Training data | ViRL39K as processed by PAPO (`PAPOGalaxy/PAPO_ViRL39K_train`, 38,870); `vapo` reads the same rows in the same order with VAPO's visual claims added as a column |
| Validation | MMK12 test (`PAPOGalaxy/PAPO_MMK12_test`, 2,000), every 5 steps, 8 samples at T=1.0, top-p 0.99 |
| Prompt / reward | `<think>` + `\boxed{}` (`examples/format_prompt/math_perception.jinja`); 0.9 accuracy + 0.1 format (`examples/reward_function/math.py:compute_score`). `grit` and `cgpo` replace these with their grounded formats (see below) |
| Rollout | 384 prompts x 8 rollouts per step (`data.rollout_batch_size=384`, `data.mini_rollout_batch_size=128`), T=1.0, top-p 0.99 |
| Policy update | GRPO; update batch 128 prompts (3 updates per step); clip 0.2 / 0.2; token-level loss |
| Regularization | low-var KL loss to the reference model, 0.01; entropy penalty 0.005 on the full-vocabulary entropy (`algorithm.invariant_entropy_coef=0.005`, `algorithm.entropy_loss_type=full`) |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 2 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels |
| GPUs | 4 (A800 80GB) |

## Methods

Every method runs on the GRPO recipe above, also those whose papers build on DAPO, so that only the
perception-aware component differs; DAPO is included as a second baseline (a stronger RL recipe
without any perception-aware component). Each method keeps the image perturbation of its paper
(column "Auxiliary view"), because its thresholds and coefficients were tuned for that
perturbation; patch masking uses Qwen3-VL's 16-px patches. Parameter names are listed in
[docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md).

| Script | Method | Auxiliary view | `ALGO_ARGS` (abridged) |
| --- | --- | --- | --- |
| `grpo.sh` | GRPO baseline | – | none |
| `dapo.sh` | DAPO baseline | – | no KL (`algorithm.disable_kl=true`), clip-higher 0.2 / 0.28, dynamic sampling on accuracy in (0.01, 0.99); the shared entropy penalty is kept |
| `papo.sh` | [PAPO](../../reproduction/papo/README.md) | random 16-px patches blackened with p=0.6, one mask per prompt | KL_prcp maximized with `visual_sensitivity_loss_coef=0.01`, `visual_sensitivity_reference=current`; no Double Entropy terms beyond the shared entropy penalty |
| `vppo.sh` | [VPPO](../../reproduction/vppo/README.md) | random 16-px patches blackened with p=0.5, one mask per response | TGF `top_perception_quantile=0.4` per response; TAS `response_advantage_scaling_method=vppo`, `vppo_response_scaling_min=0.9`; `visual_sensitivity_reference=old`; no 0.06 entropy penalty |
| `tor.sh` | [ToR](../../reproduction/tor/README.md) | image removed (`no_image`) | `top_entropy_quantile=0.3` on the entropy of the top-p 0.95 set (`entropy_top_p=0.95`) and `top_perception_quantile=0.3` on \|log p − log q\| (`visual_sensitivity_metric=sampled_abs_log_ratio`), both over every token of the rollout batch (`*_thr_granularity=batch`), `tor_use_token_weighting=true`, `tor_rsn_weight=1.0`, `tor_prcp_weight=0.5`; `visual_sensitivity_reference=old` |
| `dvrp.sh` | [DVRP](../../reproduction/dvrp/README.md) | random patches (p=0.6) and `vp_diffusion` noise | `visual_sensitivity_loss_coef=0.01` (λ_nec), `visual_robustness_loss_coef=0.01` (λ_rob), `noise_t_init=500`, `noise_gamma=10`, `noise_t_max=1000`; no view-entropy terms |
| `pgpo.sh` | [PGPO](../../reproduction/pgpo/README.md) | attention to the visual tokens masked (`mask_visual_attention`) | `advantage_scaling_method=pgpo`, `pgpo_token_scaling_threshold=0.4`, `pgpo_token_scaling_boost=2.0` as in the authors' development code (`pgpo_threshold_mode=quantile`, `pgpo_low_weight_floor=0.1`, `pgpo_mass_normalization=false`; see the PGPO README), `visual_sensitivity_metric=sampled_low_var_kl`, `visual_sensitivity_reference=old` |
| `pepo.sh` | [PEPO](../../reproduction/pepo/README.md) | none (single forward) | `visual_sensitivity_metric=hidden_state_similarity` (cosine), `advantage_scaling_method=pepo`, `advantage_scaling_schedule=linear`, `pepo_gate_alpha=0.05`, `pepo_gate_temperature=1.8` |
| `cfpo.sh` | [CFPO](../../reproduction/cfpo/README.md) | in-model attention-value counterfactual | `corrupt_image=cross_modal_attention_value_mean` (λ=2), `visual_sensitivity_loss_coef=0.02`, `visual_sensitivity_reference=current` |
| `vepo.sh` | [VEPO](../../reproduction/vepo/README.md) | Gaussian noise on the normalized pixel values, std 1.08 (about 0.54 in pixel units with Qwen3-VL's image_std 0.5, the strength VEPO's code comment and paper describe), one noisy image per prompt | `visual_sensitivity_metric=vepo`, `visual_sensitivity_jsd_weight=0.7`, `visual_sensitivity_entropy_gate=normal_entropy`, `top_perception_quantile=0.2` per response, `normalize_pg_loss_by_selected_tokens=true` |
| `noisyrollout.sh` | [NoisyRollout](../../reproduction/noisyrollout/README.md) | VP-diffusion noise on the pixels, one noised image per prompt and step, used only to sample 4 of the 8 rollouts of the prompt | `rollout_image_transform=vp_diffusion`; α₀=450, λ=60 (the official MMK12-7B script), γ = 0.1 · t_max instead of its t_max / 3 ([why](#noisyrollouts-noise-schedule)), `pixel_rounding=floor`; all 8 rollouts form one group and are trained on the clean image |
| `vgpo.sh` | [VGPO](../../reproduction/vgpo/README.md) | none (single forward) | `visual_sensitivity_metric=hidden_state_similarity` on the last layer against the mean image-token state (`visual_sensitivity_hidden_layers=last`, `visual_sensitivity_hidden_pooling=prototype`), scored before the update (`visual_sensitivity_reference=old`); `advantage_scaling_method=vgpo` with β=0.3, γ=0.5, κ=0.2; factors per prompt group |
| `vapo.sh` | [VAPO](../../reproduction/vapo/README.md) | – (no image perturbation) | `claim_probe_count=20` probes per correct response, each asking whether one of the image's GPT-5 visual claims is correct after a random cut of the reasoning, `claim_probe_late_emphasis=1.5`; the probe question without the released trailing space, after which Qwen3-VL-4B answers the probes with "1" or "0" (see the [VAPO README](../../reproduction/vapo/README.md#probe-question)); reward 0.8 accuracy + 0.1 format + 0.1 perception (`perception_weight=0.1`); trained on `virl39k_claims` (ViRL39K with the claims; the 2,289 multi-image problems have none and get the GRPO reward) |
| `grit.sh` | [GRIT](../../reproduction/grit/README.md) | – | think / rethink / answer format with inline JSON evidence boxes (`examples/system_prompt/grit_GR.txt`, no format prompt) and `grit.py:compute_score` (format, box format, rule-based answer, 0.1·BLEU-1); trained on ViRL39K instead of GRIT's 20 samples |
| `cgpo.sh` | [CGPO](../../reproduction/cgpo/README.md) | evidence regions flattened to their local mean (`cgpo_flat`, one per response) | `<region>` evidence format (`xml_grounded_reasoning_v2.jinja` and reward, format weight 0.1); `top_entropy_quantile=0.3`, `top_perception_quantile=0.3` (within each update micro-batch, `*_thr_granularity=micro_batch`, as in the CGPO paper), `include_region_tokens_in_perception_mask=true`; `advantage_scaling_method=cgpo`, `cgpo_response_scaling_coef=0.1`; grounding-consistency reward weight 0.1; `visual_sensitivity_reference=old` |

Settings that belong to a paper's base recipe rather than to its method are replaced by the shared
recipe. The methods that their papers run on DAPO (VPPO, PGPO and VGPO; PAPO, ToR and DVRP report both
GRPO and DAPO) run on GRPO here, without DAPO's dynamic sampling and clip-higher; the DAPO baseline
keeps both. VPPO's entropy penalty (which the paper applies to its baseline as well), NoisyRollout's
entropy bonus, and the frozen vision tower of VEPO and NoisyRollout are not used either. The shared entropy penalty of 0.005 keeps
Qwen3-VL-4B training stable; several methods collapse without it. Of PAPO's Double Entropy loss
only the term on the original image carries gradient, and the shared penalty takes its place. The
entropy terms on PAPO's masked view and on DVRP's two auxiliary views carry no gradient, because
those views are scored once before the update (as in PAPO's official default), so they are left out
without changing training; the PAPO authors confirmed this and found that recomputing the masked
view with gradient does not help ([PAPO issue #20](https://github.com/MikeWangWZHL/PAPO/issues/20)).
The per-paper settings are in [examples/reproduction/](../../reproduction/).

### Why DeepEyes is not included

DeepEyes changes the rollout rather than the loss: the model calls a zoom-in tool and reads crops of
the original image over several turns. It does not fit this setting:

- **Its recipe has no tool for this kind of data.** DeepEyes enables the tool only on its
  V*-derived high-resolution images and ArxivQA charts; its reasoning problems (the 11,031
  ThinkLite-VL rows) are answered in a single turn without the tool. ViRL39K, a multimodal
  reasoning set like ThinkLite-VL, would get the same treatment, so DeepEyes' recipe would train
  like the GRPO baseline here, with a different reward.
- **There is little to zoom into.** In a sample of 360 ViRL39K images the median size is about
  73,000 pixels; 72% are below the 200,704-pixel minimum, so the model already sees them enlarged,
  and only 3% exceed the 1,003,520-pixel maximum. A crop of the original image adds hardly any
  detail that the model does not already see.
- **The evaluation would not show it.** The comparison suite measures reasoning, perception and
  hallucination at the training resolution, while the tool targets fine detail in high-resolution
  images, as measured by V* and HR-Bench (not in the suite); evaluating DeepEyes also needs the
  multi-turn agent loop.
- **Cost.** The agentic rollout (several turns and up to six tool calls per trajectory) is much
  slower than the single-turn rollout of the other methods.

Enabling the tool on every ViRL39K problem would be our own adaptation rather than DeepEyes. GRIT,
which also grounds its reasoning in the image, stays single-turn and only replaces the prompt and
the reward, so it runs in the shared setting. The [DeepEyes reproduction](../../reproduction/deepeyes/README.md)
compares DeepEyes with a text-only GRPO baseline on DeepEyes' data and reward (`qwen*_grpo.sh`
against `qwen*_grpo_deepeyes.sh`).

### NoisyRollout's noise schedule

NoisyRollout samples half of a prompt's rollouts from a noised image and trains all of them on the clean
image. With the official schedule mapped onto this run's steps (full-strength noise until about step 55,
midpoint at step 67), the noised rollouts drifted away from what the policy generates on the clean image:

| | Step 0 | Step 20 | Step 50 |
| --- | --- | --- | --- |
| Tokens of the noised rollouts with probability below 1e-4 under the clean image | 0.04% | 0.53% | 14.7% |
| Mean log-probability ratio of a token, clean vs. noised image | −0.03 | −0.16 | −6.8 |
| Share of the gradient magnitude from the noised rollouts | 51% | 54% | 87% |

The drift builds up between steps 20 and 50, while the entropy falls from 0.26 to 0.07; that run was stopped
at step 55. With the midpoint at 10% of the steps, the noise is about half its strength at step 20 and almost
gone by step 30, before the drift builds up. The tables report this run. It keeps the method's noise strength
and grouping and only shortens the noisy phase (about 15 steps of strong noise), so the method may gain less
here than with its own recipe.

## Running

```bash
bash scripts/prepare_data.sh comparison
bash examples/comparison/rlvr_qwen3_vl_4b/papo.sh                      # one method
for m in grpo papo vppo tor dvrp pgpo pepo cfpo vepo noisyrollout vgpo vapo grit cgpo; do
    bash examples/comparison/rlvr_qwen3_vl_4b/$m.sh                    # all methods, one after another
done
bash scripts/prepare_eval_data.sh comparison
bash scripts/eval.sh checkpoints/Comparison-Qwen3-VL-4B/papo --suite comparison
python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/*   # after training: keep the last and the best step
                                                                   # (GRPO's best step is the teacher of the OPD comparison)
```

Checkpoints go to `checkpoints/Comparison-Qwen3-VL-4B/<method>`. Any `key=value` appended to a
script overrides the shared setting (e.g. `bash examples/comparison/rlvr_qwen3_vl_4b/papo.sh trainer.total_epochs=1`);
`N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` and `EXPERIMENT_NAME` are read from the
environment. `grit` and `cgpo` answer in their own formats, so evaluate them with the matching
prompt: `--format-prompt none --system-prompt examples/system_prompt/grit_GR.txt` for `grit` (on the
GRIT sets, `--benchmarks grit_vsr,grit_tallyqa,grit_gqa,ovdeval_position` with these flags; the `grit`
suite holds GRIT's own protocol for the reproduction models),
`--format-prompt examples/format_prompt/xml_grounded_reasoning_v2.jinja` for `cgpo` (runs with a training record
get their training prompt by default). See
[eval/README.md](../../../eval/README.md).

## Results

Suite `comparison`: 19 benchmarks in three groups.

- **Math reasoning**: Geo3K, MathVista, We-Math, MMK12, MathVerse, MathVision, DynaMath. Most of the
  information is in the text of the problem.
- **Vision-dependent reasoning**: MathVerse-V (the vision-dependent subset of MathVerse, 1,308 of its
  2,180 questions), MMMU-Pro (vision setting: the question and the options are only in the image),
  LogicVista, Counting, AI2D, MME cognition (MME's 4 cognition subtasks: commonsense reasoning and
  the numbers, code and text shown in the image).
- **Perception and hallucination**: POPE, HallusionBench, MMStar, BLINK, MME perception (MME's other
  10 subtasks), CV-Bench.

The math and vision-dependent sets except AI2D and MME cognition use the PAPO-Eval protocol (rule-based match on the last `\boxed{}`,
avg@8 at T=1.0, top-p 1.0; DynaMath and MathVision from VPPO-Eval); the others are greedy. POPE is
macro-F1 over the random / popular / adversarial splits, HallusionBench question accuracy, MME
cognition and perception the scores out of 800 and 2,000 on a 0-100 scale, and CV-Bench the overall score of its dataset card
(the mean of the 2D and 3D accuracies). **Mean of the 3 groups**, the main score (in bold), averages the
three group means, so each group weighs the same whatever its number of benchmarks; **Mean of the 19
benchmarks** is the unweighted mean, given for reference. `bash scripts/eval.sh <run> --suite comparison` writes these means as the
`comparison: ...` columns of `eval/results/summary.csv` (see [eval/README.md](../../../eval/README.md#suites)).

The summary table and the three group tables evaluate every method at its last step (step 202), so they
compare what the same training budget ends with, late drift included. MMK12 test is also the validation set monitored during
training (as in PAPO and VPPO); [Best validation step](#best-validation-step) gives the second view, each
method at the step with the best validation reward, on the other 18 benchmarks.

| Method | Math reasoning | Vision-dependent reasoning | Perception and hallucination | Mean of the 3 groups | Mean of the 19 benchmarks | Training time (h) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (baseline) | 46.5 | 62.0 | 76.0 | 61.5 | 60.7 | – |
| GRPO | 62.8 | 70.7 | 78.4 | **70.6** | 70.2 | 31.0 (1.00×) |
| DAPO | 62.0 | 70.8 | 79.1 | **70.7** | 70.2 | 50.6 (1.63×) |
| PAPO | 69.1 | 75.1 | 78.2 | **74.1** | 73.9 | 35.6 (1.15×) |
| VPPO | 67.4 | 72.8 | 77.9 | **72.7** | 72.4 | 37.4 (1.21×) |
| ToR | 68.6 | 73.3 | 78.3 | **73.4** | 73.2 | 36.0 (1.16×) |
| DVRP | 68.9 | 74.6 | 77.8 | **73.8** | 73.5 | 40.5 (1.31×) |
| PGPO | 66.1 | 71.7 | 77.7 | **71.8** | 71.5 | 31.7 (1.02×) |
| PEPO | 67.4 | 73.4 | 78.1 | **73.0** | 72.7 | 32.0 (1.03×) |
| CFPO | 67.5 | 74.7 | 79.0 | **73.7** | 73.4 | 46.7 (1.51×) |
| VEPO | 68.6 | 73.6 | 78.7 | **73.6** | 73.4 | 34.7 (1.12×) |
| NoisyRollout | 59.0 | 70.5 | 79.3 | **69.6** | 69.1 | 31.0 (1.00×) |
| VGPO | 52.2 | 63.0 | 76.6 | **63.9** | 63.3 | 29.8 (0.96×) |
| VAPO | 63.0 | 70.1 | 78.2 | **70.4** | 70.0 | 34.2 (1.10×) |
| GRIT | 65.8 | 70.4 | 77.5 | **71.2** | 70.9 | 32.0 (1.03×) |
| CGPO | 67.0 | 70.5 | 77.8 | **71.8** | 71.5 | 41.1 (1.33×) |

**Training time** is the wall-clock time in hours of the 202 training steps on 4 A800 80GB GPUs (rollout, reward,
the auxiliary views and the updates), without the validation runs and checkpoint saves, which are the same
for every method; in parentheses, the ratio to GRPO. The runs shared 8-GPU nodes two at a time, so
differences of a few percent can come from the node load.

### Math reasoning

| Method | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVision | DynaMath | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (baseline) | 48.9 | 66.4 | 57.7 | 41.9 | 47.4 | 14.0 | 49.0 | 46.5 |
| GRPO | 54.3 | 73.6 | 72.5 | 64.7 | 69.6 | 34.0 | 70.8 | 62.8 |
| DAPO | 55.6 | 73.8 | 72.7 | 61.5 | 67.9 | 32.8 | 70.1 | 62.0 |
| PAPO | 63.9 | 74.4 | 76.7 | 71.8 | 77.9 | 45.1 | 73.8 | 69.1 |
| VPPO | 62.6 | 73.2 | 76.2 | 68.5 | 75.6 | 42.0 | 73.7 | 67.4 |
| ToR | 62.7 | 73.7 | 77.1 | 70.2 | 77.4 | 44.8 | 74.7 | 68.6 |
| DVRP | 63.2 | 73.9 | 76.8 | 71.5 | 77.8 | 44.4 | 74.8 | 68.9 |
| PGPO | 62.0 | 71.8 | 74.3 | 68.6 | 74.5 | 40.5 | 71.0 | 66.1 |
| PEPO | 60.3 | 72.8 | 76.9 | 70.2 | 75.4 | 42.7 | 73.4 | 67.4 |
| CFPO | 59.7 | 75.6 | 73.8 | 70.7 | 78.0 | 39.9 | 74.7 | 67.5 |
| VEPO | 60.9 | 75.9 | 77.4 | 71.1 | 77.0 | 44.0 | 74.0 | 68.6 |
| NoisyRollout | 54.5 | 69.0 | 71.7 | 55.4 | 64.5 | 30.1 | 67.9 | 59.0 |
| VGPO | 44.8 | 68.8 | 62.3 | 49.4 | 54.9 | 21.3 | 63.7 | 52.2 |
| VAPO | 56.0 | 73.4 | 72.5 | 64.1 | 69.8 | 33.8 | 71.1 | 63.0 |
| GRIT | 59.5 | 73.4 | 74.8 | 67.6 | 74.8 | 40.1 | 70.6 | 65.8 |
| CGPO | 60.2 | 74.2 | 77.0 | 68.8 | 75.0 | 41.8 | 72.3 | 67.0 |

### Vision-dependent reasoning

| Method | MathVerse-V | MMMU-Pro | LogicVista | Counting | AI2D | MME (cognition) | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (baseline) | 44.1 | 27.1 | 41.7 | 87.9 | 78.5 | 92.5 | 62.0 |
| GRPO | 64.3 | 41.5 | 54.2 | 89.6 | 83.9 | 90.8 | 70.7 |
| DAPO | 63.8 | 39.7 | 54.9 | 90.1 | 83.9 | 92.5 | 70.8 |
| PAPO | 73.5 | 45.8 | 62.4 | 93.1 | 83.2 | 92.9 | 75.1 |
| VPPO | 72.2 | 43.5 | 58.5 | 84.9 | 83.7 | 94.1 | 72.8 |
| ToR | 73.5 | 45.0 | 59.7 | 86.9 | 82.7 | 92.0 | 73.3 |
| DVRP | 73.9 | 46.0 | 60.8 | 92.9 | 82.5 | 91.6 | 74.6 |
| PGPO | 71.1 | 42.3 | 58.2 | 81.3 | 83.2 | 94.1 | 71.7 |
| PEPO | 72.0 | 44.4 | 59.6 | 87.5 | 82.9 | 93.8 | 73.4 |
| CFPO | 74.4 | 44.7 | 59.1 | 97.1 | 81.9 | 90.9 | 74.7 |
| VEPO | 72.8 | 43.8 | 58.4 | 90.6 | 83.8 | 92.5 | 73.6 |
| NoisyRollout | 59.9 | 37.9 | 53.9 | 95.5 | 84.7 | 91.2 | 70.5 |
| VGPO | 50.4 | 33.8 | 42.7 | 80.1 | 79.4 | 91.7 | 63.0 |
| VAPO | 65.3 | 40.8 | 54.7 | 83.9 | 82.8 | 92.8 | 70.1 |
| GRIT | 70.8 | 39.3 | 52.6 | 82.9 | 81.8 | 94.7 | 70.4 |
| CGPO | 71.5 | 41.5 | 57.2 | 77.8 | 83.1 | 91.6 | 70.5 |

### Perception and hallucination

| Method | POPE | HallusionBench | MMStar | BLINK | MME (perception) | CV-Bench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (baseline) | 88.4 | 72.8 | 65.0 | 62.3 | 83.0 | 84.4 | 76.0 |
| GRPO | 89.0 | 74.1 | 71.0 | 66.1 | 84.4 | 85.5 | 78.4 |
| DAPO | 89.0 | 73.8 | 71.9 | 67.3 | 86.5 | 86.0 | 79.1 |
| PAPO | 89.3 | 75.0 | 68.9 | 66.9 | 83.0 | 86.2 | 78.2 |
| VPPO | 88.4 | 75.0 | 69.7 | 66.0 | 82.9 | 85.7 | 77.9 |
| ToR | 88.3 | 75.3 | 69.9 | 67.7 | 82.3 | 86.1 | 78.3 |
| DVRP | 88.5 | 73.7 | 69.1 | 66.7 | 82.9 | 85.8 | 77.8 |
| PGPO | 88.4 | 73.6 | 68.7 | 65.1 | 84.7 | 85.8 | 77.7 |
| PEPO | 89.3 | 74.5 | 69.7 | 66.3 | 83.3 | 85.7 | 78.1 |
| CFPO | 88.9 | 74.3 | 69.3 | 69.0 | 85.8 | 86.4 | 79.0 |
| VEPO | 89.0 | 74.0 | 71.3 | 67.7 | 83.9 | 86.1 | 78.7 |
| NoisyRollout | 89.3 | 75.8 | 71.1 | 69.0 | 83.7 | 87.1 | 79.3 |
| VGPO | 88.9 | 71.4 | 67.2 | 64.8 | 81.8 | 85.6 | 76.6 |
| VAPO | 88.9 | 74.1 | 69.7 | 66.6 | 83.6 | 86.2 | 78.2 |
| GRIT | 89.2 | 71.5 | 68.5 | 65.0 | 86.2 | 84.4 | 77.5 |
| CGPO | 88.8 | 72.6 | 68.5 | 66.5 | 84.9 | 85.8 | 77.8 |

MME (cognition) and MME (perception) are MME's cognition score (commonsense reasoning, numerical calculation,
text translation and code reasoning; 260 questions, out of 800) and its perception score (the other 10 subtasks;
2,114 questions, out of 2,000), each on a 0-100 scale.

### Best validation step

Each method at the step with the best validation reward (`best_global_step` in the run's
`checkpoint_tracker.json`; `scripts/finalize_run.py` keeps this step), evaluated with
`bash scripts/eval.sh checkpoints/Comparison-Qwen3-VL-4B/<method>/global_step_<best> --suite comparison`.
MMK12 is left out, since the step is chosen on it, so math reasoning holds 6 benchmarks like the other two
groups and the mean of the 3 groups is also the mean of the 18 benchmarks. The `comparison: ...` columns of that evaluation
still cover the 19 benchmarks; compute these means from the per-benchmark scores without MMK12. The other math sets are close to MMK12, so
the math reasoning of a method whose best step comes well before its last may still be slightly optimistic.
The last two columns give the same mean at the last step (202) and the difference between the two, from the
rounded values in the table.

| Method | Best step | Math reasoning (without MMK12) | Vision-dependent reasoning | Perception and hallucination | Mean of the 3 groups | Mean of the 3 groups at the last step | Best − last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (baseline) | – | 47.2 | 62.0 | 76.0 | 61.7 | – | – |
| GRPO | 175 | 65.9 | 72.4 | 78.0 | **72.1** | 70.5 | +1.6 |
| DAPO | 105 | 70.2 | 74.6 | 78.6 | **74.5** | 70.7 | +3.8 |
| PAPO | 202 | 68.6 | 75.1 | 78.2 | **74.0** | 74.0 | 0.0 |
| VPPO | 200 | 67.4 | 73.4 | 77.6 | **72.8** | 72.6 | +0.2 |
| ToR | 202 | 68.4 | 73.3 | 78.3 | **73.3** | 73.3 | 0.0 |
| DVRP | 200 | 68.3 | 74.6 | 77.7 | **73.5** | 73.6 | −0.1 |
| PGPO | 202 | 65.7 | 71.7 | 77.7 | **71.7** | 71.7 | 0.0 |
| PEPO | 200 | 67.0 | 73.0 | 78.1 | **72.7** | 72.8 | −0.1 |
| CFPO | 190 | 67.1 | 74.8 | 78.9 | **73.6** | 73.5 | +0.1 |
| VEPO | 170 | 68.4 | 73.2 | 78.5 | **73.4** | 73.5 | −0.1 |
| NoisyRollout | 160 | 67.0 | 74.6 | 79.1 | **73.6** | 69.8 | +3.8 |
| VGPO | 165 | 65.9 | 71.1 | 77.1 | **71.3** | 64.1 | +7.2 |
| VAPO | 155 | 66.9 | 72.5 | 78.4 | **72.6** | 70.3 | +2.3 |
| GRIT | 202 | 65.5 | 70.4 | 77.5 | **71.1** | 71.1 | 0.0 |
| CGPO | 160 | 66.8 | 71.3 | 77.6 | **71.9** | 71.7 | +0.2 |
