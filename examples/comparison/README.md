# Controlled comparison on Qwen3-VL-4B-Instruct

The perception-aware RLVR papers reproduced in [examples/reproduction/](../reproduction/) use different
backbones, training sets, baseline definitions and evaluation protocols, and several of them reuse
baseline numbers from other papers. Their reported gains are therefore hard to compare with each
other. This directory trains every method under **one** recipe: same model, same data, same GRPO
hyper-parameters and the same evaluation. Each script only sets `ALGO_ARGS`, the switches that
define the method; everything else comes from [qwen3_vl_4b/common.sh](qwen3_vl_4b/common.sh).

## Shared setting

| | `examples/comparison/qwen3_vl_4b/common.sh` |
| --- | --- |
| Model | Qwen3-VL-4B-Instruct; `<think>` / `</think>` are tokenized as plain text, because they are untrained added tokens in the Instruct checkpoint (`worker.actor.model.plain_think_tokens=auto`, see the [README](../../README.md#-configuration)) |
| Training data | ViRL39K as processed by PAPO (`PAPOGalaxy/PAPO_ViRL39K_train`, 38,870) |
| Validation | MMK12 test (`PAPOGalaxy/PAPO_MMK12_test`, 2,000), every 5 steps, 8 samples at T=1.0, top-p 0.99 |
| Prompt / reward | `<think>` + `\boxed{}` (`examples/format_prompt/math_perception.jinja`); 0.9 accuracy + 0.1 format (`examples/reward_function/math.py:compute_score`). `grit` and `cgpo` replace these with their grounded formats (see below) |
| Rollout | 384 prompts x 8 rollouts per step (`data.rollout_batch_size=384`, `data.mini_rollout_batch_size=128`), T=1.0, top-p 0.99 |
| Policy update | GRPO; update batch 128 prompts (3 updates per step); clip 0.2 / 0.2; token-level loss |
| Regularization | low-var KL loss to the reference model, 0.01; entropy penalty 0.005 on the full-vocabulary entropy (`algorithm.invariant_entropy_coef=0.005`, `algorithm.entropy_loss_type=full`) |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 2 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels |
| GPUs | 4 |

## Methods

Every method runs on the GRPO recipe above, also those whose papers build on DAPO, so that only the
perception-aware component differs; DAPO is included as a second baseline (a stronger RL recipe
without any perception-aware component). Where a paper
perturbs the image, the perturbation is harmonized to blackening 16-px patches (Qwen3-VL's patch
size) with probability 0.6, one mask per prompt (`algorithm.corrupt_image=random_patch`,
`corrupt_image_kwargs={"patch_size":16,"black_prob":0.6}`, `corrupt_image_position=prompt`); the
"Auxiliary view" column says where this replaces the paper's own perturbation. Parameter names are listed in
[docs/algorithm_parameters.md](../../docs/algorithm_parameters.md).

| Script | Method | Auxiliary view | `ALGO_ARGS` (abridged) |
| --- | --- | --- | --- |
| `grpo.sh` | GRPO baseline | – | none |
| `dapo.sh` | DAPO baseline | – | no KL (`algorithm.disable_kl=true`), clip-higher 0.2 / 0.28, dynamic sampling on accuracy in (0.01, 0.99); the shared entropy penalty is kept |
| `papo.sh` | [PAPO](../reproduction/papo/README.md) | random patch (paper: the same) | KL_prcp maximized with `visual_sensitivity_loss_coef=0.01`, `visual_sensitivity_reference=current`; no Double Entropy terms beyond the shared entropy penalty |
| `vppo.sh` | [VPPO](../reproduction/vppo/README.md) | random patch (paper: p=0.5) | TGF `top_perception_quantile=0.4` per response; TAS `response_advantage_scaling_method=vppo`, `vppo_response_scaling_min=0.9`; `visual_sensitivity_reference=old`; no 0.06 entropy penalty |
| `tor.sh` | [ToR](../reproduction/tor/README.md) | random patch (paper: image removed) | `top_entropy_quantile=0.3` and `top_perception_quantile=0.3` (batch), `tor_use_token_weighting=true`, `tor_rsn_weight=1.0`, `tor_prcp_weight=0.5`; `visual_sensitivity_reference=old` |
| `dvrp.sh` | [DVRP](../reproduction/dvrp/README.md) | random patch + `vp_diffusion` noise (paper: the same) | `visual_sensitivity_loss_coef=0.01` (λ_nec), `visual_robustness_loss_coef=0.01` (λ_rob), `noise_t_init=500`, `noise_gamma=10`, `noise_t_max=1000`; no view-entropy terms |
| `pgpo.sh` | [PGPO](../reproduction/pgpo/README.md) | random patch (paper: attention to visual tokens masked) | `advantage_scaling_method=pgpo`, `pgpo_token_scaling_threshold=0.4`, `pgpo_token_scaling_boost=2.0`, `visual_sensitivity_metric=sampled_low_var_kl`, `visual_sensitivity_reference=old` |
| `pepo.sh` | [PEPO](../reproduction/pepo/README.md) | none (single forward) | `visual_sensitivity_metric=hidden_state_similarity` (cosine), `advantage_scaling_method=pepo`, `advantage_scaling_schedule=linear`, `pepo_gate_alpha=0.05`, `pepo_gate_temperature=1.8` |
| `cfpo.sh` | [CFPO](../reproduction/cfpo/README.md) | in-model attention-value counterfactual (paper: the same) | `corrupt_image=cross_modal_attention_value_mean` (λ=2), `visual_sensitivity_loss_coef=0.02`, `visual_sensitivity_reference=current` |
| `vepo.sh` | [VEPO](../reproduction/vepo/README.md) | random patch (paper: Gaussian noise) | `visual_sensitivity_metric=vepo`, `visual_sensitivity_jsd_weight=0.7`, `visual_sensitivity_entropy_gate=normal_entropy`, `top_perception_quantile=0.2` per response, `normalize_pg_loss_by_selected_tokens=true` |
| `grit.sh` | [GRIT](../reproduction/grit/README.md) | – | think / rethink / answer format with inline JSON evidence boxes (`examples/system_prompt/grit_GR.txt`, no format prompt) and `grit.py:compute_score` (format, box format, rule-based answer, 0.1·BLEU-1); trained on ViRL39K instead of GRIT's 20 samples |
| `cgpo.sh` | CGPO ([examples/reproduction/cgpo](../reproduction/cgpo/README.md)) | evidence regions flattened to their local mean (`cgpo_flat`, one per response) | `<region>` evidence format (`xml_grounded_reasoning.jinja` and reward, format weight 0.1); `top_entropy_quantile=0.3`, `top_perception_quantile=0.3` (batch), `include_region_tokens_in_perception_mask=true`; `advantage_scaling_method=cgpo`, `cgpo_response_scaling_coef=0.1`; grounding-consistency reward weight 0.1; `visual_sensitivity_reference=old` |

Because the recipe is shared, method-specific settings that the papers tune together with their
base algorithm (DAPO dynamic sampling and clip-higher, VPPO's entropy penalty, PAPO's and DVRP's
view-entropy terms, VEPO's frozen vision tower) are not used here. The per-paper settings are in
[examples/reproduction/](../reproduction/).

## Running

```bash
bash scripts/prepare_data.sh comparison
bash examples/comparison/qwen3_vl_4b/papo.sh                      # one method
for m in grpo papo vppo tor dvrp pgpo pepo cfpo vepo grit cgpo; do
    bash examples/comparison/qwen3_vl_4b/$m.sh                    # all methods, one after another
done
bash scripts/prepare_eval_data.sh comparison
bash scripts/eval.sh checkpoints/Comparison-Qwen3-VL-4B/papo --suite comparison
```

Checkpoints go to `checkpoints/Comparison-Qwen3-VL-4B/<method>`. Any `key=value` appended to a
script overrides the shared setting (e.g. `bash examples/comparison/qwen3_vl_4b/papo.sh trainer.total_epochs=1`);
`N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` and `EXPERIMENT_NAME` are read from the
environment. `grit` and `cgpo` answer in their own formats, so evaluate them with the matching
prompt: `--format-prompt none --system-prompt examples/system_prompt/grit_GR.txt` for `grit`,
`--format-prompt examples/format_prompt/xml_grounded_reasoning.jinja` for `cgpo`. See
[eval/README.md](../../eval/README.md).

## Results

Suite `comparison`. Reasoning benchmarks use the PAPO-Eval protocol (rule-based match on the last
`\boxed{}`, avg@8 at T=1.0, top-p 1.0; DynaMath and MathVision from VPPO-Eval); POPE is greedy
macro-F1 over the random / popular / adversarial splits; HallusionBench is greedy question accuracy.
Avg is the unweighted mean of the 13 columns. MMK12 test is also the validation set monitored during
training (as in PAPO and VPPO).

| Method | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVerse-V | LogicVista | Counting | MMMU-Pro | DynaMath | MathVision | POPE | HallusionBench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct (no RL) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DAPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VPPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ToR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DVRP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PGPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PEPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CFPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VEPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRIT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CGPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Results will be added after the runs finish.

> [!IMPORTANT]
> Except for CGPO, every method in this table is our unofficial re-implementation, run in one shared
> setting (backbone, data and hyper-parameters chosen for the comparison, not for each method), on
> our hardware and software versions, with a single seed per method. Differences of a point or less
> between methods can come from training randomness alone. A method that does not beat GRPO here is
> not shown to be ineffective: the original papers remain the reference for each method's
> performance. See [About the results](../../README.md#-about-the-results); corrections from the
> authors are welcome.
