# PGPO — Not All Tokens See Equally: Perception-Grounded Policy Optimization for Large Vision-Language Models

arXiv preprint · [arXiv:2604.01840](https://arxiv.org/abs/2604.01840) · no official code released
(the announced repository `github.com/Yzk1114/PGPO` was not public as of 2026-10-01) ·
**Unofficial re-implementation in this repository.**

## Method

PGPO reshapes the advantage of each token according to how much the image changes that token's
prediction. A second "blind" forward pass masks the attention to all visual tokens; the token
visual dependency S_t is the k3 estimate of KL(π(·|s_t, I) ‖ π(·|s_t)) at the sampled token. S_t is
compressed with log(1+S), min-max normalized within each response to I_t ∈ [0, 1], and mapped to a
threshold-gated weight (I/τ below τ, 1 + β(I−τ)/(1−τ) above). The weights are renormalized to keep
their sum equal to the response length, and multiply the DAPO advantage token by token.

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct |
| Training data | ViRL39K as processed by PAPO (38,870) — `bash scripts/prepare_data.sh pgpo` |
| Validation | MMK12 test (2,000), every 5 steps, 8 samples at T=1.0, top-p 0.99 |
| Prompt / reward | `<think>` + `\boxed{}` (`math_perception.jinja`); 0.9 accuracy + 0.1 format (`math.py:compute_score`) |
| Rollout | 384 prompts x 5 rollouts per step (generated in chunks of 128 prompts), T=1.0, top-p 0.99; update batch 128 prompts |
| RL (DAPO recipe) | no KL; clip 0.2 / 0.28 with the default dual-clip bound 3.0; online filtering on accuracy in (0.01, 0.99); token-level loss |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 2 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels |
| GPUs | 4 |
| Blind pass | `algorithm.corrupt_image=mask_visual_attention`, `algorithm.corrupt_image_position=prompt` |
| Visual dependency | `algorithm.visual_sensitivity_metric=sampled_low_var_kl`, `algorithm.visual_sensitivity_reference=old` |
| Token scaling | `algorithm.advantage_scaling_method=pgpo`, `algorithm.pgpo_token_scaling_threshold=0.4` (τ), `algorithm.pgpo_token_scaling_boost=2.0` (β) |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_dapo.sh` | DAPO baseline, 3B |
| `qwen2_5_vl_3b_dapo_pgpo.sh` | PGPO, 3B |
| `qwen2_5_vl_7b_dapo.sh` | DAPO baseline, 7B |
| `qwen2_5_vl_7b_dapo_pgpo.sh` | PGPO, 7B |

```bash
bash scripts/prepare_data.sh pgpo
bash examples/reproduction/pgpo/qwen2_5_vl_7b_dapo_pgpo.sh
bash scripts/prepare_eval_data.sh pgpo
bash scripts/eval.sh checkpoints/PGPO-Reproduce/qwen2_5_vl_7b_dapo_pgpo --suite pgpo
```

Checkpoints go to `checkpoints/PGPO-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `trainer.total_epochs=1`) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`,
`LOGGER` or `EXPERIMENT_NAME` in the environment (e.g. `N_GPUS_PER_NODE=8 bash ...`). Evaluation is
described in [eval/README.md](../../../eval/README.md).

## Differences from the paper

- The official code is not released; the method is re-implemented from the paper. It follows the
  paper rather than an unlinked development snapshot by the authors (`Yzk1114/EasyR1`, method
  "vig"): τ is a fixed threshold on the normalized score (the snapshot uses a per-sequence
  quantile) and the sum-preserving normalization is enabled (the snapshot comments it out).
- Online filtering uses the accuracy reward; the paper lists "Overall" as the filter key.
- The paper uses H100 clusters (2 GPUs in its 3B overhead study); our scripts use 4 GPUs for both
  model sizes.
- PGPO on top of GRPO (paper App. B) and the general-benchmark table (MMBench / MMStar / MME) are
  not scripted.
- Evaluation (suite `pgpo`) uses the PAPO-Eval protocol. DynaMath is the VPPO-Eval file (3,666
  verifiable items across all variants) instead of "sample variant1", and MathVision is the
  2,907-item verifiable subset from VPPO-Eval instead of the full test split.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 1 (avg@8 accuracy, %). The paper reports its 7B DAPO baseline from the best
pre-collapse checkpoint (step 190 of 202).

| Script | Geo3k | MMK12 | MathVerse | DynaMath | MathVision | LogicVista | MMMU-Pro | MathVerse_V | Paper Avg | This repo (Avg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_3b_dapo` | 33.80 | 63.36 | 58.76 | 45.86 | 26.19 | 40.63 | 27.20 | 54.89 | 43.84 | TBD |
| `qwen2_5_vl_3b_dapo_pgpo` | 36.11 | 64.74 | 62.06 | 48.45 | 26.90 | 43.20 | 29.33 | 57.24 | 46.00 | TBD |
| `qwen2_5_vl_7b_dapo` | 41.47 | 78.74 | 67.95 | 55.96 | 28.16 | 47.85 | 36.99 | 64.38 | 52.69 | TBD |
| `qwen2_5_vl_7b_dapo_pgpo` | 45.20 | 80.83 | 71.45 | 57.71 | 29.02 | 47.93 | 39.01 | 66.41 | 54.70 | TBD |

The abstract's "18.7% on average" is the absolute gain over the base model; the gain over DAPO is
+2.16 (3B) and +2.01 (7B). Results of this repository will be added after the reproduction runs
finish.

## Citation

```bibtex
@article{ye2026notall,
  title={Not All Tokens See Equally: Perception-Grounded Policy Optimization for Large Vision-Language Models},
  author={Ye, Zekai and Li, Qiming and Feng, Xiaocheng and Chen, Ruihan and Li, Ziming and Ren, Haoyu and Chen, Kun and Tu, Dandan and Qin, Bing},
  journal={arXiv preprint arXiv:2604.01840},
  year={2026}
}
```
