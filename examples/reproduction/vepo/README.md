# VEPO — Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection

EMNLP 2026 Findings (per a co-author's homepage) · [arXiv:2606.03937](https://arxiv.org/abs/2606.03937) ·
[official code](https://github.com/Leonnnnnn929/VEPO) ·
**Unofficial re-implementation in this repository.**

## Method

VEPO selects which tokens receive policy gradients. For each response it runs a second
teacher-forced pass with a perturbed image and computes, per token, the Jensen-Shannon divergence
between the two output distributions, the absolute entropy gap between them, and the token
entropy. After per-response min-max normalization the score is a soft noisy-OR of the two visual
signals (weight α on the JSD) multiplied by the entropy. Only the top-k fraction of tokens of each
response (k = 0.2) is optimized, with the GRPO loss averaged over the selected tokens; the
sequence-level advantage is unchanged.

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct |
| Training data | Geometry3K train (`hiyouga/geometry3k`, 2,101) — `bash scripts/prepare_data.sh vepo` |
| Validation | Geometry3K test (601), every 20 steps, 8 samples at T=1.0, top-p 0.9 |
| Prompt / reward | `<think>` + `\boxed{}` (`math_perception.jinja`); binary accuracy (`math.py:compute_score_wo_format`) |
| Rollout | 512 prompts x 12 rollouts per step, T=1.0, top-p 1.0; update batch 128 prompts; vLLM tensor parallel 4 |
| RL | GRPO; low-var KL loss 0.01; clip 0.2 / 0.3 (EasyR1 defaults); token-level loss |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower frozen, 4 epochs |
| Lengths / pixels | max prompt 8,192, max response 4,096; 200,704-1,003,520 pixels |
| GPUs | 8 |
| Perturbed view | `algorithm.corrupt_image=gaussian_noise`, `algorithm.corrupt_image_kwargs={"std":2.0}`, `algorithm.corrupt_image_position=prompt` |
| Token score | `algorithm.visual_sensitivity_metric=vepo`, `algorithm.visual_sensitivity_jsd_weight=0.7` (α), `algorithm.visual_sensitivity_entropy_gate=normal_entropy` |
| Token selection | `algorithm.top_perception_quantile=0.2` (k), `algorithm.perception_thr_granularity=response`, `algorithm.normalize_pg_loss_by_selected_tokens=true` |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_grpo.sh` | GRPO baseline (all tokens), 7B |
| `qwen2_5_vl_7b_grpo_entropy.sh` | GRPO on the top-20% entropy tokens of each response (the paper's entropy baseline), 7B |
| `qwen2_5_vl_7b_grpo_vepo.sh` | VEPO, 7B |

```bash
bash scripts/prepare_data.sh vepo
bash examples/reproduction/vepo/qwen2_5_vl_7b_grpo_vepo.sh
bash scripts/prepare_eval_data.sh vepo
bash scripts/eval.sh checkpoints/VEPO-Reproduce/qwen2_5_vl_7b_grpo_vepo --suite vepo
```

Checkpoints go to `checkpoints/VEPO-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `algorithm.disable_kl=true`) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`,
`LOGGER` or `EXPERIMENT_NAME` in the environment. Evaluation is described in
[eval/README.md](../../../eval/README.md).

## Differences from the paper / official code

- **Training data.** The paper trains on about 4.2K samples drawn from Geometry3K and MMK12; this
  set is not released (the official script uses a placeholder path). We train on the 2,101
  Geometry3K training problems and validate on the Geometry3K test split instead of
  `xyliu6/k12-freeform` (test, 808).
- **RL recipe.** The official recipe is GRPO without KL, symmetric clip 0.2, max response 2,048,
  `total_episodes=20`, reward 0.9 accuracy + 0.1 format, and an entropy bonus of 1e-3 that the
  paper does not mention. Our scripts use a low-var KL loss 0.01, clip 0.2 / 0.3, max response
  4,096, 4 epochs, an accuracy-only reward and no entropy bonus.
- **Perturbation.** We add N(0, 2²) noise to the image in [0, 1] pixel space and clip to [0, 1]. The
  official code adds N(0, 2²) noise to the processor-normalized pixel values without clipping; the
  paper describes diffusion-step-500 noise with a sigmoid decay schedule.
- The 3B setting and the retrained baselines of Table 1 (NoisyRollout, PAPO-DAPO, VPPO, R1-ShareVL,
  top-40% entropy) are not scripted; see [examples/comparison/](../../comparison/README.md) for a controlled
  comparison of the perception-aware methods.
- **Evaluation** (suite `vepo`) uses the PAPO-Eval protocol (rule-based `\boxed{}` match, avg@8,
  T=1.0) instead of greedy decoding with a gpt-4o-mini judge for MathVista, MathVerse, MathVision
  and We-Math; MMK12 is the PAPO-Eval test split. Use `--temperature 0 --num-samples 1` for greedy
  decoding.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 1 (Qwen2.5-VL-7B trained on the 4.2K Geometry3K + MMK12 mix; greedy accuracy, %).

| Script | Geo3K | MMK12 | HalluBench | MathVista | We-Math | MathVerse | MathVision | Paper Avg | This repo (Avg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_7b_grpo` | 49.42 | 65.18 | 70.45 | 71.60 | 67.18 | 48.53 | 27.40 | 57.11 | TBD |
| `qwen2_5_vl_7b_grpo_entropy` | 47.59 | 64.93 | 68.56 | 70.40 | 67.13 | 48.25 | 28.95 | 56.54 | TBD |
| `qwen2_5_vl_7b_grpo_vepo` | 51.58 | 69.64 | 71.71 | 72.00 | 69.54 | 48.93 | 28.31 | 58.82 | TBD |

Because the training data and evaluation protocol differ, compare our runs with each other rather
than with these numbers. Results of this repository will be added after the reproduction runs
finish.

## Citation

```bibtex
@article{jin2026vepo,
  title   = {Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection},
  author  = {Jin, Senjie and Wang, Peixin and Liu, Boyang and Fan, Xiaoran and Li, Shuo and Xi, Zhiheng and Zhang, Jiazheng and Zhou, Yuhao and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  journal = {arXiv preprint arXiv:2606.03937},
  year    = {2026},
  note    = {Findings of EMNLP 2026 (per author homepage)}
}
```
