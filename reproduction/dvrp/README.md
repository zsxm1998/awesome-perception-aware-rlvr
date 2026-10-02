# DVRP — Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy

arXiv preprint · [arXiv:2601.06801](https://arxiv.org/abs/2601.06801) · no official code released ·
**Unofficial re-implementation in this repository.**

## Method

DVRP builds a "visual triplet" for every image: the original image, a masked view (random patch
masking, "−Δ") and a noised view (variance-preserving diffusion noise with a sigmoid-annealed
timestep, "+Δ"). On top of GRPO (DVRP_G) or DAPO (DVRP_D) it **maximizes** the KL between the
original-view and masked-view policies (λ_nec, the answer must need the image), **minimizes** the KL
between the original-view and noised-view policies (λ_rob, the answer should be robust to small
changes), and adds an entropy penalty on both auxiliary views (λ_ent) (paper Eq. 4).

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct |
| Training data | ViRL39K as processed by PAPO (38,870) — `bash scripts/prepare_data.sh dvrp` |
| Validation | MMK12 test (2,000), every 5 steps, 8 samples at T=1.0, top-p 0.99 |
| Prompt / reward | `<think>` + `\boxed{}` (`math_perception.jinja`); binary accuracy (`math.py:compute_score_wo_format`) |
| Rollout | 384 prompts x 5 rollouts per step, T=1.0, top-p 0.99; update batch 128 prompts |
| RL | no KL to the reference model (GRPO and DAPO arms); GRPO arms clip 0.2 / 0.3; DAPO arms clip 0.2 / 0.28, online filtering on accuracy in (0.01, 0.99), `data.mini_rollout_batch_size=128` |
| Optimization | AdamW (bf16), lr 1e-6 constant, token-level loss, vision tower trainable, 3 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels |
| GPUs | 4 |
| Masked view (−Δ) | `algorithm.corrupt_image=random_patch`, `algorithm.corrupt_image_kwargs={"patch_size":14,"black_prob":0.6}`, `algorithm.corrupt_image_position=prompt` |
| Noised view (+Δ) | `algorithm.incremental_image_transform=vp_diffusion`, `algorithm.noise_t_init=500`, `algorithm.noise_gamma=10`, `algorithm.noise_t_max=1000` |
| λ_nec / λ_rob | `algorithm.visual_sensitivity_loss_coef=0.01` / `algorithm.visual_robustness_loss_coef=0.01` |
| λ_ent | `algorithm.decremental_entropy_coef=0.05`, `algorithm.incremental_entropy_coef=0.05`, `algorithm.entropy_loss_type=sampled` |

See [docs/algorithm_parameters.md](../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_grpo.sh` / `qwen2_5_vl_7b_grpo.sh` | GRPO baseline (no KL), 3B / 7B |
| `qwen2_5_vl_3b_grpo_dvrp.sh` / `qwen2_5_vl_7b_grpo_dvrp.sh` | DVRP_G, 3B / 7B |
| `qwen2_5_vl_3b_dapo.sh` / `qwen2_5_vl_7b_dapo.sh` | DAPO baseline, 3B / 7B |
| `qwen2_5_vl_3b_dapo_dvrp.sh` / `qwen2_5_vl_7b_dapo_dvrp.sh` | DVRP_D, 3B / 7B |

```bash
bash scripts/prepare_data.sh dvrp
bash reproduction/dvrp/qwen2_5_vl_7b_grpo_dvrp.sh
bash scripts/prepare_eval_data.sh dvrp
bash scripts/eval.sh checkpoints/DVRP-Reproduce/qwen2_5_vl_7b_grpo_dvrp --suite dvrp
```

Checkpoints go to `checkpoints/DVRP-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `trainer.total_epochs=1`) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`,
`LOGGER` or `EXPERIMENT_NAME` in the environment (e.g. `N_GPUS_PER_NODE=8 bash ...`). Evaluation is
described in [eval/README.md](../../eval/README.md).

## Differences from the paper

- No official code exists; the method is re-implemented from the paper text. Values the paper
  does not give are our choices: diffusion `T_max=1000`, training max response length 2,048, clip
  ratios (EasyR1 defaults for GRPO, 0.2 / 0.28 for DAPO) and the MMK12 validation set.
- Only the math / general-domain half is reproduced. The medical training composite (Slake,
  PathVQA, VQA-RAD, PMC-VQA) is released without dataset ids, so the medical columns and the
  medical settings (P_mask 0.2, T_init 100) are not included.
- The paper sums token-level KL divergences over the full output distributions; both KL terms here
  use the sampled-token k3 estimator.
- The masked and noised views are scored once with the rollout policy before the update (no
  gradient through those branches, as in PAPO's default implementation). The KL terms therefore
  only move the original-view policy, and the two view-entropy terms (λ_ent) do not contribute
  gradients.
- Evaluation (suite `dvrp`) covers the six general-domain benchmarks with the PAPO-Eval protocol
  (avg@8, T=1.0, top-p 1.0); the paper reports top-p 0.9 in the text and 0.99 in its tables.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../README.md#-about-the-results).

Paper Table 1, general-domain columns (avg@8 accuracy, %; "MMK12" is printed as "MMKI2").

| Script | Geo3k | MathVista | WeMath | MathVerse | MathVerse-V | MMK12 | Paper Gen AVG | This repo (Gen AVG) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_3b_grpo` | 28.7 | 59.3 | 58.9 | 55.3 | 52.2 | 57.2 | 51.9 | TBD |
| `qwen2_5_vl_3b_grpo_dvrp` | 34.5 | 65.5 | 60.3 | 57.7 | 54.5 | 61.2 | 55.6 | TBD |
| `qwen2_5_vl_3b_dapo` | 31.2 | 60.9 | 60.0 | 56.3 | 53.0 | 66.8 | 54.7 | TBD |
| `qwen2_5_vl_3b_dapo_dvrp` | 35.1 | 64.9 | 60.5 | 58.1 | 54.8 | 60.9 | 55.7 | TBD |
| `qwen2_5_vl_7b_grpo` | 40.2 | 65.5 | 66.1 | 66.5 | 61.7 | 72.1 | 62.0 | TBD |
| `qwen2_5_vl_7b_grpo_dvrp` | 42.3 | 71.1 | 68.1 | 67.4 | 66.7 | 75.6 | 65.2 | TBD |
| `qwen2_5_vl_7b_dapo` | 35.9 | 61.9 | 58.5 | 55.6 | 51.0 | 71.9 | 55.8 | TBD |
| `qwen2_5_vl_7b_dapo_dvrp` | 43.4 | 70.9 | 67.8 | 68.9 | 65.3 | 74.1 | 65.1 | TBD |

Note: the paper's general-domain GRPO and DAPO rows match PAPO's Table 1 within 0.1 in most cells
although the paper says all baselines were reproduced; compare against our own baseline runs rather
than across papers. Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{gao2026thinking,
  title={Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy},
  author={Gao, Shujian and Wang, Yuan and Yan, Jiangtao and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2601.06801},
  year={2026}
}
```
