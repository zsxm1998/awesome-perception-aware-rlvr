# PAPO — Perception-Aware Policy Optimization for Multimodal Reasoning

ICLR 2026 (per the official README) · [arXiv:2507.06448](https://arxiv.org/abs/2507.06448) ·
[official code](https://github.com/MikeWangWZHL/PAPO) (evaluation: [PAPO-Eval](https://github.com/xhguo7/PAPO-Eval)) ·
**Unofficial re-implementation in this repository.**

## Method

PAPO adds an *Implicit Perception Loss* to GRPO/DAPO: for every rollout it builds a masked copy of
the image (random 14-px patches blackened with probability 0.6) and **maximizes**
KL(π(o | q, I) ‖ π(o | q, I_mask)) with weight γ, so that the policy's outputs depend on the
visual input. Because maximizing this KL can be "hacked", PAPO adds a *Double Entropy Loss* (an
entropy penalty on the original and on the masked view, weight η). The variants are PAPO-G (on
GRPO) and PAPO-D (on DAPO).

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct |
| Training data | ViRL39K as processed by PAPO (`PAPOGalaxy/PAPO_ViRL39K_train`, 38,870) — `bash scripts/prepare_data.sh papo` |
| Validation | MMK12 test (`PAPOGalaxy/PAPO_MMK12_test`, 2,000), every 5 steps, 8 samples at T=1.0, top-p 0.99 (monitoring only) |
| Prompt / reward | `<think>` reasoning + `\boxed{}` answer (`examples/format_prompt/math_perception.jinja`); 0.9 accuracy + 0.1 format (`examples/reward_function/math.py:compute_score`) |
| Rollout | 384 prompts x 5 rollouts per step, T=1.0, top-p 0.99; update batch 128 prompts |
| Optimization | AdamW (bf16), lr 1e-6 constant, weight decay 1e-2, token-level loss averaging, vision tower trainable, 2 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels per image |
| GRPO arms | low-var KL loss 0.01; clip 0.2 / 0.3 |
| DAPO arms | no KL; clip 0.2 / 0.28; online filtering on accuracy in (0.01, 0.99), `data.mini_rollout_batch_size=128`, at most 20 generation rounds |
| GPUs | 2 (3B), 4 (7B) |
| Masked view | `algorithm.corrupt_image=random_patch`, `algorithm.corrupt_image_kwargs={"patch_size":14,"black_prob":0.6}`, `algorithm.corrupt_image_position=prompt` |
| γ (KL_prcp) | `algorithm.visual_sensitivity_loss_coef`: 0.02 (PAPO-G), 0.01 (PAPO-D and no-KL-ref) |
| η (Double Entropy) | `algorithm.invariant_entropy_coef` = `algorithm.decremental_entropy_coef`: 0.05 (PAPO-G-7B), 0.03 (PAPO-D, no-KL-ref), off (PAPO-G-3B); `algorithm.entropy_loss_type=sampled` |

See [docs/algorithm_parameters.md](../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_grpo.sh` | GRPO baseline, 3B |
| `qwen2_5_vl_3b_grpo_papo.sh` | PAPO-G, 3B (γ=0.02, no entropy terms) |
| `qwen2_5_vl_3b_grpo_papo_no_kl_ref.sh` | PAPO-G without reference KL, 3B (γ=0.01, η=0.03; paper App. F) |
| `qwen2_5_vl_3b_dapo.sh` | DAPO baseline, 3B |
| `qwen2_5_vl_3b_dapo_papo.sh` | PAPO-D, 3B (γ=0.01, η=0.03) |
| `qwen2_5_vl_7b_grpo.sh` | GRPO baseline, 7B |
| `qwen2_5_vl_7b_grpo_papo.sh` | PAPO-G, 7B (γ=0.02, η=0.05) |
| `qwen2_5_vl_7b_grpo_papo_no_kl_ref.sh` | PAPO-G without reference KL, 7B (γ=0.01, η=0.03) |
| `qwen2_5_vl_7b_dapo.sh` | DAPO baseline, 7B |
| `qwen2_5_vl_7b_dapo_papo.sh` | PAPO-D, 7B (γ=0.01, η=0.03) |

```bash
bash scripts/prepare_data.sh papo
bash reproduction/papo/qwen2_5_vl_7b_grpo_papo.sh
bash scripts/prepare_eval_data.sh papo
bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo
```

Checkpoints go to `checkpoints/PAPO-Reproduce/<script name>`. Any `key=value` appended to the
command overrides the script (e.g. `bash reproduction/papo/qwen2_5_vl_7b_grpo_papo.sh trainer.total_epochs=1`),
and `N_GPUS_PER_NODE=4 bash ...`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` and `EXPERIMENT_NAME` are read
from the environment. The "GRPO + No KL_ref" baseline of App. F has no dedicated script:
`EXPERIMENT_NAME=qwen2_5_vl_3b_grpo_no_kl_ref bash reproduction/papo/qwen2_5_vl_3b_grpo.sh algorithm.disable_kl=true`.
Evaluation is described in [eval/README.md](../../eval/README.md).

## Differences from the paper / official code

- Re-implemented on EasyR1 with the shared perception-reasoning modules of this repository, not a
  copy of the official fork. Hyper-parameters follow the official scripts and paper Table 3.
- As in the official default (`RECOMPUTE_AUG_LOG_PROBS=False`), the masked-image log-probs are
  computed once before the update and are not recomputed with gradient. Losses are averaged per
  token (EasyR1 default, also what the official configs use; the paper's Eq. 2 is written per
  sequence).
- The Qwen3-VL-2B-Thinking setting (official branch `main_qwen3`) is not included.
- Evaluation uses this repository's harness with the PAPO-Eval data (`PAPO-Galaxy/PAPO_eval`) and
  protocol (rule-based match on the last `\boxed{}`, avg@8 at T=1.0, top-p 1.0, 2,048 new
  tokens), not the LLaMA-Factory-based PAPO-Eval code itself. The "Counting" column is the 200-item
  `BUAADreamer/clevr_count_70k` split shipped with PAPO-Eval.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../README.md#about-the-results).

Paper Table 1 (avg@8 accuracy, %); Gen AVG = Geo3k, MathVista, We-Math, MMK12, MathVerse; Vis AVG =
LogicVista, Counting, MMMU-Pro, MathVerse_V. No-KL-ref rows: paper Table 4 (overall only).

| Script | Paper Gen AVG | Paper Vis AVG | Paper Overall | This repo (Overall) |
| --- | --- | --- | --- | --- |
| `qwen2_5_vl_3b_grpo` | 51.89 | 42.97 | 47.92 | TBD |
| `qwen2_5_vl_3b_grpo_papo` | 53.39 | 45.57 | 49.92 | TBD |
| `qwen2_5_vl_3b_grpo_papo_no_kl_ref` | – | – | 53.15 | TBD |
| `qwen2_5_vl_3b_dapo` | 55.02 | 49.11 | 52.40 | TBD |
| `qwen2_5_vl_3b_dapo_papo` | 57.09 | 52.93 | 55.24 | TBD |
| `qwen2_5_vl_7b_grpo` | 62.51 | 54.11 | 58.78 | TBD |
| `qwen2_5_vl_7b_grpo_papo` | 63.50 | 59.37 | 61.66 | TBD |
| `qwen2_5_vl_7b_grpo_papo_no_kl_ref` | – | – | 61.47 | TBD |
| `qwen2_5_vl_7b_dapo` | 57.58 | 51.79 | 55.01 | TBD |
| `qwen2_5_vl_7b_dapo_papo` | 65.83 | 59.82 | 63.16 | TBD |

The paper notes that DAPO-7B collapses late in training. Results of this repository will be added
after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{wang2025perception,
  title     = {Perception-Aware Policy Optimization for Multimodal Reasoning},
  author    = {Wang, Zhenhailong and Guo, Xuehang and Stoica, Sofia and Xu, Haiyang and Wang, Hongru and Ha, Hyeonjeong and Chen, Xiusi and Chen, Yangyi and Yan, Ming and Huang, Fei and Ji, Heng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv preprint arXiv:2507.06448 (2025); ICLR 2026 acceptance per official GitHub README}
}
```
