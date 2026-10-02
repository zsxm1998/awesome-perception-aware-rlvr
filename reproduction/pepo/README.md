# PEPO — Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought

arXiv preprint · [arXiv:2603.22847](https://arxiv.org/abs/2603.22847) ·
[official code](https://github.com/xzxxntxdy/PEPO) ·
**Unofficial re-implementation in this repository.**

## Method

PEPO (Perception-Exploration Policy Optimization) reweights token advantages without any extra
forward pass. A token's perception prior VS_t is the cosine similarity between its hidden state and
the hidden states of the vision tokens, averaged over vision tokens and layers; its exploration
signal is the token entropy H_t. Both are min-max normalized per response and combined into a gate
1 + α·tanh(·); the gated perception scores go through a softmax over the response (scaled to mean
1) to give token weights w_t. The token advantage is [(1−λ) + λ·w_t]·A, with λ increasing linearly
from 0 to 1 over training. Variants: PEPO_G (on GRPO) and PEPO_D (on DAPO).

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-3B-Instruct, InternVL3-2B-Instruct |
| Training data | Geometry3K train (`hiyouga/geometry3k`, 2,101) — `bash scripts/prepare_data.sh pepo` |
| Validation | Geometry3K test (601), 8 samples at T=1.0, top-p 1.0 (`trainer.val_freq=400` in `common.sh`, 25 in the DAPO / PEPO_D / InternVL leaves) |
| Prompt / reward | `<think>…</think><answer>…</answer>` (`examples/format_prompt/r1v.jinja`); 0.5 accuracy + 0.5 format (`examples/reward_function/r1v.py`) |
| Rollout | 64 prompts x 8 rollouts per step (`data.rollout_batch_size=64`), T=1.0, top-p 1.0 |
| RL | low-var KL loss 0.001; clip 0.2 / 0.2; GRPO arms: sequence-level loss averaging (`worker.actor.loss_avg_mode=seq`); DAPO arms: token-level averaging and online filtering on accuracy in (0.01, 0.99) |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 1 epoch |
| Lengths / pixels | max prompt 1,024, max response 1,024; 200,704-1,003,520 pixels |
| GPUs | 8 |
| Perception prior | `algorithm.visual_sensitivity_metric=hidden_state_similarity`, `algorithm.visual_sensitivity_hidden_metric=cosine`, `algorithm.visual_token` (`auto`, `<|image_pad|>` for Qwen, `<IMG_CONTEXT>` for InternVL) |
| Token weighting | `algorithm.advantage_scaling_method=pepo`, `algorithm.advantage_scaling_schedule=linear` (λ: 0 → 1), `algorithm.pepo_gate_alpha=0.05` (α), `algorithm.pepo_gate_temperature=1.8` |

See [docs/algorithm_parameters.md](../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_grpo.sh` | GRPO baseline, Qwen2.5-VL-3B |
| `qwen2_5_vl_3b_pepo.sh` | PEPO_G, Qwen2.5-VL-3B |
| `qwen2_5_vl_3b_dapo.sh` | DAPO baseline, Qwen2.5-VL-3B |
| `qwen2_5_vl_3b_pepo_d.sh` | PEPO_D, Qwen2.5-VL-3B |
| `internvl3_2b_grpo.sh` | GRPO baseline, InternVL3-2B |
| `internvl3_2b_pepo.sh` | PEPO_G, InternVL3-2B |
| `internvl3_2b_pepo_d.sh` | PEPO_D, InternVL3-2B (no InternVL DAPO baseline script) |

```bash
bash scripts/prepare_data.sh pepo
bash reproduction/pepo/qwen2_5_vl_3b_pepo.sh
bash scripts/prepare_eval_data.sh pepo_geometry
bash scripts/eval.sh checkpoints/PEPO-Reproduce/qwen2_5_vl_3b_pepo --suite pepo_geometry
```

The `pepo_geometry` suite evaluates Geometry3K-trained models with the same `<answer>` prompt they
were trained with; the `pepo` suite holds the benchmarks of the paper's ViRL39K scaling table.
Checkpoints go to `checkpoints/PEPO-Reproduce/<script name>`. Append `key=value` overrides to the
command and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the
environment. Evaluation is described in [eval/README.md](../../eval/README.md).

## Differences from the paper / official code

- Training runs on EasyR1 (FSDP + vLLM) instead of ms-swift with DeepSpeed ZeRO-2.
- The vision tower is trainable; the paper freezes it.
- The learning rate is constant; the paper uses a cosine schedule for the geometry task.
- Validation uses the Geometry3K test split (601) instead of the 300-item validation split.
- Only the geometry task is scripted (the official repository also releases only this pipeline);
  the grounding, few-shot classification, puzzle and ViRL39K-scaling settings are not included.
- The DAPO variants keep the upper clip ratio at 0.2, as in the released script (the paper's table
  lists 0.28), and allow up to 20 regeneration rounds (the paper: at most 3 resamples).
- `pepo_gate_temperature=1.8` follows the released script; the paper does not mention it.
- Evaluation scores every benchmark with rule-based exact match (falling back to `<answer>` when no
  `\boxed{}` is present) instead of the official MathVista checker; the Geometry3K validation
  column of Table 1 is not included, and LogicVista has 447 items (PEPO keeps 443 single-letter
  items).

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../README.md#-about-the-results).

Paper Table 1 (Qwen2.5-VL-3B trained on Geometry3K, avg@8 accuracy, %). The paper average also
includes Geometry3K val; InternVL3-2B rows give the paper average only.

| Script | Geo3K test | MathVista-mini | MathVerse-mini | LogicVista | Paper Avg (5) | This repo (Avg of 4) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_3b_grpo` | 23.79 | 51.56 | 40.54 | 28.30 | 32.64 | TBD |
| `qwen2_5_vl_3b_pepo` | 27.27 | 54.45 | 45.42 | 34.45 | 36.70 | TBD |
| `qwen2_5_vl_3b_dapo` | 27.00 | 53.68 | 44.44 | 33.13 | 36.18 | TBD |
| `qwen2_5_vl_3b_pepo_d` | 27.89 | 54.75 | 44.23 | 33.89 | 36.63 | TBD |
| `internvl3_2b_grpo` | – | – | – | – | 33.74 | TBD |
| `internvl3_2b_pepo` | – | – | – | – | 37.25 | TBD |
| `internvl3_2b_pepo_d` | – | – | – | – | 37.66 | TBD |

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{li2026pepo,
  title   = {Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought},
  author  = {Li, Yunheng and Kuang, Hangyi and Zhang, Hengrui and Cao, Jiangxia and Liu, Zhaojie and Hou, Qibin and Cheng, Ming-Ming},
  journal = {arXiv preprint arXiv:2603.22847},
  year    = {2026}
}
```
