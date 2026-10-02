# VPPO — Spotlight on Token Perception for Multimodal Reinforcement Learning

ICLR 2026 · [arXiv:2510.09285](https://arxiv.org/abs/2510.09285) ·
[official code](https://github.com/huaixuheqing/VPPO-RL) ·
**Unofficial re-implementation in this repository.**

## Method

VPPO measures the *visual dependency* of every generated token as the KL between the policy's
predictions with the original image and with a perturbed image (random 14-px patches blackened with
p=0.5). It then (1) applies **Token-level Gradient Filtering (TGF)**: only the top-k (k=0.4)
most visually dependent tokens of each response receive policy gradients, and (2) applies
**Trajectory-level Advantage Shaping (TAS)**: each response's advantage is scaled by a factor in
[β_min, β_max] that grows with the response's mean visual dependency. It is built on DAPO.

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-7B-Instruct; Qwen3-VL-8B-Instruct (the post-paper VPPO-8B recipe) |
| Training data | ViRL39K as processed by PAPO (byte-identical to `chamber111/VPPO_ViRL39K_train`, 38,870) — `bash scripts/prepare_data.sh vppo` |
| Validation | MMK12 test (2,000; identical to `chamber111/VPPO_MMK12_validation`), every 5 steps, 8 samples at T=1.0, top-p 0.99 |
| Prompt / reward | `<think>` + `\boxed{}` (`math_perception.jinja`); binary accuracy, no format term (`math.py:compute_score_wo_format`) |
| Rollout | 384 prompts x 8 rollouts per step, T=1.0, top-p 0.99; update batch 128 prompts |
| RL (DAPO recipe) | no KL; clip 0.2 / 0.28; online filtering on accuracy in (0.01, 0.99) with unlimited regeneration (`trainer.max_try_make_batch=-1`); token-level loss |
| Entropy penalty | 0.06 on sampled tokens for both baseline and VPPO (`algorithm.invariant_entropy_coef=0.06`, `algorithm.entropy_loss_type=sampled`); 0.12 for Qwen3-VL-8B |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 2 epochs (Qwen3-VL-8B: `trainer.max_steps=130`) |
| Lengths / pixels | max prompt 4,096; max response 2,048 (Qwen3-VL-8B: 8,192); 200,704-1,003,520 pixels |
| GPUs | 8 |
| Perturbed view | `algorithm.corrupt_image=random_patch`, `algorithm.corrupt_image_kwargs={"patch_size":14,"black_prob":0.5}` (patch 16 for Qwen3-VL), `algorithm.corrupt_image_position=response` |
| TGF | `algorithm.top_perception_quantile=0.4`, `algorithm.perception_thr_granularity=response` |
| TAS | `algorithm.response_advantage_scaling_method=vppo`, `algorithm.vppo_response_scaling_min=0.9` (β_min; β_max is dynamic) |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_dapo.sh` | DAPO baseline (with the 0.06 entropy penalty), Qwen2.5-VL-7B |
| `qwen2_5_vl_7b_dapo_vppo.sh` | VPPO, Qwen2.5-VL-7B |
| `qwen3_vl_8b_dapo.sh` | DAPO baseline (0.12 entropy penalty), Qwen3-VL-8B |
| `qwen3_vl_8b_dapo_vppo.sh` | VPPO, Qwen3-VL-8B (VPPO-8B recipe) |

```bash
bash scripts/prepare_data.sh vppo
bash examples/reproduction/vppo/qwen2_5_vl_7b_dapo_vppo.sh
bash scripts/prepare_eval_data.sh vppo
bash scripts/eval.sh checkpoints/VPPO-Reproduce/qwen2_5_vl_7b_dapo_vppo --suite vppo
```

Checkpoints go to `checkpoints/VPPO-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `trainer.total_epochs=1`) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`,
`LOGGER` or `EXPERIMENT_NAME` in the environment (e.g. `N_GPUS_PER_NODE=4 bash ...`). The
Qwen3-VL-8B runs are trained with 8,192-token responses while the suite default is 2,048 new
tokens; use `--max-new-tokens` to change it (see [eval/README.md](../../../eval/README.md)).

## Differences from the paper / official code

- The visual dependency uses the sampled-token k3 estimator ("low_var_kl"), as in the official
  code, rather than the full-distribution KL written in the paper (Eq. 3).
- TAS min-max normalization uses the minimum and maximum over the whole rollout batch; the
  official code uses per-device mini-batch statistics (its cross-device sync is commented out).
- The 32B setting is not included.
- Qwen3-VL-8B: the model card describes 150 steps with entropy coefficient 0.12 for steps 0-130
  and 0.18 for steps 131-150; the official script and ours stop at `max_steps=130` with 0.12. The
  8B DAPO baseline has no official counterpart.
- Evaluation uses this repository's harness with the VPPO-Eval / PAPO-Eval splits and the PAPO-Eval
  protocol (rule-based `\boxed{}` match, avg@8, T=1.0, top-p 1.0), not PAPO-Eval itself.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 1 (avg@8 accuracy, %). The Qwen3-VL-8B row comes from the official README figure
(not in the paper; no 8B baseline is given).

| Script | MathVerse | DynaMath | MMK12 | Geo3k | MathVision | We-Math | LogicVista | MMMU-Pro | Paper Avg | This repo (Avg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_7b_dapo` | 68.3 | 66.6 | 82.1 | 41.5 | 30.5 | 68.0 | 46.8 | 35.9 | 55.0 | TBD |
| `qwen2_5_vl_7b_dapo_vppo` | 71.6 | 68.1 | 82.8 | 46.5 | 33.3 | 71.5 | 47.9 | 37.9 | 57.5 | TBD |
| `qwen3_vl_8b_dapo` | – | – | – | – | – | – | – | – | – | TBD |
| `qwen3_vl_8b_dapo_vppo` | 83.8 | 80.0 | 80.6 | 71.9 | 55.9 | 84.3 | 62.3 | 52.4 | 71.4 | TBD |

For reference, the paper's GRPO-7B average is 53.1 and the base Qwen2.5-VL-7B scores 38.3. Results
of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{huang2025spotlight,
  title     = {Spotlight on Token Perception for Multimodal Reinforcement Learning},
  author    = {Huang, Siyuan and Qu, Xiaoye and Li, Yafu and Luo, Yun and He, Zefeng and Liu, Daizong and Cheng, Yu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv preprint arXiv:2510.09285 (2025)}
}
```
