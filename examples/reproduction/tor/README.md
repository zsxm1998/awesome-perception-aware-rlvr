# ToR — Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs

arXiv preprint · [arXiv:2603.25077](https://arxiv.org/abs/2603.25077) · no official code released ·
**Unofficial re-implementation in this repository.**

## Method

ToR (Token Reweighting) replaces the uniform per-token weight of the GRPO/DAPO objective with two
token sets selected per batch: *reasoning tokens* (top 30% by entropy, weight γ_r = 1.0) and
*perception tokens* (top 30% by how much the token's log-probability changes when the image is
removed, weight γ_p = 0.5). Tokens in both sets get the reasoning weight; tokens in neither set get
weight 0 and are excluded from optimization. Variants: ToR-GRPO and ToR-DAPO.

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct |
| Training data | Geometry3K train (`hiyouga/geometry3k`, 2,101) — `bash scripts/prepare_data.sh tor` |
| Validation | Geometry3K test (601), every 20 steps, 8 samples at T=1.0, top-p 0.9 |
| Prompt / reward | `<think>` + `\boxed{}` (`math_perception.jinja`); binary accuracy (`math.py:compute_score_wo_format`) |
| Rollout | 512 prompts x 12 rollouts per step, T=1.0, top-p 1.0; update batch 128 prompts; vLLM tensor parallel 4 |
| GRPO arms | low-var KL loss 0.01; clip 0.2 / 0.2 |
| DAPO arms | no KL; clip 0.2 / 0.28; online filtering on accuracy in (0.01, 0.99) with unlimited regeneration, `data.mini_rollout_batch_size=128` |
| Optimization | AdamW (bf16), lr 1e-6 constant, token-level loss, vision tower trainable, 15 epochs (60 steps), max prompt 8,192, max response 2,048, images 262,144–4,194,304 pixels |
| GPUs | 8 |
| Perception pass | `algorithm.corrupt_image=no_image` (image removed), `algorithm.visual_sensitivity_reference=old` |
| Reasoning set T_r | `algorithm.top_entropy_quantile=0.3`, `algorithm.entropy_thr_granularity=batch` (over all tokens of the rollout batch), `algorithm.entropy_top_p=0.95` (entropy of the top-p 0.95 set, Eq. 5) |
| Perception set T_p | `algorithm.top_perception_quantile=0.3`, `algorithm.perception_thr_granularity=batch`, `algorithm.visual_sensitivity_metric=sampled_abs_log_ratio` (\|log π(o_t \| I, q) − log π(o_t \| ∅, q)\|, Eq. 8) |
| Token weights | `algorithm.tor_use_token_weighting=true`, `algorithm.tor_rsn_weight=1.0` (γ_r), `algorithm.tor_prcp_weight=0.5` (γ_p) |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_grpo.sh` | GRPO baseline, 7B |
| `qwen2_5_vl_7b_grpo_tor.sh` | ToR-GRPO, 7B |
| `qwen2_5_vl_7b_dapo.sh` | DAPO baseline, 7B |
| `qwen2_5_vl_7b_dapo_tor.sh` | ToR-DAPO, 7B |

```bash
bash scripts/prepare_data.sh tor
bash examples/reproduction/tor/qwen2_5_vl_7b_grpo_tor.sh
bash scripts/prepare_eval_data.sh tor
bash scripts/eval.sh checkpoints/ToR-Reproduce/qwen2_5_vl_7b_grpo_tor --suite tor
# greedy decoding, as the paper (which also scores with a Gemini-2.0-Flash judge)
bash scripts/eval.sh checkpoints/ToR-Reproduce/qwen2_5_vl_7b_grpo_tor --suite tor --temperature 0 --num-samples 1
```

Checkpoints go to `checkpoints/ToR-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `data.val_files=data/geometry3k/validation.parquet` to validate on the 300-item
validation split) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or
`EXPERIMENT_NAME` in the environment. Evaluation is described in [eval/README.md](../../../eval/README.md).

## Differences from the paper

- No official code exists; the method is re-implemented from the paper text. The paper states
  "the default settings in EasyR1" (lr 1e-6, rollout batch 512, update batch 128, n=12); for the
  rest we use EasyR1's defaults when the paper appeared, which NoisyRollout's GRPO baseline (the
  paper's GRPO row) also uses: T=1.0, top-p 1.0, KL 0.01, max response 2,048, images of
  262,144–4,194,304 pixels and 15 epochs (60 steps on Geometry3K). The clip ratios above are our
  choices.
- Validation uses the Geometry3K test split (601) instead of the 300-item validation split.
- Both thresholds are taken over every response token of the rollout batch, before the update
  (Eq. 6, 7, 9), with the entropy and the perception score of the rollout policy. The entropy of
  the top-p 0.95 set is that of the renormalized distribution top-p sampling draws from; the paper
  does not say whether it renormalizes. Most tokens have a truncated entropy of exactly 0, so the
  threshold is often 0: we keep exactly 30% (the paper's text), drawing the tokens tied at the
  threshold uniformly at random, while Eq. 7 read literally (entropy at least the 70th
  percentile) would keep every token.
- The ToR-DAPO run on ViRL-39K and the 3B runs are not scripted.
- Evaluation (suite `tor`) uses the PAPO-Eval protocol (rule-based `\boxed{}` match, avg@8, T=1.0)
  instead of NoisyRollout's greedy decoding with a Gemini-2.0-Flash judge. MathVerse is the
  multiple-choice 2,180-item PAPO-Eval split and MathVision the 2,907-item verifiable subset from
  VPPO-Eval; HallusionBench is scored greedily (question accuracy, image split). Use
  `--temperature 0 --num-samples 1` for greedy decoding.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 4, Qwen2.5-VL-7B trained on Geometry3K (2.1K). The paper reports no average; the Avg
column is the mean of the five benchmarks. The paper's GRPO row is identical to NoisyRollout's
GRPO row.

| Script | MathVerse | MathVision | MathVista | WeMath | HallusionBench | Avg (computed) | This repo (Avg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_7b_grpo` | 50.8 | 27.3 | 70.5 | 67.4 | 69.8 | 57.16 | TBD |
| `qwen2_5_vl_7b_grpo_tor` | 53.0 | 28.6 | 71.9 | 68.9 | 72.4 | 58.96 | TBD |
| `qwen2_5_vl_7b_dapo` | 50.6 | 26.5 | 70.3 | 69.3 | 67.9 | 56.92 | TBD |
| `qwen2_5_vl_7b_dapo_tor` | 53.4 | 27.9 | 72.6 | 72.1 | 71.8 | 59.56 | TBD |

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{lu2026bridging,
  title={Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs},
  author={Lu, Jinda and Wu, Junkang and Li, Jinghan and Huang, Kexin and Yang, Shuo and Wang, Guoyin and Wu, Jiancan and Wang, Xiang and He, Xiangnan},
  journal={arXiv preprint arXiv:2603.25077},
  year={2026}
}
```
