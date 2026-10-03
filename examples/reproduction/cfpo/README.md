# CFPO — CFPO: Counterfactual Policy Optimization for Multimodal Reasoning

ICML 2026 · [arXiv:2606.23206](https://arxiv.org/abs/2606.23206) ·
[official code](https://github.com/Raven-July/CFPO) ·
**Unofficial re-implementation in this repository.**

## Method

CFPO builds the counterfactual inside the model instead of on the input image. In every decoder
self-attention layer it takes the text-to-image attention sub-matrix, marks entries above μ + λσ
(λ = 2) as high-saliency, and replaces the corresponding image-token **values** with the mean of
all image-token values. The policy is then pushed away from this counterfactual by **maximizing**
γ·KL(π(o | Z) ‖ π(o | do(Z = Z_cf))) (k3 estimator on the sampled tokens), on top of GRPO (CFPO_G)
or DAPO (CFPO_D, which also adds an entropy term).

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-3B-Instruct |
| Training data | CFPO's ViRL39K (`RavenInJuly/CFPO_Datasets`: `ViRL39K_train.json` and `ViRL39K_images.zip`, 38,870 problems) — `bash scripts/prepare_data.sh cfpo` |
| Validation | MMK12 test (2,000), greedy, 1 sample, every 5 steps |
| Prompt / reward | CFPO's system prompt (`examples/system_prompt/cfpo.txt`, the official `base.jinja`: `<think>` + `\boxed{}`), the problem as the user message; 0.9 accuracy + 0.1 format (`math.py:compute_score`) |
| Rollout | 384 prompts x 5 rollouts per step, T=1.0, top-p 0.99; update batch 128 prompts |
| Optimization | AdamW (bf16), lr 1e-6 with 5% warmup, weight decay 1e-2, token-level loss, vision tower trainable, 2 epochs |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels |
| GRPO arms | low-var KL loss 0.01; clip 0.2 / 0.3 |
| DAPO arms | no KL; clip 0.2 / 0.28; online filtering on accuracy in (0.01, 0.99), `data.mini_rollout_batch_size=128` |
| GPUs | 2 |
| Counterfactual | `algorithm.corrupt_image=cross_modal_attention_value_mean`, `algorithm.corrupt_image_kwargs={"saliency_std_multiplier":2.0}` (λ) |
| KL_cf | `algorithm.visual_sensitivity_metric=sampled_low_var_kl`, `algorithm.visual_sensitivity_reference=current`, `algorithm.visual_sensitivity_loss_coef` γ = 0.02 (CFPO_G) / 0.01 (CFPO_D) |
| Entropy (CFPO_D) | `algorithm.invariant_entropy_coef=0.03`, `algorithm.entropy_loss_type=sampled` |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the full list of `algorithm.*` switches.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_grpo.sh` | GRPO baseline, 3B |
| `qwen2_5_vl_3b_cfpo_g.sh` | CFPO_G, 3B |
| `qwen2_5_vl_3b_dapo.sh` | DAPO baseline, 3B |
| `qwen2_5_vl_3b_cfpo_d.sh` | CFPO_D, 3B |

```bash
bash scripts/prepare_data.sh cfpo
bash examples/reproduction/cfpo/qwen2_5_vl_3b_cfpo_g.sh
bash scripts/prepare_eval_data.sh cfpo
bash scripts/eval.sh checkpoints/CFPO-Reproduce/qwen2_5_vl_3b_cfpo_g --suite cfpo
```

Checkpoints go to `checkpoints/CFPO-Reproduce/<script name>`. Append `key=value` overrides to the
command (e.g. `trainer.total_epochs=1`) and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`,
`LOGGER` or `EXPERIMENT_NAME` in the environment (e.g. `N_GPUS_PER_NODE=4 bash ...`). Evaluation is
described in [eval/README.md](../../../eval/README.md).

## Differences from the paper / official code

- Re-implemented on EasyR1 with the shared perception-reasoning modules of this repository;
  hyper-parameters follow the scripts `examples/qwen2_5_vl_3b_CFPO-{G,D}-math.sh` of the official
  repository.
- The saliency mask is computed per sample with a causal mask, as the paper describes. In the
  official configuration (`padding_free: true`), as we read its code, the samples of a
  micro-batch are packed into one sequence and the hand-computed attention used for the saliency
  statistics receives no attention mask, so it is not causal, spans the packed samples, and the
  image-value mean V̄ is taken over them. We do not reproduce this.
- The training data is CFPO's copy of ViRL39K. It holds the same problems and images as PAPO's
  copy (used by the other ViRL39K methods) in the same order, but the text of 13,426 problems
  differs (option format, `<image>` at the start instead of the end), which also changes the rows
  that enter the saliency statistics.
- Where the paper and the official code differ, we follow the code:
  - V̄ (Eq. 12) is the mean image-token value vector in the paper; the code averages over the
    image tokens and the hidden dimensions, one scalar per head.
  - The saliency statistics use the text query tokens of the input in the paper (App. A); the
    code uses every position after the image, the response included.
  - The paper writes KL_cf = KL(π(o | Z) ‖ π(o | do(Z = Z_cf))); the code uses the sampled-token
    k3 estimator, whose gradient as a loss is that of the reverse KL, KL(π_cf ‖ π) (derivation in
    [docs/implementation_notes.md](../../../docs/implementation_notes.md#the-k3-estimator-used-as-a-loss)).
  - The paper describes the entropy term of CFPO_D as entropy maximization; the code adds
    η·(−log π) to the loss, which lowers the entropy of the sampled tokens.
  - The counterfactual log-probabilities are computed once with the rollout policy before the
    update (no gradient through them); the paper writes them with the current parameters.
- Validation during training uses the MMK12 test set; the official scripts monitor `*_val_V3`
  files (Geometry3K, LogicVista, MathVerse, C-VQA-Real, MARS-Bench) that are subsets of the
  corresponding test sets.
- The Qwen2.5-VL-7B and Qwen3-VL-2B-Thinking appendix settings are not scripted.
- Evaluation (suite `cfpo`) covers all 10 Table 1 benchmarks. C-VQA-Real, MARS-Bench and TextVQA
  use CFPO's released files and grading (avg@8, mathruler match on the last `\boxed{}`, or on the
  whole response when there is none); their COCO / TextVQA images are fetched by
  `bash scripts/prepare_eval_data.sh cfpo` (about 4 GB). POPE is scored greedily with macro-F1 over the random / popular / adversarial splits, whereas
  CFPO reports avg@8 accuracy on the 9,000 pooled items. CFPO's own harness applies extra
  answer-parsing fallbacks to non-PAPO models only; our scorer applies the same rules to every model.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 1 (Qwen2.5-VL-3B trained on ViRL39K, avg@8 accuracy, %); Overall is the mean of the ten
benchmarks.

| Script | C-VQA-Real | MARS-Bench | TextVQA | POPE | MMMU-Pro(V) | Geo3k | We-Math | MMK12 | MathVerse | LogicVista | Paper Overall | This repo (Overall) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_3b_grpo` | 63.76 | 48.35 | 73.89 | 87.09 | 27.02 | 28.93 | 58.40 | 57.69 | 54.76 | 38.87 | 53.88 | TBD |
| `qwen2_5_vl_3b_cfpo_g` | 64.21 | 49.39 | 75.07 | 87.92 | 28.33 | 30.87 | 60.68 | 58.34 | 57.36 | 40.82 | 55.30 | TBD |
| `qwen2_5_vl_3b_dapo` | 65.27 | 50.26 | 76.13 | 88.37 | 28.74 | 30.20 | 59.73 | 61.41 | 55.84 | 40.07 | 55.60 | TBD |
| `qwen2_5_vl_3b_cfpo_d` | 67.40 | 53.21 | 77.33 | 88.46 | 29.87 | 34.80 | 63.15 | 64.83 | 60.84 | 44.98 | 58.49 | TBD |

The abstract's "3.17%–6.25%" gains are means of per-benchmark relative gains; the absolute overall
gains are +1.42 (CFPO_G vs GRPO) and +2.89 (CFPO_D vs DAPO). Results of this repository will be added
after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{yu2026cfpo,
  title     = {{CFPO}: Counterfactual Policy Optimization for Multimodal Reasoning},
  author    = {Yu, Zhangyuan and Sun, Wanran and Yang, Guangjing and Wu, Xiaohu and Lao, Qicheng},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2606.23206}
}
```
