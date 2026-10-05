# VGPO — Visually-Guided Policy Optimization for Multimodal Reasoning

ACL 2026 · [arXiv:2604.09349](https://arxiv.org/abs/2604.09349) ·
[official code](https://github.com/wzb-bupt/VGPO) ·
**Unofficial re-implementation in this repository.**

## Method

VGPO reweights the advantages of a DAPO-style objective by how much each token attends to the image,
measured without an extra forward pass. A token's visual score is the cosine between its last-layer
hidden state and the mean hidden state of the image tokens of its prompt. Because this similarity
tends to fade over long reasoning chains, *Visual Attention Compensation* boosts the scores of the
most visual tokens in the second half of each response by a factor that grows with the position
(1 + β t / T). The compensated scores then give two factors that multiply the advantage:

- **token level**: within a response, the normalized score minus its mean, so tokens that look at the
  image more get more credit (the factor averages 1 over the response);
- **response level**: within the responses to the same prompt, the normalized sum of scores minus its
  mean, so responses that rely more on the image get more credit.

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct (3B and 32B scripts as well) |
| Training data | ViRL39K as processed by PAPO (`PAPOGalaxy/PAPO_ViRL39K_train`, 38,870) — `bash scripts/prepare_data.sh vgpo` |
| Validation | MMK12 test (`PAPOGalaxy/PAPO_MMK12_test`, 2,000), every 5 steps, one sample at T=0.6, top-p 0.95 |
| Prompt / reward | `<think>` + `\boxed{}` (`examples/format_prompt/math_perception.jinja`, the official `math_format_perception.jinja`); accuracy only (`math.py:compute_score_wo_format`) |
| Rollout | 512 prompts x 8 rollouts per step, T=1.0, top-p 0.99; vLLM tensor parallel 2 |
| RL | DAPO: no KL, clip 0.2 / 0.28, dynamic sampling on accuracy in (0.01, 0.99), no entropy term, token-level loss; update batch 128 prompts (4 updates per step) |
| Optimization | AdamW (bf16), lr 1e-6 constant, weight decay 1e-2, vision tower trainable, 2 epochs (about 150 steps); data seed 1 |
| Lengths / pixels | max prompt 4,096, max response 2,048; 262,144-4,194,304 pixels |
| GPUs | 8 (32 for 32B) |
| Visual score | `algorithm.visual_sensitivity_metric=hidden_state_similarity`, `visual_sensitivity_hidden_layers=last`, `visual_sensitivity_hidden_pooling=prototype`, `visual_sensitivity_reference=old` |
| Reweighting | `algorithm.advantage_scaling_method=vgpo`, `vgpo_compensation_strength=0.3` (β), `vgpo_gate_tail_ratio=0.5` (γ), `vgpo_gate_top_ratio=0.2` (κ) |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#advantage-scaling) for the
switches. Training logs the cosine distribution (`vgpo/cosine_mean`, `cosine_min`,
`cosine_negative_fraction`), the gate pass rate (about γκ = 0.1), the token- and response-level factors
and the share of responses at the lower clamp of 0.9.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_dapo.sh` | DAPO baseline, 7B |
| `qwen2_5_vl_7b_dapo_vgpo.sh` | VGPO, 7B |
| `qwen2_5_vl_3b_dapo.sh` | DAPO baseline, 3B |
| `qwen2_5_vl_3b_dapo_vgpo.sh` | VGPO, 3B |
| `qwen2_5_vl_32b_dapo.sh` | DAPO baseline, 32B, 4 nodes |
| `qwen2_5_vl_32b_dapo_vgpo.sh` | VGPO, 32B, 4 nodes |

```bash
bash scripts/prepare_data.sh vgpo
bash examples/reproduction/vgpo/qwen2_5_vl_7b_dapo_vgpo.sh
bash scripts/prepare_eval_data.sh vgpo
bash scripts/eval.sh checkpoints/VGPO-Reproduce/qwen2_5_vl_7b_dapo_vgpo --suite vgpo
# greedy decoding, as the paper
bash scripts/eval.sh checkpoints/VGPO-Reproduce/qwen2_5_vl_7b_dapo_vgpo --suite vgpo --temperature 0 --num-samples 1
```

Checkpoints go to `checkpoints/VGPO-Reproduce/<script name>`. Append `key=value` overrides to the
command and set `N_GPUS_PER_NODE`, `NNODES`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME`
in the environment. The 32B scripts default to `NNODES=4`; start a Ray cluster across the nodes first
(see the [Vision-OPD README](../vision_opd/README.md#two-nodes)). Evaluation is described in
[eval/README.md](../../../eval/README.md).

## Differences from the paper / official code

The official repository has no license file, so the implementation is written in this repository;
each step follows the released code (`verl/workers/actor/dp_actor.py`, `verl/trainer/core_algos.py`):
the last-layer hidden state after the final norm, the `<|image_pad|>` mean as prototype, the gate
start ⌊T/2⌋, ⌊κ n⌋ (at least 1) gated tokens with ties included, the compensation β t / (T − 1), the
per-response min-max with ε = 1e-8, and the two clamps. Our factors agree with the released functions
to within 4e-6 on random inputs when the groups coincide. The differences:

- **Response-level groups.** Eq. 10–11 normalize the response scores within the rollouts of the same
  prompt. The released code takes each run of 8 adjacent responses of an update micro-batch as a group
  (and the whole micro-batch when it has fewer than 8); after the length balancing across GPUs and
  the dynamic micro-batches of the official script, these runs hold responses to different prompts, as
  the function's own comment anticipates. We group by prompt, as the paper describes.
- **When the scores are computed.** The released code scores the tokens in the update forward, with
  the current parameters. We score them in the old log-probability forward and compute all factors
  on the driver before the update, so that a prompt's whole group is available and the factors do not
  depend on how the batch is split; the first of the four updates of a step sees the same parameters,
  the later three differ by one to three updates at lr 1e-6. No extra forward pass is needed in
  either case.
- **Score offset.** The paper maps cosines to [0, 1] with (S + 1) / 2 (Eq. 4). The released code
  min-max normalizes over the micro-batch with the padding positions (cosine 0) included, which, for
  the gate and the normalized factors, amounts to scores S − min(0, smallest cosine). We use that offset
  with the smallest cosine of the whole rollout batch and without padding (`vgpo_score_offset=official`);
  `vgpo_score_offset=paper` uses Eq. 4. The offset changes how strong the compensation is relative to
  the spread of the scores.
- **Response score.** As the released code, a response's score sums the normalized compensated scores
  ŵ; Eq. 9 sums them before normalization (`vgpo_trajectory_score=raw`).
- **Clamps.** As the released code, the token-level factor is clamped to [0.1, 2] and the
  response-level factor to [0.9, 2]; the paper has no clamps. Whenever the lower clamp of 0.9 applies,
  the mean response factor of the group exceeds 1.
- **Precision.** Cosines are computed in fp32; the released code stores them in bf16, which creates
  more ties at the gate threshold.
- **γ.** `vgpo_gate_tail_ratio` is the tail fraction γ of the paper (the gate starts at ⌊(1 − γ) T⌋);
  the released option of the same role is the start fraction. Both are 0.5 by default.
- **Recipe.** As the official script; the paper lists a validation batch of 1,024 where the script
  uses 512, which does not affect training. The 3B and 32B scripts reuse the 7B recipe, as the paper
  states; the official repository only has the 7B script. The memory settings of the 32B scripts
  (vLLM tensor parallel 4, micro-batches of 2 and 4) are ours and have not been run.
- The Geometry3K and MMK12 training settings of Table 2 and the ablations are not scripted.
- **Evaluation** (suite `vgpo`) uses the PAPO-Eval data as the paper. The paper decodes greedily
  (`--temperature 0 --num-samples 1`); the suite default is PAPO-Eval's avg@8 at T=1.0. GeoMath has no
  public data and is not included, so Avg-Math here averages five benchmarks. The official evaluation
  code is not released.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Tables 1 and 2 (greedy accuracy, %). Avg-Math averages the six math benchmarks including
GeoMath (not available here), Avg-Vision the four vision-dependent ones.

| Script | MathVista | MathVerse | We-Math | MMK12 | GeoMath | Geo3K | Avg-Math | LogicVista | Counting | MMMU-Pro | MathVerse-V | Avg-Vision | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_7b_dapo` | 68.7 | 69.6 | 70.8 | 77.0 | 51.3 | 45.6 | 63.8 | 47.4 | 85.5 | 39.0 | 66.6 | 59.6 | TBD |
| `qwen2_5_vl_7b_dapo_vgpo` | 74.1 | 71.6 | 72.5 | 81.5 | 54.3 | 45.8 | 66.6 | 49.4 | 95.5 | 40.5 | 67.6 | 63.3 | TBD |
| `qwen2_5_vl_3b_dapo` | 63.9 | 57.1 | 62.9 | 64.1 | 47.6 | 36.3 | 55.3 | 43.0 | 65.5 | 31.5 | 53.1 | 48.3 | TBD |
| `qwen2_5_vl_3b_dapo_vgpo` | 65.0 | 61.4 | 62.1 | 71.6 | 50.2 | 36.1 | 57.7 | 45.9 | 78.0 | 32.5 | 58.1 | 53.6 | TBD |
| `qwen2_5_vl_32b_dapo` | 69.7 | 73.0 | 78.7 | 82.0 | 55.2 | 51.6 | 68.4 | 59.1 | 85.5 | 47.3 | 67.4 | 64.8 | TBD |
| `qwen2_5_vl_32b_dapo_vgpo` | 75.3 | 73.8 | 78.7 | 87.3 | 56.5 | 52.3 | 70.7 | 59.3 | 90.0 | 47.3 | 70.1 | 66.7 | TBD |

Counting (SuperCLEVR) has 200 questions. Because the evaluation protocol differs, compare our runs
with each other rather than with these numbers. Results of this repository will be added after the
reproduction runs finish.

## Citation

```bibtex
@inproceedings{wang2026vgpo,
  title     = {Visually-Guided Policy Optimization for Multimodal Reasoning},
  author    = {Wang, Zengbin and Xiong, Feng and Lin, Liang and Hu, Xuecai and Wang, Yong and Wang, Yanlin and Zhang, Man and Chu, Xiangxiang},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```
