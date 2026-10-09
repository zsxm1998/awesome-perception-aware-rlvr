# CGPO — Counterfactual Grounding Policy Optimization

**CGPO: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology
Vision-Language Reasoning**<br>
Shengxuming Zhang, Linyun Zhou, Hengrui Lou, Zhenyang Wang, Xiuming Zhang, Zunlei Feng<br>
ACM MM 2026 **Oral** (Proceedings of the 34th ACM International Conference on Multimedia, Rio de Janeiro,
Brazil) · [Paper](https://doi.org/10.1145/3767308.3835969) · [BibTeX](#citation) · **This directory is the official implementation.**

## Method

CGPO trains a vision-language model to perform *evidence-sensitive reasoning* (ESR): key
reasoning steps localize the visual evidence they rely on, and the conclusion changes when that
evidence is removed. It needs only answer-level supervision:

1. **Counterfactual images.** The policy grounds evidence inline in its chain of thought. The
   predicted evidence regions *E* are masked to build the counterfactual image *I \ E*; each region
   is filled with its own local mean colour, which removes structure without creating
   out-of-distribution artifacts.
2. **Token-level evidence dependence.** *S_t* = KL between the output distributions on *I* and on
   *I \ E*, evaluated with the rollout (old) policy.
3. **ESR-aware response scaling.** Response dependence *S_i* = mean of *S_t*; the advantage scale
   is `alpha_i = 1 + lambda * tanh((S_i - median_B S) / MAD_B S)` over the batch.
4. **Token advantage modulation.** Only reasoning-critical tokens (top-`rho_r` entropy) and
   perception-critical tokens (top-`rho_p` evidence dependence, plus the tokens that specify
   *E*) are optimized; perception-critical tokens receive `alpha_i * A_i`, reasoning-critical
   tokens `A_i`.
5. **Grounding consistency regularization (GCR).** For every grounded entity the same policy
   re-detects the entity on the original image; an F1-weighted overlap between the matched
   predicted and re-detected boxes is added to the reward (`R + gamma * R_gc`, only for correct
   answers). `R_gc` is the mean of this score over the response's own entities, an entity that the
   re-detection does not find scoring 0 (`grounding_consistency_aggregation=response`). This prevents
   the policy from inflating the evidence boxes to increase *S*. (The paper describes greedy CIoU
   matching; this release uses optimal one-to-one IoU matching, see `compute_detection_reward` in
   `verl/trainer/grounding_consistency.py`.)

### Where it lives in the code

| Component | Config | Code |
| --- | --- | --- |
| Inline evidence format | `data.format_prompt=examples/format_prompt/xml_grounded_reasoning_v2.jinja` | `<region name=".." image_idx=".." id="..">[[x1,y1,x2,y2],...]</region>` tags inside `<think>` (the paper calls them entity tags), boxes in the 0-1000 range |
| Counterfactual image | `algorithm.corrupt_image=cgpo_flat`, `corrupt_image_kwargs={"fill_type":"local_mean"}`, `corrupt_image_position=response` | `verl/trainer/perception_reasoning_data.py::cgpo_flat` |
| Evidence dependence *S_t* | `algorithm.visual_sensitivity_reference=old` (sampled KL estimator) | `verl/trainer/visual_sensitivity.py` |
| Response scaling (`lambda`) | `algorithm.advantage_scaling_method=cgpo`, `cgpo_response_scaling_coef` | `perception_reasoning_loss.py::_compute_cgpo_response_scaling` |
| Token selection (`rho_r`, `rho_p`, Span_E) | `top_entropy_quantile`, `top_perception_quantile`, `*_thr_granularity=micro_batch` (thresholds within each update micro-batch, as in the paper experiments), `include_region_tokens_in_perception_mask=true` | `perception_reasoning_loss.py` |
| GCR (`gamma`) | `use_grounding_consistency_reward=true`, `grounding_consistency_reward_weight`, `grounding_consistency_aggregation=response` (`grounding_consistency_detector=self`; `grounding-dino` is available as an external detector) | `verl/trainer/grounding_consistency.py`, `examples/reward_function/xml_grounded_reasoning.py` |

## Reproduction in this repository (natural images)

The paper trains pathology models in two stages: SFT on public and **in-house** pathology data,
then RLVR on PathVQA plus an in-house VQA set. The in-house data cannot be released, so the
pathology pipeline is not part of this repository. Instead, this directory reproduces the RLVR
stage of CGPO on public natural-image data with the paper's backbones and RL hyper-parameters:

| | Setting |
| --- | --- |
| Models | Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct (no SFT stage) |
| Training data | ViRL39K as processed by PAPO (`bash scripts/prepare_data.sh cgpo`) |
| Validation | MMK12 test (2,000) |
| RL | GRPO, lr 1e-6, 384 prompts x 8 rollouts per step (generation batch 3,072), 1,024 responses per update, 200 steps, KL 0.04 (low-var KL), clip 0.2, max response 2,048, 8 GPUs |
| CGPO | `lambda = 0.1`, `rho_r = rho_p = 0.3`, `gamma = 0.5` |
| Reward | 0.1 format + 0.9 accuracy (`\boxed{}` answer, grounded-format checks) |

The GRPO baseline uses exactly the same output format, reward and hyper-parameters; only the
policy-optimization algorithm differs.

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_grpo.sh` | GRPO baseline, Qwen2.5-VL-7B |
| `qwen2_5_vl_7b_cgpo.sh` | CGPO, Qwen2.5-VL-7B |
| `qwen3_vl_8b_grpo.sh` | GRPO baseline, Qwen3-VL-8B |
| `qwen3_vl_8b_cgpo.sh` | CGPO, Qwen3-VL-8B |

```bash
bash scripts/prepare_data.sh cgpo
bash examples/reproduction/cgpo/qwen3_vl_8b_cgpo.sh
bash scripts/eval.sh checkpoints/CGPO-Reproduce/qwen3_vl_8b_cgpo --suite cgpo
```

CGPO is also part of the controlled comparison on Qwen3-VL-4B
([examples/comparison/](../../comparison/README.md)), where it shares every hyper-parameter with the other
methods (GCR weight 0.1 there).

Notes:
- Instruct models are not trained to produce inline evidence, so early in training many
  responses contain no regions; for those responses the counterfactual image equals the original,
  so their evidence dependence is zero and they mainly learn through the entropy-selected tokens.
  The paper starts from an SFT model that already produces ESR-style responses.
- GCR re-runs the policy as a referring detector inside the rollout engine; its cost is reported
  as `algo/gcr/detection_time_s`. Detection is skipped for groups without a correct answer.
- Earlier versions of these scripts scored GCR per group (`grounding_consistency_aggregation=group`,
  still the default of the option): the frequency-weighted share of the group's re-detected regions
  that a response grounds consistently. Naming more regions raises that score; in the Qwen3-VL-4B
  comparison run the policy went from 1.4 regions per response at step 10 to 23 at step 200, mostly
  coarse, repeated boxes listed after the reasoning. The scripts now use the paper's per-response
  mean (`response`).
- `xml_grounded_reasoning_v2.jinja` replaces `xml_grounded_reasoning.jinja` (kept for runs trained
  with it): its example box no longer has round coordinates, and rule 5 asks for tight boxes and no
  whole-image boxes. Qwen3-VL-4B-Instruct copies the example: on MMK12 test (512 prompts x 8
  samples), 80% of its box coordinates are multiples of 100 with the earlier prompt and 40% with v2.

## Results

> [!NOTE]
> The paper's results were obtained with the original ms-swift implementation on pathology data. The
> natural-image numbers of this repository come from the EasyR1 re-implementation, trained once with
> a single seed and evaluated with this repository's harness, together with baselines run in the
> same way. See [About the results](../../../README.md#-about-the-results).

Pathology results reported in the paper (PathMMU overall accuracy, %):

| Backbone | Split | SFT | + GRPO | + CGPO |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B | PathMMU test-tiny | 60.1 | 66.6 | 70.1 |
| Qwen2.5-VL-7B | PathMMU test | 58.9 | 63.8 | 69.0 |
| Qwen3-VL-8B | PathMMU test-tiny | 61.4 | 68.4 | 72.2 |
| Qwen3-VL-8B | PathMMU test | 60.4 | 67.3 | 70.4 |

Natural-image reproduction in this repository (eval suite `cgpo`):

| Backbone | GRPO | CGPO |
| --- | --- | --- |
| Qwen2.5-VL-7B | TBD | TBD |
| Qwen3-VL-8B | TBD | TBD |

## Citation

```bibtex
@inproceedings{zhang2026cgpo,
  title     = {{CGPO}: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning},
  author    = {Zhang, Shengxuming and Zhou, Linyun and Lou, Hengrui and Wang, Zhenyang and Zhang, Xiuming and Feng, Zunlei},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  location  = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {10},
  doi       = {10.1145/3767308.3835969},
  url       = {https://doi.org/10.1145/3767308.3835969}
}
```
