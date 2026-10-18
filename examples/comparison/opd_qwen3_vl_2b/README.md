# OPD controlled comparison on Qwen3-VL-2B-Instruct

[English](README.md) | [简体中文](README_zh.md)

On-policy distillation (OPD) trains a student on its own rollouts toward a stronger teacher's next-token
distributions. The five OPD papers reproduced in this repository ([VA-OPD](../../reproduction/va_opd/README.md),
[VGS](../../reproduction/vgs/README.md), [VCSD](../../reproduction/vcsd/README.md),
[Vision-OPD](../../reproduction/vision_opd/README.md), [S-OPD](../../reproduction/s_opd/README.md)) use different
students, teachers, data and evaluations. This directory trains the four that do not need region annotations, the
two base OPD objectives and a GRPO reference under **one** setting, that of the [RLVR comparison](../rlvr_qwen3_vl_4b/README.md) with a 2B student and
a teacher trained in that comparison: Qwen3-VL-4B-Instruct after GRPO, at its best validation step (see
[Teacher](#teacher)). Each script only sets `ALGO_ARGS`; everything else comes from [common.sh](common.sh). The
other comparisons are listed in [examples/comparison/](../README.md).

> [!IMPORTANT]
> **The results are a reference, not a verdict on the methods.** The methods are our re-implementations,
> each configuration is trained once (in the RLVR comparison two runs of one configuration ended 0.6 points apart
> in the mean of the 3 groups), and the shared setting is not the one each method was designed and tuned for.
> Read [About the results](../README.md#about-the-results) before the tables.

## Shared setting

| | `examples/comparison/opd_qwen3_vl_2b/common.sh` | RLVR comparison (`../rlvr_qwen3_vl_4b/common.sh`) |
| --- | --- | --- |
| Student / teacher | Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct trained with GRPO in the RLVR comparison, at its best validation step (VCSD: an EMA of the student) | Qwen3-VL-4B-Instruct / – |
| Data, validation, prompt | ViRL39K; MMK12 test every 15 steps; `math_perception.jinja` with `<think>` as plain text | same; every 5 steps |
| Rollout | 128 prompts x 8 per step, **one update per step**, T=1.0, **top-p 1.0** | 384 x 8 per step, update batch 128 (3 updates per step), top-p 0.99 |
| Loss | the method's objective, token-level averaging | GRPO, token-level, clip 0.2 / 0.2 |
| Regularization | **none** (no reference KL, no entropy term) | KL loss 0.01, entropy penalty 0.005 |
| Reward | 0.9 accuracy + 0.1 format, logged only (it enters only `grpo.sh`) | the same, in the loss |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 2 epochs (606 updates) | same (606 updates) |
| Lengths / pixels | max prompt 4,096, max response 2,048; 200,704-1,003,520 pixels | same |
| GPUs | 4 | 4 |

Both recipes use 1,024 responses per update and the same number of updates. OPD updates once per rollout batch
so that every update is on the current policy: the distribution-level losses then need no importance weights,
and the OPD papers do the same. The rollout is not truncated (top-p 1.0) because the objectives are
expectations under the student's own distribution; with a teacher the trainer requires it. The teacher
reads the student's token ids and pixel values; the trainer checks at startup that the two models share them.

## Teacher

The teacher is `../rlvr_qwen3_vl_4b/grpo.sh` at the step with the highest validation reward
(`best_global_step` of its `checkpoint_tracker.json`; step 175 of 202 in our run), not an Instruct model:

- **Length.** Under this prompt and the 2,048-token cap, the Instruct models write past the cap on many
  questions: on 128 MMK12 test questions with 4 samples each at T=1.0, Qwen3-VL-8B-Instruct exceeds 2,048
  tokens in 43% of its answers (accuracy 0.676 without a cap, 0.469 within it), Qwen3-VL-2B-Instruct in 44%,
  and a non-thinking chat template does not shorten them. An OPD student keeps its teacher's verbose reasoning
  ([COPD](https://arxiv.org/abs/2607.19046), Table 1: 2.23K tokens after standard OPD from Qwen3-VL-8B against
  2.12K for the 2B base model), while the GRPO reference learns to finish within the cap. The GRPO run
  truncates about 1-2% of its validation answers. [VGS](../../reproduction/vgs/README.md), with the same model
  family and cap, also trains its teacher with GRPO before distilling.
- **Size.** Qwen3-VL-8B-Instruct scores only 1-3.5 points above the 4B on most reasoning benchmarks of the
  Qwen3-VL technical report (Table 4), and in VA-OPD's Table 2 standard OPD brings the same 2B student to
  almost the same scores from a 4B and an 8B teacher (math average 45.3 and 45.4); the 4B GRPO run needs no
  extra training and halves the teacher's cost.
- **Step.** GRPO's entropy and KL drift over its last steps (from about step 188 in our run). The best
  validation step also scores above the last step on the comparison suite without MMK12 (in our run 72.1
  against 70.5, mostly on the math group). MMK12 test is the validation set, so the students' MMK12 scores
  may be slightly favored by this choice.

Run `../rlvr_qwen3_vl_4b/grpo.sh` first and keep its best step (`python3 scripts/finalize_run.py
checkpoints/Comparison-Qwen3-VL-4B/grpo` keeps the last and the best step, or merge it with
`scripts/model_merger.py`); the scripts read it from there and stop before training when that step has no complete
weights. `TEACHER_PATH` or `worker.teacher.model.model_path=...` on the command line sets another teacher, and
`GRPO_TEACHER_RUN` another GRPO run.

## Methods

| Script | Method | `ALGO_ARGS` (abridged) |
| --- | --- | --- |
| `grpo.sh` | GRPO reference | the RLVR comparison's `grpo.sh` with the 2B student |
| `opd_sampled.sh` | OPD from sampled tokens | `adv_estimator=teacher_log_ratio` (log q − log π_old per token, clamped to ±10) with the PPO loss |
| `opd_full.sh` | OPD on the full distributions (the "Standard OPD" of VA-OPD and VGS) | `distill_loss_coef=1`, `policy_loss_coef=0`, `distill_divergence=reverse_kl` |
| `va_opd.sh` | [VA-OPD](../../reproduction/va_opd/README.md) | `opd_full` + `distill_weighting=va_opd` (pixelation 0.1, teacher-scored `sampled_positive_log_ratio`, τ 1, p_v 0.2, λ 0.5) |
| `vgs.sh` | [VGS](../../reproduction/vgs/README.md) | `distill_target=visual_gain`, `distill_contrast_view=no_image`, γ 2, λ 0.01, top 30% by the teacher's visual dependency per micro-batch, η 0.41 |
| `vcsd.sh` | [VCSD](../../reproduction/vcsd/README.md) | EMA teacher (0.05), `distill_target=contrast_sharpened`, `distill_contrast_view=black`, forward KL at T=2 (times 4), α 1, β 0.1 |
| `s_opd.sh` | [S-OPD](../../reproduction/s_opd/README.md) | `opd_sampled` + TPC (`random_patch` 16 px, p 0.6, `visual_sensitivity_loss_weighting=teacher`, 0.02) + PA (`rgb_gaussian_noise` σ 0.2, `visual_robustness_loss_coef` 0.02) |
| `opd_full_seq.sh` | optional ablation, not in the table | `opd_full.sh` averaged per response, then over responses |

Settings of each paper that belong to its own recipe rather than to its method (batch sizes, learning rates,
lengths, its data and prompt) are replaced by the shared setting; the method's own components and constants
are kept. Notes:

- **Not an equal comparison with GRPO.** The OPD methods learn from the teacher (the 4B GRPO run), which the
  GRPO row does not see; `grpo.sh` places them next to RL on the same student, data and budget. VCSD needs no
  external teacher (it distills from an EMA of the student), so it is the only OPD row without the teacher.
- **Loss averaging.** Every method averages over the response tokens of the batch, as the released OPD code
  (VCSD, Vision-OPD) and verl, TRL and NeMo-RL do; the papers write their losses as means over each response,
  then over responses. VA-OPD normalizes its weights per prompt by definition (its weights of a prompt sum to
  1), so its difference to `opd_full.sh` also contains this normalization; `opd_full_seq.sh` (per-response
  averaging) separates that part.
- **VGS** uses η = 0.41, the paper's value for a 2B student; with gradient clipping and AdamW a constant factor on
  the whole loss changes little. Its gate takes the 70% quantile within each update micro-batch.
- **VCSD** applies the distillation temperature to every distribution, as its paper describes (its released code
  applies only the T² factor; see the VCSD README).
- **S-OPD** builds on OPD from sampled tokens, so its row compares with `opd_sampled.sh`. It uses the weights of
  its paper's main experiments (0.02 and 0.02), masks 16 px blocks with probability 0.6 and adds noise with σ 0.2;
  the masked and the noisy copies are drawn on the resized image, as the comparison's other views, and one per
  response (its released code draws one per sample). Its teacher gate uses the comparison's teacher.
- **Vision-OPD** is not included: it needs a region annotation per question for the teacher's crop, which
  ViRL39K does not have.
- The student is the 2B model: Qwen3-VL-2B-Instruct scores 10-22 points below the 4B and 8B on most math and
  reasoning benchmarks of the Qwen3-VL technical report (Table 4), so it has room to learn from the teacher.

## Running

```bash
bash scripts/prepare_data.sh opd_comparison
bash examples/comparison/rlvr_qwen3_vl_4b/grpo.sh                    # the teacher, if the RLVR comparison has not run it
python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/grpo   # keeps its last and best step
for m in grpo opd_sampled opd_full va_opd vgs vcsd s_opd; do
    bash examples/comparison/opd_qwen3_vl_2b/$m.sh
done
bash scripts/prepare_eval_data.sh opd
bash scripts/eval.sh checkpoints/Comparison-OPD-Qwen3-VL-2B/va_opd --suite opd
python3 scripts/finalize_run.py checkpoints/Comparison-OPD-Qwen3-VL-2B/*   # after training: keep the last and the best step
```

Checkpoints go to `checkpoints/Comparison-OPD-Qwen3-VL-2B/<method>`. `MODEL_PATH` and `TEACHER_PATH` set the
student and the teacher; any `key=value` appended to a script overrides the shared setting. The teacher adds
forward passes and about 2 GB of bf16 weights per GPU on 4 GPUs: in the update, one forward per micro-batch for
the distillation losses (two for VGS and VCSD, with their contrast view); before the update, the log-probs of
the sampled tokens for OPD from sampled tokens, VA-OPD's weights and S-OPD (VA-OPD also on the pixelated image,
S-OPD on the masked one, where the student also scores its masked and noisy copies).

## Results

Suite `opd`: the 19 benchmarks of the RLVR comparison in the same three groups, with the same protocol and the
same two means (the mean of the 3 groups, in bold, and the mean of the 19 benchmarks); see
[the RLVR comparison](../rlvr_qwen3_vl_4b/README.md#results). Every method is evaluated at its last step.

| Method | Math reasoning | Vision-dependent reasoning | Perception and hallucination | Mean of the 3 groups | Mean of the 19 benchmarks | Training time (h) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (baseline) | 36.7 | 51.8 | 69.0 | 52.5 | 51.7 | – |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | 66.4 | 72.4 | 78.0 | 72.3 | 72.0 | – |
| GRPO | 55.3 | 64.5 | 72.3 | **64.0** | 63.6 | 21.0 (1.00×) |
| OPD (sampled tokens) | 51.3 | 63.1 | 72.7 | **62.4** | 61.8 | 23.0 (1.10×) |
| OPD (full distributions) | 51.8 | 63.8 | 72.9 | **62.8** | 62.3 | 25.2 (1.20×) |
| VA-OPD | 46.7 | 60.7 | 73.2 | **60.2** | 59.5 | 34.4 (1.64×) |
| VGS | 49.5 | 61.3 | 72.4 | **61.0** | 60.4 | 35.3 (1.68×) |
| VCSD (no external teacher) | 2.7 | 16.3 | 35.5 | **18.2** | 17.3 | 40.3 (1.92×) |
| S-OPD | 51.1 | 62.7 | 72.8 | **62.2** | 61.6 | 35.7 (1.70×) |

**Training time** is the wall-clock time in hours of the training steps on 4 A800 80GB GPUs (606 OPD steps, or
the 202 GRPO steps of 3 updates each: the same 606 updates of 1,024 responses), with the rollout, the teacher's
forward passes, the reward and the updates, without the validation runs and checkpoint saves; in parentheses,
the ratio to GRPO. Some runs shared an 8-GPU node with another run, so differences of a few percent can come
from the node load. VCSD's time grows as more of its answers run to the 2,048-token limit.

### Math reasoning

| Method | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVision | DynaMath | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (baseline) | 30.9 | 56.6 | 46.8 | 29.6 | 36.7 | 8.7 | 47.2 | 36.7 |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | 58.4 | 74.1 | 75.2 | 69.3 | 73.9 | 39.9 | 73.7 | 66.4 |
| GRPO | 45.4 | 65.2 | 68.1 | 52.2 | 64.0 | 29.5 | 62.4 | 55.3 |
| OPD (sampled tokens) | 41.4 | 61.6 | 63.4 | 48.5 | 57.4 | 24.4 | 62.4 | 51.3 |
| OPD (full distributions) | 41.9 | 61.7 | 63.7 | 49.7 | 58.4 | 24.2 | 63.2 | 51.8 |
| VA-OPD | 37.8 | 59.3 | 58.1 | 43.6 | 50.2 | 17.8 | 60.2 | 46.7 |
| VGS | 38.8 | 60.4 | 60.8 | 45.7 | 56.0 | 22.8 | 61.7 | 49.5 |
| VCSD (no external teacher) | 0.0 | 15.1 | 0.0 | 0.1 | 0.1 | 0.0 | 3.4 | 2.7 |
| S-OPD | 40.9 | 61.7 | 63.3 | 48.2 | 56.9 | 24.1 | 62.4 | 51.1 |

### Vision-dependent reasoning

| Method | MathVerse-V | MMMU-Pro | LogicVista | Counting | AI2D | MME (cognition) | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (baseline) | 33.8 | 20.2 | 31.2 | 69.3 | 68.0 | 88.0 | 51.8 |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | 69.6 | 43.6 | 58.7 | 87.7 | 83.0 | 92.1 | 72.4 |
| GRPO | 60.4 | 31.8 | 47.3 | 83.2 | 74.6 | 89.6 | 64.5 |
| OPD (sampled tokens) | 53.2 | 29.0 | 46.6 | 85.7 | 74.1 | 90.0 | 63.1 |
| OPD (full distributions) | 54.6 | 29.1 | 47.0 | 86.9 | 74.7 | 90.5 | 63.8 |
| VA-OPD | 45.6 | 26.8 | 42.7 | 86.1 | 73.3 | 89.4 | 60.7 |
| VGS | 51.7 | 27.7 | 44.5 | 82.8 | 73.2 | 87.6 | 61.3 |
| VCSD (no external teacher) | 0.1 | 0.1 | 0.5 | 71.4 | 3.1 | 22.8 | 16.3 |
| S-OPD | 53.3 | 29.0 | 45.8 | 84.8 | 73.2 | 89.8 | 62.7 |

### Perception and hallucination

| Method | POPE | HallusionBench | MMStar | BLINK | MME (perception) | CV-Bench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (baseline) | 88.2 | 65.2 | 53.9 | 52.6 | 79.3 | 75.1 | 69.0 |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | 88.8 | 74.0 | 69.6 | 66.3 | 84.0 | 85.2 | 78.0 |
| GRPO | 88.3 | 67.5 | 58.6 | 57.8 | 81.6 | 80.0 | 72.3 |
| OPD (sampled tokens) | 88.7 | 67.9 | 62.1 | 57.3 | 81.8 | 78.5 | 72.7 |
| OPD (full distributions) | 88.4 | 70.0 | 60.7 | 55.9 | 83.9 | 78.7 | 72.9 |
| VA-OPD | 88.8 | 69.0 | 60.1 | 58.8 | 83.1 | 79.4 | 73.2 |
| VGS | 87.6 | 68.6 | 60.5 | 57.4 | 81.8 | 78.7 | 72.4 |
| VCSD (no external teacher) | 89.0 | 26.0 | 8.0 | 6.3 | 53.9 | 29.7 | 35.5 |
| S-OPD | 88.0 | 69.4 | 61.1 | 57.4 | 82.9 | 78.1 | 72.8 |

MME (cognition) and MME (perception) are MME's cognition score (commonsense reasoning, numerical calculation,
text translation and code reasoning; 260 questions, out of 800) and its perception score (the other 10 subtasks;
2,114 questions, out of 2,000), each on a 0-100 scale.

### Best validation step

Each method at the step with the best validation reward (`best_global_step` in the run's
`checkpoint_tracker.json`; MMK12 test every 15 steps), on the 18 benchmarks other than MMK12, as in
[the RLVR comparison](../rlvr_qwen3_vl_4b/README.md#best-validation-step). The last two columns give the same
mean at the last step (606; GRPO: 202) and the difference between the two, from the rounded values in the table.

| Method | Best step | Math reasoning (without MMK12) | Vision-dependent reasoning | Perception and hallucination | Mean of the 3 groups | Mean of the 3 groups at the last step | Best − last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | 202 | 55.8 | 64.5 | 72.3 | **64.2** | 64.2 | 0.0 |
| OPD (sampled tokens) | 606 | 51.8 | 63.1 | 72.7 | **62.5** | 62.5 | 0.0 |
| OPD (full distributions) | 600 | 52.1 | 63.6 | 72.7 | **62.8** | 63.0 | −0.2 |
| VA-OPD | 255 | 52.3 | 62.9 | 72.1 | **62.5** | 60.4 | +2.1 |
| VGS | 600 | 49.9 | 61.2 | 72.0 | **61.0** | 61.3 | −0.3 |
| VCSD (no external teacher) | 30 | 43.3 | 55.9 | 71.1 | **56.8** | 18.3 | +38.5 |
| S-OPD | 600 | 51.3 | 62.4 | 72.6 | **62.1** | 62.3 | −0.2 |

VCSD's validation reward is highest at step 30 and falls afterwards; at the last step most of its answers run
to the 2,048-token limit. Its paper trains 90 steps of 32 prompts x 8 responses, about 4% of the 606 updates of
1,024 responses here, so this budget is far longer than the one VCSD was designed for.
