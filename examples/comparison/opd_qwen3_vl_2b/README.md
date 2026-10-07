# Controlled comparison of on-policy distillation on Qwen3-VL-2B-Instruct

On-policy distillation (OPD) trains a student on its own rollouts toward a stronger teacher's next-token
distributions. The four OPD papers reproduced in this repository ([VA-OPD](../../reproduction/va_opd/README.md),
[VGS](../../reproduction/vgs/README.md), [VCSD](../../reproduction/vcsd/README.md),
[Vision-OPD](../../reproduction/vision_opd/README.md)) use different students, teachers, data and evaluations.
This directory trains the three that do not need region annotations, the two base OPD objectives and a GRPO
reference under **one** setting, the RL comparison's ([../README.md](../README.md)) with a 2B student and a teacher
trained in that comparison: Qwen3-VL-4B-Instruct after GRPO, at its best validation step (see [Teacher](#teacher)).
Each script only sets `ALGO_ARGS`; everything else comes from [common.sh](common.sh).

## Shared setting

| | `examples/comparison/opd_qwen3_vl_2b/common.sh` | RL comparison (`../qwen3_vl_4b/common.sh`) |
| --- | --- | --- |
| Student / teacher | Qwen3-VL-2B-Instruct / the RL comparison's GRPO run (Qwen3-VL-4B-Instruct) at its best validation step (VCSD: an EMA of the student) | Qwen3-VL-4B-Instruct / – |
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

The teacher is `../qwen3_vl_4b/grpo.sh` at the step with the highest validation reward
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

Run `../qwen3_vl_4b/grpo.sh` first and keep its best step (`python3 scripts/finalize_run.py
checkpoints/Comparison-Qwen3-VL-4B/grpo` keeps the last and the best step, or merge it with
`scripts/model_merger.py`); the scripts read it from there and stop before training when that step has no complete
weights. `TEACHER_PATH` or `worker.teacher.model.model_path=...` on the command line sets another teacher, and
`GRPO_TEACHER_RUN` another GRPO run.

## Methods

| Script | Method | `ALGO_ARGS` (abridged) |
| --- | --- | --- |
| `grpo.sh` | GRPO reference | the RL comparison's `grpo.sh` with the 2B student |
| `opd_sampled.sh` | OPD from sampled tokens | `adv_estimator=teacher_log_ratio` (log q − log π_old per token, clamped to ±10) with the PPO loss |
| `opd_full.sh` | OPD on the full distributions (the "Standard OPD" of VA-OPD and VGS) | `distill_loss_coef=1`, `policy_loss_coef=0`, `distill_divergence=reverse_kl` |
| `va_opd.sh` | [VA-OPD](../../reproduction/va_opd/README.md) | `opd_full` + `distill_weighting=va_opd` (pixelation 0.1, teacher-scored `sampled_positive_log_ratio`, τ 1, p_v 0.2, λ 0.5) |
| `vgs.sh` | [VGS](../../reproduction/vgs/README.md) | `distill_target=visual_gain`, `distill_contrast_view=no_image`, γ 2, λ 0.01, top 30% by the teacher's visual dependency per micro-batch, η 0.41 |
| `vcsd.sh` | [VCSD](../../reproduction/vcsd/README.md) | EMA teacher (0.05), `distill_target=contrast_sharpened`, `distill_contrast_view=black`, forward KL at T=2 (times 4), α 1, β 0.1 |
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
- **Vision-OPD** is not included: it needs a region annotation per question for the teacher's crop, which
  ViRL39K does not have.
- The student is the 2B model: Qwen3-VL-2B-Instruct scores 10-22 points below the 4B and 8B on most math and
  reasoning benchmarks of the Qwen3-VL technical report (Table 4), so it has room to learn from the teacher.

## Running

```bash
bash scripts/prepare_data.sh opd_comparison
bash examples/comparison/qwen3_vl_4b/grpo.sh                    # the teacher, if the RL comparison has not run it
python3 scripts/finalize_run.py checkpoints/Comparison-Qwen3-VL-4B/grpo   # keeps its last and best step
for m in grpo opd_sampled opd_full va_opd vgs vcsd; do
    bash examples/comparison/opd_qwen3_vl_2b/$m.sh
done
bash scripts/prepare_eval_data.sh opd
bash scripts/eval.sh checkpoints/Comparison-OPD-Qwen3-VL-2B/va_opd --suite opd
python3 scripts/finalize_run.py checkpoints/Comparison-OPD-Qwen3-VL-2B/*   # after training: keep the last and the best step
```

Checkpoints go to `checkpoints/Comparison-OPD-Qwen3-VL-2B/<method>`. `MODEL_PATH` and `TEACHER_PATH` set the
student and the teacher; any `key=value` appended to a script overrides the shared setting. The teacher adds
its forward passes to the update (one per micro-batch; two for VGS, VCSD and VA-OPD's scoring of the
pixelated image) and about 2 GB of bf16 weights per GPU on 4 GPUs.

## Results

Suite `opd`: the 19 benchmarks of the RL comparison in the same three groups, with the same protocol and the
same overall score (the mean of the three group means; All is the mean of the 19 benchmarks); see
[the comparison README](../README.md#results). Every method is evaluated at its last step.

| Method | Math reasoning | Vision-dependent reasoning | Perception and hallucination | Overall | All |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (student, no training) | TBD | TBD | TBD | TBD | TBD |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD |
| OPD (sampled tokens) | TBD | TBD | TBD | TBD | TBD |
| OPD (full distributions) | TBD | TBD | TBD | TBD | TBD |
| VA-OPD | TBD | TBD | TBD | TBD | TBD |
| VGS | TBD | TBD | TBD | TBD | TBD |
| VCSD (no external teacher) | TBD | TBD | TBD | TBD | TBD |

### Math reasoning

| Method | Geo3K | MathVista | We-Math | MMK12 | MathVerse | MathVision | DynaMath | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (student, no training) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (sampled tokens) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (full distributions) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VA-OPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VGS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VCSD (no external teacher) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Vision-dependent reasoning

| Method | MathVerse-V | MMMU-Pro | LogicVista | Counting | AI2D | MME (cognition) | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (student, no training) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (sampled tokens) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (full distributions) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VA-OPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VGS | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VCSD (no external teacher) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Perception and hallucination

| Method | POPE | HallusionBench | MMStar | BLINK | MME (perception) | CV-Bench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (student, no training) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Teacher: GRPO on Qwen3-VL-4B-Instruct, best validation step | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (sampled tokens) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD (full distributions) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VA-OPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VGS | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VCSD (no external teacher) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

MME (cognition) and MME (perception) are MME's cognition score (commonsense reasoning, numerical calculation,
text translation and code reasoning; 260 questions, out of 800) and its perception score (the other 10 subtasks;
2,114 questions, out of 2,000), each on a 0-100 scale.

Results will be added after the runs finish.

> [!IMPORTANT]
> Every method in this table is our unofficial re-implementation, run in one shared setting (backbone, data
> and hyper-parameters chosen for the comparison, not for each method), on our hardware and software versions,
> with a single seed per method. Differences of a point or less between methods can come from training
> randomness alone. The original papers remain the reference for each method's performance. See
> [About the results](../../../README.md#-about-the-results); corrections from the authors are welcome.
