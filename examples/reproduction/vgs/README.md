# VGS — Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

ICML 2026 (Spotlight) · [arXiv:2606.00564](https://arxiv.org/abs/2606.00564) ·
[code repository](https://github.com/hee-suk-yoon/Decomposed_OPD) (no code released as of 2026-10) ·
**Unofficial re-implementation in this repository.**

## Method

Standard on-policy distillation (OPD) minimizes the reverse KL between the student p and the teacher q on the
student's rollouts. VGS (Visual Gradient Steering) splits the teacher's distribution into a language prior
q(· | text) and a visual gain q(· | image, text) / q(· | text), and adds a target q* ∝ p(· | text) · q(· | image,
text) / q(· | text) that keeps the student's own language prior and asks it to match the teacher's visual gain
(the student's text-only distribution is detached in q*). A language-prior term pulls the student's text-only
distribution toward the teacher's on the 30% most vision-dependent tokens (by the teacher's KL between its image
and text-only distributions), so that the steering does not unlearn the language prior:

    L = η [ KL(p ‖ q) + γ KL(p ‖ q*) + λ 1[VDS_t > Q_0.7] KL(p_text ‖ q_text) ],  γ = 2, λ = 0.01.

η rescales the steered gradient to the norm of the standard one; the paper uses fixed values (0.41 for a 2B,
0.36 for a 4B student). The teacher is Qwen3-VL-8B-Instruct trained with GRPO on the same data.

## Setting

| | Teacher (stage 1, Table 3) | Students (stage 2, Table 4) | GRPO combinations (Table 2) |
| --- | --- | --- | --- |
| Model | Qwen3-VL-8B-Instruct | Qwen3-VL-2B / 4B-Instruct | Qwen3-VL-2B-Instruct |
| Data | Vision-SR1-47K as split by the authors (`tomyoon2/OPD_train`, 45,246; validation `OPD_val`, 2,382) — `bash scripts/prepare_data.sh vgs` | same | same |
| Prompt / reward | the VGS system prompt (`examples/system_prompt/vgs.txt`: `<reason></reason>` and `\boxed{}`) with the problem as the user message; 0/1 correctness (`examples/reward_function/vgs.py`) | same (the reward is logged only) | same |
| Rollout | 512 prompts x 8, update batch 128 (4 updates per step), T=1.0, top-p 0.99 | 128 prompts x 4, one update per step, T=1.0, top-p 1.0 | 512 x 8, update batch 128, T=1.0, top-p 1.0 |
| Loss | GRPO without KL, clip 0.2 / 0.2 | Standard OPD (reverse KL) or VGS, averaged over tokens; no KL | GRPO, 0.7 GRPO + 0.3 Standard OPD, or 0.7 GRPO + 0.3 VGS (Eq. 22, without η) |
| Optimization | AdamW, lr 1e-6 constant, weight decay 1e-2, vision tower trainable, 2 epochs (176 steps) | same, 1 epoch (353 steps) | same, 1 epoch (88 steps, 352 updates) |
| Lengths | max prompt 12,800, max response 2,048 | max prompt 16,384, max response 2,048 | same as the students |
| GPUs | 8 | 8 | 8 |

Pixels 262,144-4,194,304 for all stages. The gate of the language-prior term takes the 70% quantile within each
update micro-batch (`algorithm.vgs_vds_scope=micro_batch`; `global` takes it over the whole step with an extra
teacher pass). See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#on-policy-distillation).

## Scripts

| Script | Paper | Run |
| --- | --- | --- |
| `qwen3_vl_8b_grpo_teacher.sh` | Table 3 | stage 1: the GRPO teacher |
| `qwen3_vl_2b_opd.sh` / `qwen3_vl_2b_vgs.sh` | Table 1 (8B → 2B) | Standard OPD / VGS (η 0.41) |
| `qwen3_vl_4b_opd.sh` / `qwen3_vl_4b_vgs.sh` | Table 1 (8B → 4B) | Standard OPD / VGS (η 0.36) |
| `qwen3_vl_2b_grpo.sh` | Table 2 | GRPO |
| `qwen3_vl_2b_grpo_opd.sh` | Table 2 | 0.7 GRPO + 0.3 Standard OPD |
| `qwen3_vl_2b_grpo_vgs.sh` | Table 2 | 0.7 GRPO + 0.3 VGS |

```bash
bash scripts/prepare_data.sh vgs
# stage 1: train the teacher, then turn its last step into a Hugging Face model directory
bash examples/reproduction/vgs/qwen3_vl_8b_grpo_teacher.sh
python3 scripts/finalize_run.py checkpoints/VGS-Reproduce/qwen3_vl_8b_grpo_teacher
# stage 2: the students and the combinations read the teacher from TEACHER_PATH
bash examples/reproduction/vgs/qwen3_vl_2b_vgs.sh
bash scripts/prepare_eval_data.sh vgs
bash scripts/eval.sh checkpoints/VGS-Reproduce/qwen3_vl_2b_vgs --suite vgs
bash scripts/eval.sh checkpoints/VGS-Reproduce/qwen3_vl_2b_vgs --suite vgs --temperature 0 --num-samples 1   # Acc@1
```

`TEACHER_PATH` defaults to `checkpoints/VGS-Reproduce/qwen3_vl_8b_grpo_teacher/global_step_176/actor`, the
finalized last step of stage 1 with the full training split (2 epochs of 88 steps); set it if the run ended at
another step, or to use another teacher. Checkpoints go to `checkpoints/VGS-Reproduce/<script name>`; append
`key=value` overrides and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in
the environment. The students run four forwards per micro-batch (the student and the teacher, with and without
the image); the paper reports 1.375 times the step time of Standard OPD.

**GPUs.** The long prompt limits make dynamic batching pack few rows per micro-batch (`common.sh` sets 2 / 4 for
the update and the forward passes). On 80 GB GPUs, our smoke tests of two steps on 4 GPUs peaked at 67 GB per
GPU for the 8B teacher, 68 GB for the 2B VGS student and 61 GB for the 2B GRPO + VGS combination.

## Differences from the paper

The repository named in the paper holds no code; everything below that the paper does not state is our choice.

- **Teacher reward.** The paper only mentions a correctness reward. Ours takes the last `\boxed{}`, or the text
  after `</reason>` when there is none (the paper's sample outputs often answer there), compares option letters for
  multiple-choice answers and uses mathruler otherwise; no format term. The teacher's clip ratio (0.2 / 0.2) and
  the absence of a KL term follow Table 3; the rest of GRPO is this repository's.
- **Students.** 128 prompts x 4 rollouts per step for 1 epoch gives the 353 steps of the paper's training curves;
  the paper states a global batch of 512 without the rollouts per prompt. The loss is averaged over tokens, as
  TRL (which the paper uses) does; the paper writes the loss per response, then over responses. The student's
  text-only distribution in q* is detached, and the gate's quantile is taken per micro-batch; the paper states
  neither. The text-only input drops the image from the user message, with the same chat template and system
  prompt.
- **Table 2.** The paper gives no rollout settings for the combinations; they use the teacher's GRPO recipe with
  top-p 1.0 (distillation needs untruncated sampling, so all three scripts use it), 1 epoch. As Eq. 22, the
  distillation term is not multiplied by η. The distillation term of the 2nd to 4th update of a step uses
  rollouts of the policy before the step.
- **Evaluation** (suite `vgs`): our MathVerse vision-only and MMMU-Pro sets come from PAPO-Eval and differ from
  the authors' multiple-choice subsets; VisualPuzzles and VlmsAreBlind are not available. The paper reports Acc@1
  (greedy) and Acc@16 (mean accuracy of 16 samples at T=1.0); the suite defaults to avg@8 at T=1.0.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository come from
> our unofficial re-implementation, trained once with a single seed on our hardware and software versions and
> evaluated with this repository's harness. They may differ from the official results, including smaller or no
> gains over the baselines, and do not replace them; corrections from the authors are welcome. See
> [About the results](../../../README.md#-about-the-results).

Paper Table 1 (Acc@1 / Acc@16, %; average of 7 benchmarks: MMMU-Pro-4, LogicVista, MathVerse VD and VO,
VisualPuzzles, MathVision, VlmsAreBlind).

| Script | Paper Acc@1 | Paper Acc@16 | This repo |
| --- | --- | --- | --- |
| teacher (`qwen3_vl_8b_grpo_teacher`) | 61.37 | – | TBD |
| Qwen3-VL-2B-Instruct (initial) | 31.32 | – | TBD |
| `qwen3_vl_2b_opd` | 43.74 | 45.07 | TBD |
| `qwen3_vl_2b_vgs` | 46.10 | 46.14 | TBD |
| Qwen3-VL-4B-Instruct (initial) | 46.68 | – | TBD |
| `qwen3_vl_4b_opd` | 56.64 | 56.86 | TBD |
| `qwen3_vl_4b_vgs` | 58.12 | 57.27 | TBD |

Paper Table 2 (2B student, Acc@1 / Acc@16): GRPO 44.83 / 45.68; GRPO + Standard OPD 45.41 / 45.22; GRPO + VGS
47.20 / 46.57.

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{yoon2026vgs,
  title     = {Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding},
  author    = {Yoon, Hee Suk and Yoon, Eunseop and Jang, Jaehyun and Eom, SooHwan and Hong, Ji Woo and Hasegawa-Johnson, Mark and Dai, Qi and Luo, Chong and Yoo, Chang D.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
