# S-OPD — Unlocking Student-side Learning in Multimodal On-Policy Distillation

[arXiv:2609.39120](https://arxiv.org/abs/2609.39120) · [official code](https://github.com/Sirilaw/S-OPD) ·
**Unofficial re-implementation in this repository.**

## Method

On-policy distillation (OPD) trains the student on its own rollouts toward the teacher. S-OPD keeps OPD from sampled
tokens (the advantage log q(y_t) − log π(y_t), clamped to ±10, with the PPO loss) and adds two terms on the
student's own predictions (Eq. 7, L = L_OPD + λ_TPC L_TPC + λ_PA L_PA):

- **TPC (teacher-calibrated policy contrast)** pushes the student's prediction on the image away from its
  prediction on a masked copy (blocks of the image blackened with probability ρ = 0.6): the k3 estimate of
  KL(π(· | image) ‖ π(· | masked)) at the sampled token, with no gradient through the masked branch, is
  **maximized**, but only where the teacher relies on the image. The teacher scores the same masked copy, and token
  t is weighted by its log-probability drop w_t = max(log q(y_t | image) − log q(y_t | masked), 0); the weighted
  terms are averaged over the tokens with w_t > 0.
- **PA (policy agreement)** pulls the student's prediction on the image toward its prediction on a copy with
  Gaussian noise (σ = 0.2 on [0, 1] pixels, clipped and rounded to 8 bits): the same k3 estimate, **minimized**
  over all response tokens.

## Setting

| | This repository |
| --- | --- |
| Student / teacher | Qwen3-VL-2B-Instruct or Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Instruct, or that model trained with GRPO (the `_grpo_teacher` scripts, OPD_G and S-OPD_G in Table 1) |
| Training data | ViRL39K (38,870) — `bash scripts/prepare_data.sh s_opd`; the component analysis on Geometry3K train (2,101) |
| Validation | MMK12 test (Geometry3K test for the Geometry3K runs), one sample at T=1.0, for monitoring only; we report the last step |
| Prompt | system prompt "You are a helpful assistant." ([helpful_assistant.txt](../../system_prompt/helpful_assistant.txt)) and [math.jinja](../../format_prompt/math.jinja) (the think / `\boxed{}` instruction after the question); the reward is logged only |
| Rollout | 192 prompts x 1 rollout per step (Geometry3K: 128), one update per step, T=1.0, top-p 1.0 |
| Base loss | `adv_estimator=teacher_log_ratio` (clamped to ±10) with the PPO loss, clip 0.2 / 0.2, averaged over the response tokens; no reference KL, no entropy term |
| TPC | λ_TPC 0.02 (Geometry3K: 0.005); `corrupt_image=random_patch` with ρ 0.6 and blocks of the model's patch size (16 px for Qwen3-VL), drawn on the original image before resizing; `visual_sensitivity_loss_coef`, `visual_sensitivity_loss_weighting=teacher` |
| PA | λ_PA 0.02; `incremental_image_transform=rgb_gaussian_noise` with σ 0.2, drawn on the original image before resizing; `visual_robustness_loss_coef` |
| Optimization | AdamW, lr 1e-6 constant, weight decay 1e-2, gradient clipping 1.0, vision tower trainable; ViRL39K 1 epoch (about 200 steps), Geometry3K 20 epochs (about 320 steps) |
| Lengths / pixels | max prompt 4,096, max response 4,096 (Geometry3K: 1,024 / 2,048); 65,536-16,777,216 pixels (the Qwen3-VL processor's defaults) |
| GPUs | 8 |

The weights, ρ, σ, block size, batch sizes, lengths and epochs are those of the paper's Table 6, §4.2 and
Appendix A.1. See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#visual-sensitivity-scoring)
for the switches.

## Scripts

| Script | Paper | Run |
| --- | --- | --- |
| `qwen3_vl_2b_opd.sh` / `qwen3_vl_2b_s_opd.sh` | Table 1 | OPD / S-OPD, 8B → 2B, ViRL39K |
| `qwen3_vl_4b_opd.sh` / `qwen3_vl_4b_s_opd.sh` | Table 1 | OPD / S-OPD, 8B → 4B, ViRL39K |
| `qwen3_vl_8b_grpo_teacher.sh` | Table 6 | the GRPO-trained 8B teacher: 192 prompts x 5 rollouts per step, 1 epoch |
| `qwen3_vl_{2b,4b}_opd_grpo_teacher.sh` / `qwen3_vl_{2b,4b}_s_opd_grpo_teacher.sh` | Table 1 | OPD_G / S-OPD_G, from the GRPO-trained teacher |
| `geo3k_qwen3_vl_2b_{opd,tpc,pa,s_opd}.sh` | Table 2 | component analysis, 8B → 2B, Geometry3K: OPD, + TPC, + PA, S-OPD |

```bash
bash scripts/prepare_data.sh s_opd
bash examples/reproduction/s_opd/qwen3_vl_2b_s_opd.sh
# the GRPO-trained teacher, then the students that distill from it
bash examples/reproduction/s_opd/qwen3_vl_8b_grpo_teacher.sh
python3 scripts/finalize_run.py checkpoints/S-OPD-Reproduce/qwen3_vl_8b_grpo_teacher   # keeps its last step
bash examples/reproduction/s_opd/qwen3_vl_2b_s_opd_grpo_teacher.sh
# evaluation
bash scripts/prepare_eval_data.sh s_opd
bash scripts/eval.sh checkpoints/S-OPD-Reproduce/qwen3_vl_2b_s_opd --suite s_opd --temperature 0 --num-samples 1
```

Checkpoints go to `checkpoints/S-OPD-Reproduce/<script name>`. The `_grpo_teacher` scripts read the last step of
the teacher run and stop before training when it has no complete weights; `TEACHER_PATH` sets another teacher and
`GRPO_TEACHER_RUN` another GRPO run. Append `key=value` overrides to the command and set `N_GPUS_PER_NODE`,
`MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the environment.

**Cost.** Before each update, S-OPD adds a teacher forward on the masked copies and two student forwards without
gradient (masked and noisy copies) to OPD from sampled tokens. The paper reports 189.4 → 243.6 s per step (+28.6%)
for the 2B student on ViRL39K (Appendix B.6, 4 A100s). The 8B teacher holds about 16 GB of bf16 weights, about
2 GB per GPU on 8 GPUs.

## Differences from the paper

The method follows the paper and the released code (commit
[`e3c7cc9`](https://github.com/Sirilaw/S-OPD/tree/e3c7cc9427f99e3aa196453d6f01b1c73da4c77e)); the hyper-parameters
are the paper's. On random inputs, the TPC and PA terms, their gradients, the masks and the noise of this
repository match the released functions given the same random draws.

- **Averaging of TPC.** The weighted TPC terms are averaged over the tokens with w_t > 0 of the whole update
  (all GPUs), as the released code. `worker.actor.loss_avg_mode=seq` averages them within each response instead
  (at least one token in the denominator), then over the responses, the form of Eq. 5; it also averages the OPD
  and PA terms per response.
- **Clamps.** Both k3 terms clamp the log-ratio to ±20 and each token's k3 to at most 10, as the released code.
- **When the copies are scored.** The student's log-probs on the masked and the noisy copies come from one forward
  before the update with the rollout policy, and the teacher's from one forward on the same masked copy; the
  released code scores the student's copies inside each update micro-batch without gradient. With one update per
  step both use the same parameters.
- **Random draws.** One mask and one noise draw per prompt and step (one rollout per prompt), from seeds derived from
  the prompt and the step; the released code draws them from the unseeded global random generator.
- **Data.** The training set is PAPO's preparation of ViRL39K (`bash scripts/prepare_data.sh s_opd`), as the VA-OPD
  and VCSD reproductions; its problems and answers are those of TIGER-Lab's `39Krelease.parquet` row by row, up to
  whitespace and the `\boxed{}` around the answers, and its 42,908 images have the pixels of `images.zip`. For the
  851 problems without an `<image>` placeholder, PAPO's preparation puts the image before the text with no space
  between them (the released preprocessing adds a space).
- **Teacher placement.** The teacher runs on the training GPUs (FSDP, bf16) and scores the student's token ids and
  pixel values; the paper serves it on separate GPUs (2 A100s for the student, 2 for the teacher).
- **GRPO teacher.** Table 6 gives the teacher's data, batch size, rollouts, lengths, learning rate and epochs;
  everything else is this repository's GRPO (low-variance KL loss 0.01 to the reference model, clip 0.2 / 0.2) with
  the setting above.
- **Not provided.** VA-OPD with S-OPD (Table 1): this repository's VA-OPD distills the full next-token distributions,
  and that loss does not combine with the perception-aware losses here. The OPSD runs of Fig. 5 (the hint column of
  their Geometry3K data is not released) and the gating ablations of Table 3. The coefficient and ρ / σ sweeps
  (Tables 4, 5 and 10) and the 4B component analysis are the scripts above with other values or `MODEL_PATH`.
- **Evaluation** (suite `s_opd`): the paper evaluates with VLMEvalKit, greedy decoding and a Qwen3-30B-A3B model
  for answer extraction and grading. The suite asks with the training prompt at the training image size and scores
  with this repository's rules (see [eval/README.md](../../../eval/README.md) for each benchmark's split and scorer),
  so the numbers are not directly comparable with the paper's; evaluate the untrained student first and compare it
  with the base rows of Table 1 to calibrate the protocols. The command above decodes greedily, as the paper (the
  math sets otherwise default to avg@8 at T=1.0). ZeroBench is a gated dataset: accept its terms on Hugging Face
  before preparing it. Geometry3K test (the in-domain set of Table 2) is the benchmark `geo3k`.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository come from
> our unofficial re-implementation, trained once with a single seed on our hardware and software versions and
> evaluated with this repository's harness. They may differ from the official results, including smaller or no
> gains over the baselines, and do not replace them; corrections from the authors are welcome. See
> [About the results](../../comparison/README.md#about-the-results).

Paper Table 1 (ViRL39K; greedy decoding with VLMEvalKit, %). The subscript G marks the GRPO-trained 8B teacher.

| Student | Method | MathVerse | MathVista | MathVision | WeMath | LogicVista | VisualPuzzles | ZeroBench | MMMU | Avg | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2B | Base model | 45.51 | 61.2 | 30.29 | 32.29 | 34.52 | 16.18 | 13.17 | 45.78 | 34.87 | TBD |
| 2B | OPD | 46.41 | 65.2 | 34.35 | 36.67 | 36.24 | 12.16 | 9.28 | 48.44 | 36.09 | TBD |
| 2B | S-OPD | 47.81 | 67.0 | 35.93 | 37.43 | 37.65 | 12.67 | 11.98 | 49.22 | 37.46 | TBD |
| 2B | OPD_G | 50.68 | 67.6 | 35.95 | 39.24 | 39.49 | 14.89 | 12.57 | 47.33 | 38.47 | TBD |
| 2B | S-OPD_G | 52.76 | 68.3 | 36.03 | 40.29 | 39.25 | 15.62 | 13.78 | 49.11 | 39.39 | TBD |
| 4B | Base model | 42.28 | 73.8 | 49.31 | 55.43 | 54.81 | 25.43 | 15.57 | 61.56 | 47.27 | TBD |
| 4B | OPD | 46.14 | 74.2 | 47.33 | 50.95 | 54.81 | 29.71 | 19.46 | 61.44 | 48.01 | TBD |
| 4B | S-OPD | 45.84 | 75.2 | 47.57 | 52.38 | 55.93 | 30.22 | 20.05 | 61.78 | 48.62 | TBD |
| 4B | OPD_G | 46.60 | 76.6 | 48.55 | 55.71 | 54.14 | 33.99 | 20.36 | 64.33 | 50.04 | TBD |
| 4B | S-OPD_G | 46.81 | 77.0 | 48.22 | 57.43 | 58.39 | 34.25 | 20.05 | 64.76 | 50.86 | TBD |

Paper Table 2 (2B student, Geometry3K; Avg over Geometry3K test and the eight benchmarks): OPD 40.93 / 37.48,
+ TPC 42.56 / 38.07, + PA 41.93 / 38.60, S-OPD 44.26 / 39.08 (Geometry3K / Avg).

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{liu2026sopd,
  title   = {Is Better Teacher Supervision Enough? Unlocking Student-side Learning in Multimodal On-Policy Distillation},
  author  = {Liu, Siyuan and Tian, Kanghui and Duan, Yue and He, Yutao and Yang, Shangdong and Zhang, Jian and Shi, Yinghuan},
  journal = {arXiv preprint arXiv:2609.39120},
  year    = {2026}
}
```
