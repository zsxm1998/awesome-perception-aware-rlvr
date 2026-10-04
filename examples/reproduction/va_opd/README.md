# VA-OPD — Visual-Advantage On-Policy Distillation for Vision-Language Models

[arXiv:2605.21924](https://arxiv.org/abs/2605.21924) · no official code ·
**Unofficial re-implementation in this repository.**

## Method

On-policy distillation (OPD) trains the student on its own rollouts toward the teacher's next-token
distribution with a per-token reverse KL. VA-OPD weights this loss by the teacher's **visual advantage**: the
teacher scores each sampled token with the original image and with a pixelated copy (bilinear down to 10% of
the side, nearest-neighbor back), and a_t = max(log q(y_t | image) − log q(y_t | pixelated), 0). Within the K
responses of a prompt, the mean advantage of each response is z-normalized and turned into response weights by a
softmax (τ = 1); within a response, the top p_v = 20% tokens by a_t share λ = 0.5 of its weight and the other
tokens the rest. The weights of a prompt sum to 1, so with equal advantages and λ equal to the share of the top
group (⌈p_v T⌉ / T, i.e. p_v when p_v T is an integer) the loss is the standard OPD loss.

## Setting

| | This repository |
| --- | --- |
| Student / teacher | Qwen3-VL-2B-Instruct / Qwen3-VL-8B-Instruct (Table 2 also 4B and 32B teachers) |
| Training data | Geometry3K train (2,101) — `bash scripts/prepare_data.sh va_opd`; Table 2 also ViRL39K (38,870) |
| Validation | Geometry3K test (601) for Geometry3K runs, MMK12 test for ViRL39K runs, 8 samples at T=1.0 (monitoring only) |
| Prompt | `examples/format_prompt/math.jinja` (`\boxed{}`); the reward is logged only |
| Rollout | 16 prompts x 4 rollouts per step, one update per step, T=1.0, top-p 1.0 |
| Loss | full-vocabulary reverse KL to the teacher: per response then over responses for Standard OPD (`loss_avg_mode=seq`, Eq. 1); VA-OPD's per-token weights (`distill_weighting=va_opd`), whose per-prompt sums are 1 (Eq. 6); no reference KL, no entropy term |
| VA-OPD | `corrupt_image=pixelation` (ratio 0.1), `visual_sensitivity_reference=teacher`, `visual_sensitivity_metric=sampled_positive_log_ratio`, τ 1, p_v 0.2, λ 0.5 |
| Optimization | AdamW, lr 1e-6 constant, weight decay 1e-2, gradient clipping 1.0, vision tower trainable, 5 epochs (about 655 steps on Geometry3K, 12,150 on ViRL39K) |
| Lengths / pixels | max prompt 2,048, max response 2,048; 262,144-4,194,304 pixels |
| GPUs | 8 (80 GB GPUs for the 32B teacher) |
| Checkpoints | every 50 steps (Geometry3K) or 500 steps (ViRL39K), all kept, weights only (see "Best checkpoint") |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#on-policy-distillation) for the switches.

## Scripts

| Script | Paper | Run |
| --- | --- | --- |
| `qwen3_vl_2b_opd.sh` | Table 1, Table 2 | Standard OPD, 8B → 2B, Geometry3K |
| `qwen3_vl_2b_va_opd.sh` | Table 1, Table 2 | VA-OPD, 8B → 2B, Geometry3K |
| `qwen3_vl_2b_opd_teacher_4b.sh` / `qwen3_vl_2b_va_opd_teacher_4b.sh` | Table 2 | 4B → 2B, Geometry3K |
| `qwen3_vl_2b_opd_teacher_32b.sh` / `qwen3_vl_2b_va_opd_teacher_32b.sh` | Table 2 | 32B → 2B, Geometry3K |
| `qwen3_vl_2b_opd_virl39k.sh` / `qwen3_vl_2b_va_opd_virl39k.sh` | Table 2 | 8B → 2B, ViRL39K |
| `qwen3_vl_2b_grpo.sh` | Table 1 | GRPO baseline, Geometry3K |
| `qwen3_vl_2b_papo.sh` | Table 1 | PAPO baseline (GRPO + KL_prcp), Geometry3K |

```bash
bash scripts/prepare_data.sh va_opd
bash examples/reproduction/va_opd/qwen3_vl_2b_va_opd.sh
bash scripts/prepare_eval_data.sh va_opd
```

Checkpoints go to `checkpoints/VA-OPD-Reproduce/<script name>`. `TEACHER_PATH` sets the teacher and
`VA_OPD_DATA` (`geo3k` or `virl39k`) the training set; append `key=value` overrides to the command and set
`N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the environment.

**Cost.** The paper trains 8B → 2B on Geometry3K in about 19 hours on 8 A100s (Fig. 5). At that speed the
ViRL39K setting (5 epochs, about 12,150 steps) takes about two weeks. The 32B teacher holds about 64 GB of bf16
weights, about 8 GB per GPU on 8 GPUs.

**Best checkpoint.** The paper reports each method at its best checkpoint. The scripts keep every checkpoint
(weights only, `trainer.save_limit=-1`, `trainer.save_model_only=true`; a run cannot be resumed from them, and a
restarted script stops with an error when it finds one: pass `trainer.find_last_checkpoint=false` to start over, or
train with `trainer.save_model_only=false` to be able to resume). Evaluate every step, pick the step with the best mean over the eight
benchmarks, and report the last step as well:

```bash
bash scripts/eval.sh checkpoints/VA-OPD-Reproduce/qwen3_vl_2b_va_opd --suite va_opd --all-steps \
    --temperature 1.0 --num-samples 8
python3 scripts/finalize_run.py checkpoints/VA-OPD-Reproduce/qwen3_vl_2b_va_opd/global_step_N   # keep step N
```

The best step is selected on the test benchmarks themselves, so it overstates the performance of a model chosen
without them; the last step is the unbiased number.

## Differences from the paper

There is no released code; everything below that the paper does not state is our choice.

- **Not given in the paper**: the learning rate (we use 1e-6 constant, as this repository's GRPO), the update
  schedule (one update per step of 16 prompts x 4 rollouts; with 5 epochs this gives the about 650 steps of the
  paper's training curves), the lengths, the image resolution, the prompt, the sampling (T=1.0, top-p 1.0) and ε
  of the z-score (1e-6). The standard deviation is the population one (ddof=0), the only choice that reproduces the
  weights of the paper's Fig. 3 example. The top-20% group has max(1, ⌈0.2 T⌉) tokens; ties (frequent, since a_t
  is clipped at 0) are broken by a random order seeded from `data.seed` and the step.
- **Standard OPD** averages each response's KL over its tokens, then over responses (`loss_avg_mode=seq`), as Eq.
  1 writes it; only then does VA-OPD with equal advantages and λ = ⌈p_v T⌉ / T (p_v when p_v T is an integer; with
  λ = p_v = 0.2 the per-token weights differ by up to a few percent when T is not a multiple of 5) reduce to it
  exactly. The controlled comparison averages over tokens for every method ([its
  README](../../comparison/opd_qwen3_vl_2b/README.md)).

- **Teacher scoring.** The teacher scores the sampled tokens with the original and the pixelated image in a pass
  before the update (two forwards), and again with the original image in the update for the KL; the paper
  describes one extra forward per rollout.
- **Baselines.** GRPO and PAPO use this repository's GRPO (low-variance KL 0.01 to the reference model, clip 0.2)
  and PAPO's settings from [examples/reproduction/papo](../papo/README.md) (patches of 16 px blackened with
  p=0.6 on the original image, KL_prcp coefficient 0.02) on the same data, rollouts and optimizer; the paper gives
  no settings for them. CoT-SFT and off-policy KD (Table 1) need supervised fine-tuning, which this repository does
  not provide.
- **Evaluation** (suite `va_opd`): the paper reports avg@8 at T=1.0 with official scoring and a GPT-4o judge where
  applicable; we score with rules. The suite asks with the training prompt at the training image size and decodes
  HallusionBench, AI2D, MMMU and MMStar greedily by default; the command above samples 8 responses at T=1.0
  everywhere, as the paper (HallusionBench then scores the first). OCRBench is not available; see
  [eval/README.md](../../../eval/README.md) for the splits of the other benchmarks. Evaluate the untrained student
  first and compare it with the Base row of Table 1 to calibrate the protocols.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository come from
> our unofficial re-implementation, trained once with a single seed on our hardware and software versions and
> evaluated with this repository's harness. They may differ from the official results, including smaller or no
> gains over the baselines, and do not replace them; corrections from the authors are welcome. See
> [About the results](../../../README.md#-about-the-results).

Paper Table 1 (8B → 2B, Geometry3K; avg@8 at T=1.0, best checkpoint, %).

| Method | WeMath | MathVista | MathVerse | Math Avg | HalluB | AI2D | MMMU | MMStar | OCRBench | Visual Avg | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 36.8 | 63.9 | 19.6 | 40.1 | 52.6 | 77.8 | 49.1 | 56.1 | 85.9 | 64.3 | TBD |
| GRPO | 44.1 | 64.6 | 24.9 | 44.5 | 54.0 | 77.5 | 52.9 | 57.8 | 84.8 | 65.4 | TBD |
| PAPO | 44.8 | 64.5 | 25.8 | 45.0 | 54.0 | 77.8 | 52.7 | 58.0 | 84.7 | 65.4 | TBD |
| Standard OPD | 43.3 | 63.7 | 29.1 | 45.4 | 52.0 | 75.8 | 50.9 | 59.7 | 84.7 | 64.6 | TBD |
| VA-OPD | 46.6 | 66.4 | 31.9 | 48.3 | 54.5 | 78.2 | 51.5 | 59.9 | 86.4 | 66.1 | TBD |

Paper Table 2 (Math / Visual Avg of Standard OPD → VA-OPD): 4B → 2B 45.3 → 47.4 / 64.4 → 65.2; 8B → 2B 45.4 → 48.3
/ 64.6 → 66.1; 32B → 2B 51.1 → 54.8 / 65.3 → 67.3; 8B → 2B on ViRL39K 46.4 → 50.2 / 65.5 → 68.0.

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{liu2026vaopd,
  title   = {Visual-Advantage On-Policy Distillation for Vision-Language Models},
  author  = {Liu, Ruiqi and Lv, Xiaolei and Li, Gengsheng and Zhu, Ximo and Wang, Zhiheng and Zhang, Zhengbo and Chen, Junkai and Li, Zhiheng and Li, Bo and Gao, Jun and Wu, Shu},
  journal = {arXiv preprint arXiv:2605.21924},
  year    = {2026}
}
```
