# VCSD — Visual Contrastive Self-Distillation

[arXiv:2607.21556](https://arxiv.org/abs/2607.21556) · [official code](https://github.com/joliang17/VCSD) ·
**Unofficial re-implementation in this repository.**

## Method

VCSD is on-policy self-distillation without an external teacher, answers or rewards. The teacher is an
exponential moving average (EMA) of the student. On the student's own rollouts, the teacher scores every response
position twice, with the original image (q) and with a black image of the same size (q_ctrl). The target sharpens
q toward the tokens the image makes more likely,

    q*(v) ∝ q(v)^anchor (q(v) / q_ctrl(v))^α   on   S = {v : q(v) ≥ β max q},

with the end-of-sequence tokens not contrasted, and the student minimizes the forward KL(q* ‖ p) at the
distillation temperature T_KD, times T_KD². α = 1, β = 0.1, anchor 1, T_KD = 2, EMA rate 0.05.

## Setting

| | This repository |
| --- | --- |
| Models | Qwen3-VL-2B / 4B / 8B-Instruct, Qwen3.5-2B / 4B / 9B (Table 1) |
| Training data | ViRL39K (38,870) — `bash scripts/prepare_data.sh vcsd`; the problem as the user message, no format instruction |
| Chat template | the official one (`examples/chat_template/qwen_no_thinking.jinja`: an empty think block after the generation prompt) with the released tokenizer (`plain_think_tokens=false`) |
| Rollout | 32 prompts x 8 rollouts per step, one update per step, T=1.0, top-p 1.0 |
| Teacher | EMA of the student (`worker.teacher.source=ema`, rate 0.05, fp32), updated after each step, saved with the checkpoints |
| Target / loss | `distill_target=contrast_sharpened`, `distill_contrast_view=black`, `vcsd_alpha=1`, `vcsd_support_beta=0.1`, `vcsd_anchor_coef=1`, end-of-sequence ids from the generation config; forward KL at T=2 on every distribution, times 4; averaged over the response tokens of the step; no reference KL, no entropy term |
| Optimization | AdamW, lr 2e-6 after 10 linear warm-up steps, weight decay 1e-2, gradient clipping 1.0, vision tower trainable, 90 steps |
| Lengths / pixels | max prompt 6,144 (longer prompts filtered), max response 1,024; 262,144-4,194,304 pixels |
| GPUs | 8 |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#on-policy-distillation) for the switches.

## Scripts

| Script | Model |
| --- | --- |
| `qwen3_vl_2b_vcsd.sh`, `qwen3_vl_4b_vcsd.sh`, `qwen3_vl_8b_vcsd.sh` | Qwen3-VL-2B / 4B / 8B-Instruct |
| `qwen3_5_2b_vcsd.sh`, `qwen3_5_4b_vcsd.sh`, `qwen3_5_9b_vcsd.sh` | Qwen3.5-2B / 4B / 9B |
| `qwen3_vl_2b_vcsd_code_temperature.sh` | Qwen3-VL-2B-Instruct with the temperature as the released code applies it (see below) |

```bash
bash scripts/prepare_data.sh vcsd
bash examples/reproduction/vcsd/qwen3_vl_2b_vcsd.sh
bash scripts/prepare_eval_data.sh vcsd
bash scripts/eval.sh checkpoints/VCSD-Reproduce/qwen3_vl_2b_vcsd --suite vcsd --format-prompt none \
    --chat-template examples/chat_template/qwen_no_thinking.jinja --plain-think-tokens false
```

Checkpoints go to `checkpoints/VCSD-Reproduce/<script name>`; append `key=value` overrides and set
`N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the environment. Qwen3.5 models
need the fast-path packages (`QWEN35_FASTPATH_ONLY=1 bash scripts/install_env.sh`, see the
[README](../../../README.md)).

**GPUs.** The paper trains on 8 B200 (192 GB). On 80 GB GPUs the 2B and 4B models fit on one node of 8; the EMA
teacher adds 4 bytes per parameter (8B: about 4 GB per GPU on 8 GPUs). For the 8B and 9B models, if memory runs
short, keep the teacher on the CPU as the released code does (`worker.teacher.offload.offload_params=true`) or use
two nodes (`NNODES=2`, see the [Vision-OPD README](../vision_opd/README.md#two-nodes)). These settings are
validated with `DRY_RUN=1` only; the 8B and 9B runs have not been run in this repository.

## Differences from the paper / official code

- **Distillation temperature.** The paper computes the teacher's two distributions, the support set, the target
  and the student at T_KD = 2 and multiplies the loss by T_KD². The released code computes all of them at the
  rollout temperature 1 and uses 2 only for the factor 4, so its support set is ten times stricter in probability
  ratio (T = 2 with β = 0.1 equals T = 1 with β = 0.01) and the target is often the teacher's argmax alone; the
  authors' issue tracker reports a falling target and student entropy with the released settings. We follow the
  paper (`distill_temperature_scope=all`); `qwen3_vl_2b_vcsd_code_temperature.sh` runs the code's behavior
  (`loss_scale_only`), for which our per-token loss equals the released `vcsd.py` on the same logits with the
  default `vcsd_anchor_coef=1` (with another anchor the end-of-sequence tokens differ: ours keep anchor · log q,
  the released code log q).

- **Loss averaging.** As the released code, the loss is a mean over response tokens; the code takes it per
  micro-batch and weights the micro-batches by their number of rows, which depends on how dynamic batching splits
  a step, while we take one mean over all response tokens of the step. The paper writes a mean per response, then
  over responses.
- **End-of-sequence tokens.** The released code fixes them to 151643 and 151645 (Qwen3-VL's `<|endoftext|>` and
  `<|im_end|>`); these ids are ordinary tokens in Qwen3.5. We read them from each model's generation config
  (Qwen3.5: 248044 and 248046).
- **Black image.** As in the code, the black image has the original size, so the control view has the same tokens;
  we replace the processed pixel values by the normalized value of black instead of processing a black image.
- **Batch and steps.** 32 prompts and one update per step (the paper); the released script defaults to 96 prompts
  and a mini-batch of 96. 90 steps, as the paper (the README example trains 150).
- **EMA checkpoints.** The EMA teacher is saved with the checkpoints and restored on resume; the released code
  does not save it, so a resumed run restarts the teacher from the base model.
- **Data and prompt.** The paper trains on ViRL39K; its training file and prompt format are not released. We use
  the problems of this repository's ViRL39K as the user message. The OPSD baseline of Table 1 (a teacher prompted
  with the reference answer) is not scripted.
- **Evaluation** (suite `vcsd`): the paper does not state its evaluation protocol. The suite uses greedy decoding
  for the high-resolution benchmarks and reports HallusionBench as (aAcc + fAcc + qAcc) / 3 (`aqf_mean` in the
  details), as the paper; Acc. is the unweighted mean of the seven scores. Evaluate with the training prompt
  (no format instruction), chat template and tokenizer, as in the command above.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository come from
> our unofficial re-implementation, trained once with a single seed on our hardware and software versions and
> evaluated with this repository's harness. They may differ from the official results, including smaller or no
> gains over the baselines, and do not replace them; corrections from the authors are welcome. See
> [About the results](../../../README.md#-about-the-results).

Paper Table 1 (Acc., the mean of BLINK, MMStar, V*, MathVista, HR-Bench 4K / 8K and HallusionBench, %).

| Model | Paper Base | Paper VCSD | This repo Base | This repo VCSD |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct | 62.27 | 67.04 | TBD | TBD |
| Qwen3-VL-4B-Instruct | 71.30 | 73.16 | TBD | TBD |
| Qwen3-VL-8B-Instruct | 72.51 | 76.26 | TBD | TBD |
| Qwen3.5-2B | 68.61 | 71.51 | TBD | TBD |
| Qwen3.5-4B | 73.94 | 76.77 | TBD | TBD |
| Qwen3.5-9B | 74.97 | 79.24 | TBD | TBD |

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{liang2026vcsd,
  title   = {Visual Contrastive Self-Distillation},
  author  = {Liang, Yijun and Tian, Yunjie and Li, Yijiang and Jia, Yuqi and Huang, Furong and Zhou, Tianyi and Fu, Di},
  journal = {arXiv preprint arXiv:2607.21556},
  year    = {2026}
}
```
