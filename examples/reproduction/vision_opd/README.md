# Vision-OPD — Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation

[arXiv:2605.18740](https://arxiv.org/abs/2605.18740) · [official code](https://github.com/VisionOPD/Vision-OPD) ·
[data](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) · **Unofficial re-implementation in this
repository.**

## Method

Multimodal models answer questions about a small region better when they see a crop of it than when they see
the whole image. Vision-OPD distills this ability into the model itself. The student sees the whole image with
a red box around the region and answers a four-option question about it; the teacher, an exponential moving
average (EMA) of the student, scores the student's answer with the same question and a zoomed crop around the
box. The student minimizes the Jensen-Shannon divergence (β = 0.5) between its next-token distribution and the
teacher's, on the student's top-100 tokens plus one bucket for the remaining probability; no labels or rewards
are used.

## Setting

| | This repository |
| --- | --- |
| Models | Qwen3.5-4B (one node of 8 GPUs), Qwen3.5-9B (two nodes of 8 GPUs), non-thinking mode |
| Training data | Vision-OPD-6K (6,241 questions on SA-1B images; `images` with the red box for the student, the released crops `teacher_images` for the teacher; the prompt kept verbatim) — `bash scripts/prepare_data.sh vision_opd` (about 31 GB) |
| Chat template | the official one (`examples/chat_template/qwen_no_thinking.jinja`: an empty think block after the generation prompt), no system prompt, no format instruction |
| Rollout | 96 prompts x 8 rollouts per step, one update per step, T=1.0, top-p 1.0 |
| Teacher | EMA of the student (`worker.teacher.source=ema`, rate 0.05, fp32) reading the crop (`algorithm.teacher_view=data_image`, `data.teacher_image_key=teacher_images`) |
| Loss | JSD with β = 0.5 on the student's top-100 ids plus a tail bucket (`distill_divergence=jsd`, `distill_support=student_top_k`), fp32, averaged over the response tokens of the step; no reference KL, no entropy term, no reward |
| Optimization | AdamW, lr 2e-6 after 10 linear warm-up steps (the first update has lr 0), weight decay 1e-2, gradient clipping 1.0, vision tower trainable, 1 epoch (65 steps); the last step is kept |
| Lengths / pixels | max prompt 8,192 (not filtered), max response 1,024; 65,536-16,777,216 pixels |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#on-policy-distillation) for the switches.

## Scripts

| Script | Model | GPUs |
| --- | --- | --- |
| `qwen3_5_4b_vision_opd.sh` | Qwen3.5-4B | 8 (one node) |
| `qwen3_5_9b_vision_opd.sh` | Qwen3.5-9B | 16 (two nodes, see below) |

```bash
bash scripts/prepare_data.sh vision_opd
bash examples/reproduction/vision_opd/qwen3_5_4b_vision_opd.sh
bash scripts/prepare_eval_data.sh vision_opd
bash scripts/eval.sh checkpoints/Vision-OPD-Reproduce/qwen3_5_4b_vision_opd --suite vision_opd
bash scripts/eval.sh Qwen/Qwen3.5-4B --suite vision_opd   # the base model
```

Qwen3.5 needs the fast-path packages (`QWEN35_FASTPATH_ONLY=1 bash scripts/install_env.sh`, see the
[README](../../../README.md)); the authors report about 5 hours for 4B on 8 H800 GPUs and 7 hours for 9B on 16,
with GPUs of 80 GB or more. Checkpoints go to `checkpoints/Vision-OPD-Reproduce/<script name>`; append
`key=value` overrides and set `N_GPUS_PER_NODE`, `NNODES`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or
`EXPERIMENT_NAME` in the environment.

### Two nodes

The 9B setting uses 16 GPUs, as the paper. To run it on two machines with 8 GPUs each:

1. Both machines mount the same shared file system (the repository, `data/` and `checkpoints/`) and have the same
   conda environment.
2. On the head node: `conda activate <env> && ray start --head --port=6379 --num-gpus=8`.
3. On the other node: `conda activate <env> && ray start --address=<head node IP>:6379 --num-gpus=8`.
4. On the head node, `ray status` should list 16 GPUs.
5. On the head node, in the repository: `NNODES=2 N_GPUS_PER_NODE=8 bash examples/reproduction/vision_opd/qwen3_5_9b_vision_opd.sh`
   (the script sets `NNODES=2` by default).
6. When the run has finished, run `ray stop` on both nodes.

While the cluster is up, the trainer's `ray.init()` attaches to it, so any other training started on these
machines joins the same cluster; start other jobs only after `ray stop`. Multi-node communication uses NCCL;
on machines with several network interfaces set `NCCL_SOCKET_IFNAME` (and `NCCL_IB_HCA` for InfiniBand) to the
interface the nodes share. The two-node setting is validated with `DRY_RUN=1` only and has not been run in this
repository.

## Differences from the paper / official code

- **Teacher image.** The paper describes the crop as the region alone, enlarged twice; the released teacher
  images are crops of the student image (red box included) with a margin of about 10% of the box, enlarged twice
  (as the authors explained in the issue tracker). We use the released images as they are.
- **Divergence.** As the code: both sides normalized over the whole support, the tail bucket log(1 − Σp) clamped
  at −1e-7, JSD with β = 0.5. We compute it in fp32 (the code takes the log-softmax in bf16), only at the response
  positions, and over the tokenizer's ids (the code also normalizes over the padding rows of the LM head).
- **Loss averaging.** The code averages over the tokens of each micro-batch and weights the micro-batches by their
  rows; we take one mean over all response tokens of the step. The paper writes a mean per response.
- **EMA checkpoints.** The EMA teacher is saved with the checkpoints and restored on resume; the released code
  does not save it.
- **Importance weights.** The code configures a truncated importance weight (2.0) on the loss; with one update
  per step it is always 1, so it is not used.
- **Image sizes.** The student images go through `data.min_pixels` / `data.max_pixels` (65,536-16,777,216, the
  bounds of the code's processing); the code sends the student images through `qwen_vl_utils` and the teacher
  crops directly to the processor, which gives the same sizes.
- **Validation.** The code has no validation set. The scripts list MMK12 test, which the trainer needs, and only
  validate on it once after the last step (the reward is not meaningful for this prompt).
- **Ablations** of the paper (teacher from the current, initial or trust-region policy; forward or reverse KL;
  512-token responses; distillation on the sampled token only) are not scripted. Overrides for some of them:
  `algorithm.distill_divergence=forward_kl` or `reverse_kl`, `data.max_response_length=512`, and a frozen initial
  teacher with `worker.teacher.source=model worker.teacher.model.model_path=<the same model>`. The trust-region
  teacher and the sampled-token variant with the crop are not implemented.
- **Evaluation** (suite `vision_opd`): the paper judges free-form answers with gpt-oss-120b and matches option
  letters for the multiple-choice benchmarks; we score ZoomBench with rules (see
  [eval/README.md](../../../eval/README.md)) and the others with option-letter matching, with greedy decoding. A
  checkpoint is asked as it was trained (the dataset prompt without a system prompt, 65,536-16,777,216 pixels, the
  non-thinking chat template, read from its run); the suite gives the base models the same prompt. We have
  MME-RealWorld-Lite instead of the full English and Chinese sets; MMVP is not available and CV-Bench is not in the
  suite (`--benchmarks cvbench` adds it).
  Third-party reproductions in the authors' issue tracker reach 1-3 points below the released weights and report
  intermediate steps that score higher than the last one.
- **Data license.** The dataset card states Apache-2.0; the student images are SA-1B images, which Meta
  distributes under its own research license.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository come from
> our unofficial re-implementation, trained once with a single seed on our hardware and software versions and
> evaluated with this repository's harness. They may differ from the official results, including smaller or no
> gains over the baselines, and do not replace them; corrections from the authors are welcome. See
> [About the results](../../../README.md#-about-the-results).

Paper Table 1 (accuracy, %).

| Model | V* | ZoomBench | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Average | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 84.29 | 47.69 | 84.38 | 80.13 | 63.86 | 63.70 | 70.68 | TBD |
| Vision-OPD 4B | 92.15 | 59.76 | 84.50 | 80.38 | 74.88 | 70.76 | 77.07 | TBD |
| Qwen3.5-9B | 82.72 | 52.07 | 85.75 | 80.63 | 71.40 | 67.67 | 73.37 | TBD |
| Vision-OPD 9B | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 | TBD |

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@article{yuan2026visionopd,
  title   = {Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author  = {Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal = {arXiv preprint arXiv:2605.18740},
  year    = {2026}
}
```
