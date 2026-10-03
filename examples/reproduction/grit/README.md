# GRIT — GRIT: Teaching MLLMs to Think with Images

NeurIPS 2025 · [arXiv:2505.15879](https://arxiv.org/abs/2505.15879) ·
[official code](https://github.com/UCSB-AI/GRIT) (moved from `eric-ai-lab/GRIT`) ·
**Unofficial re-implementation in this repository.**

## Method

GRIT trains a model to produce *grounded reasoning chains* in a single turn:
`<think>` reasoning interleaved with bounding-box coordinates `</think>` `<rethink>` … `</rethink>`
followed by the answer. No pixels are fed back after a box is generated. Training uses GRPO-GR,
i.e. GRPO with rewards for the output structure, for producing at least one box (plus a counting
bonus when the number of boxes equals the TallyQA count) and for the answer (a judged answer score
plus 0.1·BLEU-1). There is no reward on box correctness, and the paper trains on only 20
image-question-answer triplets without box or reasoning annotations.

## Setting

| | This repository |
| --- | --- |
| Base models | Qwen2.5-VL-3B-Instruct, InternVL3-2B (`OpenGVLab/InternVL3-2B`) |
| Training data | 20 samples (10 VSR + 10 TallyQA) from `yfan1997/GRIT_data`; images are downloaded from COCO / Visual Genome — `bash scripts/prepare_data.sh grit` |
| Validation | GRIT VSR (288) + TallyQA (491) test sets, greedy, every 50 steps |
| Prompt | official GRIT prompt (`examples/format_prompt/grit.jinja`): the image, then `Question: <question>` and the instruction to think with `bbox_2d` coordinates, then rethink, then answer after `<answer>` |
| Reward | `examples/reward_function/grit.py:compute_score_official`, summed with weight 1: binary answer match + 0.1·BLEU-1 + answer format (≤ 0.5) + repetition term (≤ 0.5, negative when 8-grams repeat back to back, in words and in tokens) + grounded format (0.5 for ≥ 1 box before `<rethink>`, +1.0 if the number of boxes equals a numeric counting answer) + think/rethink structure (≤ 0.5); the repetition and structure terms are GRIT's code |
| Rollout | all 20 prompts x 4 rollouts per step (80 rollouts), T=0.9, top-p 1.0; update batch 20 prompts |
| RL | GRPO; low-var KL loss 0.01; symmetric clip 0.28; loss averaged over each response's tokens, then over responses (`worker.actor.loss_avg_mode=seq`, as GRIT's trainer) |
| Optimization | AdamW (bf16), lr 2e-6 with cosine schedule, vision tower trainable, 200 steps (`trainer.max_steps=200`); the checkpoints of steps 50, 100, 150 and 200 are all kept (`trainer.save_limit=4`) |
| Lengths / pixels | max prompt 1,024, max response 1,024; 3,136-200,704 pixels (256·28·28). InternVL: images at their resolution cut into at most 2 tiles of 448 px plus a thumbnail (`worker.actor.model.max_dynamic_patch=2`), max prompt 1,500 |
| GPUs | 8 |

GRIT changes only the output format and the reward; no `algorithm.*` switch is used. See
[docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md) for the switches of the other methods.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_3b_grpo_grit.sh` | GRIT (GRPO-GR), Qwen2.5-VL-3B |
| `internvl3_2b_grpo_grit.sh` | GRIT (GRPO-GR), InternVL3-2B |

```bash
bash scripts/prepare_data.sh grit
bash examples/reproduction/grit/qwen2_5_vl_3b_grpo_grit.sh
bash scripts/prepare_eval_data.sh grit
bash scripts/eval.sh checkpoints/GRIT-Reproduce/qwen2_5_vl_3b_grpo_grit --suite grit
# InternVL3-2B: images at their resolution, at most 2 tiles, as in training
bash scripts/eval.sh checkpoints/GRIT-Reproduce/internvl3_2b_grpo_grit --suite grit \
    --max-pixels 12845056 --max-dynamic-patch 2
```

The `grit` suite follows GRIT's evaluation: the official GRIT prompt (`examples/format_prompt/grit.jinja`,
no system prompt), the bare question and GRIT's image size (3,136 to 256·28·28 pixels). Checkpoints go
to `checkpoints/GRIT-Reproduce/<script name>`. Append `key=value` overrides to the training command
and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the
environment. Evaluation is described in [eval/README.md](../../../eval/README.md).

## Differences from the paper / official code

- **Answer reward.** The official answer term is a GPT-4o judgement (binarized). We replace it with
  a rule-based normalized match (yes/no answers by the first yes/no word, counting answers by the
  first number, short open-ended answers by containment of the reference phrase, with a `mathruler`
  fallback), so training needs no API access. The other reward terms follow the released code,
  which differs from the paper text (counting bonus 1.0 instead of 0.5, structure term capped at
  0.5, extra answer-format and repetition terms).
- **Batch.** Each step uses all 20 training prompts (80 rollouts). The paper states a total batch of
  128; the released Qwen script uses 1 sample per device without gradient accumulation on 8 GPUs.
  The InternVL run uses the same settings as the Qwen run except for the image tiles and the
  maximum prompt length, which follow the official InternVL script (with InternVL3-2B's default
  of up to 12 tiles, every training prompt exceeds 1,500 tokens).
- **Training length.** We train the paper's 200 steps with a cosine schedule over them. The
  released checkpoints come from runs planned for 500 steps (cosine over 500 steps, per their
  `trainer_state.json`): `GRIT-20-Qwen2.5-VL-3B` stopped at step 150 and `GRIT-20-InternVL-2B` at
  step 50. All four of our checkpoints are kept, so the step-50 and step-150 ones can be compared
  with them.
- **Framework.** EasyR1 with vLLM rollouts instead of the official TRL trainer (no vLLM); maximum
  response 1,024 tokens instead of 1,000.
- **Validation** runs on the VSR and TallyQA test sets only (the official scripts validate on all six
  test sets).
- **Evaluation** (suite `grit`) covers VSR, TallyQA, GQA and OVDEval-position; MathVista and MME are
  not included. Answers are scored with rules instead of GPT-4o: the primary metric is answer
  accuracy with relaxed exact match, and `answer/grit_rule_accuracy` uses the answer rule of the
  training reward. The paper's grounding IoU (union of predicted boxes vs. union of GT boxes) is in
  the metric details: `grounding/grit_iou` reads bracketed boxes, `grounding/grit_iou_grit_pattern`
  any four comma-separated integers, as GRIT's `extract_eval_results.py`. The questions are asked
  without the box instruction other suites append, as the models are trained. The released
  OVDEval file has 2,146 items (the paper lists 2,164).

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Table 1 (ACC = GPT-4o answer score ×100; GIoU = grounding IoU). InternVL3-2B's OVDEval GIoU is
printed on a 0-100 scale in the paper. The paper reports no RL baseline without grounding.

| Benchmark | Paper, Qwen2.5-VL-3B | This repo, Qwen2.5-VL-3B | Paper, InternVL3-2B | This repo, InternVL3-2B |
| --- | --- | --- | --- | --- |
| VSR ACC / GIoU | 72.9 / 0.325 | TBD | 64.9 / 0.495 | TBD |
| TallyQA ACC / GIoU | 47.8 / 0.447 | TBD | 44.2 / 0.324 | TBD |
| GQA ACC / GIoU | 62.8 / 0.485 | TBD | 63.2 / 0.457 | TBD |
| MathVista ACC | 59.8 | not evaluated | 48.2 | not evaluated |
| MME ACC | 89.3 | not evaluated | 82.0 | not evaluated |
| OVDEval GIoU | 0.398 | TBD | 56.0 | TBD |

For reference, Qwen2.5-VL-3B with direct querying scores VSR 49.5, TallyQA 40.8, GQA 55.4 (ACC).
Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{fan2025grit,
  title     = {{GRIT}: Teaching {MLLMs} to Think with Images},
  author    = {Fan, Yue and He, Xuehai and Yang, Diji and Zheng, Kaizhi and Kuo, Ching-Chen and Zheng, Yuting and Narayanaraju, Sravana Jyothi and Guan, Xinze and Wang, Xin Eric},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2505.15879}
}
```
