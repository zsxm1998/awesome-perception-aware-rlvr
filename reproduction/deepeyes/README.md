# DeepEyes — DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

ICLR 2026 · [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) ·
[official code](https://github.com/Visual-Agent/DeepEyes) ·
**Unofficial re-implementation in this repository.**

## Method

DeepEyes trains multi-turn "interleaved multimodal chain-of-thought" with end-to-end RL. After each
reasoning step the model either answers or calls `image_zoom_in_tool` with a bounding box; the crop
of the original image is appended as a new observation (observation tokens are excluded from the
loss) and GRPO optimizes the whole trajectory. The reward combines answer accuracy, a format
penalty and a tool bonus that is granted only when the answer is correct **and** at least one
zoom-in was used.

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct (as in the paper) or Qwen3-VL-8B-Instruct |
| Training data | DeepEyes-Datasets-47k (`ChenShawn/DeepEyes-Datasets-47k`, 47,052 = 22,362 V*-derived + 13,659 ArxivQA charts + 11,031 ThinkLite-VL) — `bash scripts/prepare_data.sh deepeyes` |
| Validation | none during training (`trainer.val_freq=-1`); the prepared `val.parquet` (100 training rows per source) is for monitoring only |
| Interaction | agentic rollout (`worker.rollout.interaction_mode=agentic`); Qwen2.5-VL: `bbox_2d` in absolute pixels of the image the model sees, system prompt `examples/system_prompt/deepeyes_pixel.txt`, chat template `examples/chat_template/qwen2_5_vl_tool_call.jinja`; Qwen3-VL: `bbox_2d` in 0-1000 coordinates, `examples/system_prompt/deepeyes.txt` through the native tool-call template (see [Coordinates](#coordinates)); at most 6 tool calls (`worker.rollout.agent_max_tool_calls=6`), 10,240 tokens per turn, 20,480 tokens per trajectory, at most 16 images per sample |
| Reward | `examples/reward_function/deepeyes.py:compute_score_official`: V* / chart samples 0.8·acc + 0.2·format (0 / −1) + 1.2·tool (tool only if correct and ≥ 1 successful zoom-in); ThinkLite samples 1.2·acc + 0.4·format, no tool bonus |
| Answer judge | OpenAI-compatible judge when `DEEPEYES_JUDGE_BASE_URL`, `DEEPEYES_JUDGE_MODEL` (and optionally `DEEPEYES_JUDGE_API_KEY`) are set (the paper uses Qwen2.5-72B-Instruct served by vLLM); otherwise a rule-based matcher. ThinkLite answers are first checked with `mathruler` |
| Rollout | 256 prompts x 16 rollouts per step, T=1.0, top-p 1.0; one policy update per step (update batch 256 prompts) |
| RL | GRPO; no KL; clip 0.2 / 0.2; no entropy term |
| Optimization | AdamW (bf16), lr 1e-6 constant, vision tower trainable, 80 steps (`trainer.max_steps=80`) |
| Lengths / pixels | max prompt 8,192; 200,704-1,003,520 pixels per input image |
| GPUs | 8 (one node) |

DeepEyes changes the rollout mode and the reward; no `algorithm.*` switch is used.

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_grpo.sh` | "RL with text-only CoT" baseline on Qwen2.5-VL-7B: same data and reward, no tool (`examples/system_prompt/deepeyes_text_only.txt`, max response 10,240, `compute_score_text_only`) |
| `qwen2_5_vl_7b_grpo_deepeyes.sh` | DeepEyes on Qwen2.5-VL-7B (the paper's backbone; absolute pixel coordinates) |
| `qwen3_vl_8b_grpo.sh` | text-only CoT baseline on Qwen3-VL-8B |
| `qwen3_vl_8b_grpo_deepeyes.sh` | DeepEyes on Qwen3-VL-8B (0-1000 coordinates) |

```bash
bash scripts/prepare_data.sh deepeyes
DEEPEYES_JUDGE_BASE_URL=http://<judge-host>:8000/v1 DEEPEYES_JUDGE_MODEL=judge \
    bash reproduction/deepeyes/qwen2_5_vl_7b_grpo_deepeyes.sh
bash scripts/prepare_eval_data.sh deepeyes
bash scripts/eval.sh checkpoints/DeepEyes-Reproduce/qwen2_5_vl_7b_grpo_deepeyes --suite deepeyes
```

The `deepeyes` suite runs the agent loop by default. For the text-only baseline, append
`--interaction-mode one_shot --system-prompt examples/system_prompt/deepeyes_text_only.txt`.
Checkpoints go to `checkpoints/DeepEyes-Reproduce/<script name>`. Append `key=value` overrides to
the training command and set `N_GPUS_PER_NODE`, `NNODES`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or
`EXPERIMENT_NAME` in the environment. Evaluation is described in [eval/README.md](../../eval/README.md).

## Coordinates

Qwen3-VL (and InternVL) write boxes as integers normalized to 0-1000, which the tool scales to the
source image directly. Qwen2-VL and Qwen2.5-VL write **absolute pixel coordinates of the image they
see**, and that image is not the source image: the data pipeline first resizes it into
`[data.min_pixels, data.max_pixels]`, and the processor then rounds height and width to multiples
of 28 (`smart_resize`). A 3000x2000 photo, for example, reaches the model as 1232x812, so a box
taken literally on the source image would be shifted and shrunk by about 2.4x. With
`worker.rollout.agent_bbox_format=pixel` the tool

1. computes for every source image the exact frame the model sees (the same `smart_resize` the
   processor applies, with the processor's own pixel limits),
2. clips the predicted box to that frame and scales it to the source resolution (floor/ceil, so
   the crop never loses pixels), and crops the **original-resolution** image,
3. returns `bbox_2d` to the model unchanged (in its own frame) and logs the box both in source
   pixels and as 0-1000 of the source image (`bbox_norm1000`), so pixel and 0-1000 runs can be
   compared.

The default `agent_bbox_format=auto` picks `pixel` for `qwen2_vl` / `qwen2_5_vl` checkpoints and
`norm1000` otherwise. Configuration checks refuse a system prompt that describes the other
convention, and refuse Qwen2-VL / Qwen2.5-VL without the tool-call chat template: their stock
template drops the tool definitions and prints the `tool` role verbatim.
`examples/chat_template/qwen2_5_vl_tool_call.jinja` adds the Hermes tool format of Qwen2.5
(`<tools>`, `<tool_call>`, `<tool_response>` with images) and is otherwise identical to the stock
template.

`scripts/check_bbox_mapping.py` checks a backbone's convention end to end: the model grounds the
245 V* target objects in its native format on the image the agent loop feeds it, the tool maps
the boxes to the source image, and the result is compared with the V* annotations. With the
default pixel limits (greedy decoding, 2026-10-02):

| Backbone (convention) | Tool mapping | Coordinates read as source pixels | Data-side resize only (no 28-px rounding) |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct (`pixel`) | mean IoU 0.54 | 0.00 | 0.43 |
| Qwen3-VL-8B-Instruct (`norm1000`) | mean IoU 0.55 | 0.00 | – |

```bash
bash scripts/prepare_eval_data.sh vstar
python scripts/check_bbox_mapping.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

## Differences from the paper / official code

- **Backbone.** The `qwen2_5_vl_7b_*` scripts use the paper's Qwen2.5-VL-7B-Instruct. The
  `qwen3_vl_8b_*` scripts are an extra setting on Qwen3-VL-8B-Instruct.
- **Box mapping.** The official code feeds images of up to 16384·28·28 pixels, so for most images
  the model's frame differs from the original only by the rounding to multiples of 28, and it
  crops the original image with the predicted coordinates as they are. Here inputs are resized to
  at most 1,003,520 pixels (`data.max_pixels`), so the box is mapped from the model's frame back to
  the original resolution explicitly (see [Coordinates](#coordinates)). As in the official tool,
  a box whose shorter side is at most 30 pixels of the original image is rejected with a tool error.
- **Judge.** The official reward uses a Qwen2.5-72B-Instruct judge; here the judge is optional and a
  rule-based matcher is used when no judge is configured. Results without a judge are not directly
  comparable.
- **ThinkLite samples** go through the same tool-enabled rollout (the official environment disables
  the tool for them) but keep the math reward without tool bonus.
- **Length of training.** 80 steps with at most 6 tool calls, following the paper ("80
  iterations", "up to 6" active perceptions); the released 7B script uses `total_epochs=32` and
  `max_turns=5`.
- **Hardware.** One 8-GPU node by default; the official README recommends at least 32 GPUs for the
  7B model, so expect long steps.
- **Evaluation** (suite `deepeyes`) covers V* Bench, HR-Bench 4K / 8K, MME-RealWorld-Lite and POPE
  with rule-based option-letter matching instead of rule-based matching followed by a Qwen2.5-72B
  judge. Tables 3-4 are not in the suite: refCOCO / refCOCO+ / refCOCOg are available as
  `--suite refcoco` and the reasoning benchmarks through their own keys (e.g. `mathvista`), while
  ReasonSeg is not included. DeepEyes feeds
  images of up to 16384·28·28 pixels (`--max-pixels 12845056` to match). POPE's primary metric is
  greedy macro-F1 over the three splits, while the paper reports accuracy.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../README.md#about-the-results).

Paper Tables 1-3 (Qwen2.5-VL-7B backbone, accuracy %; MME-RealWorld-Lite is the Overall column of
Table 2); text-only CoT from paper Table 9 (v3).
The Qwen3-VL-8B runs are an extra setting, so only their gains over the base model are comparable
with the paper.

| Benchmark | Qwen2.5-VL-7B (paper) | RL text-only CoT (paper) | DeepEyes-7B (paper) | Qwen2.5-VL-7B (this repo) | `qwen2_5_vl_7b_grpo` | `qwen2_5_vl_7b_grpo_deepeyes` | Qwen3-VL-8B (this repo) | `qwen3_vl_8b_grpo` | `qwen3_vl_8b_grpo_deepeyes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V* Bench | 71.2 | 88.5 | 90.1 | TBD | TBD | TBD | TBD | TBD | TBD |
| HR-Bench 4K | 68.8 | 75.4 | 75.1 | TBD | TBD | TBD | TBD | TBD | TBD |
| HR-Bench 8K | 65.3 | 60.8 | 72.6 | TBD | TBD | TBD | TBD | TBD | TBD |
| MME-RealWorld-Lite | 42.3 | – | 53.2 | TBD | TBD | TBD | TBD | TBD | TBD |
| POPE (overall) | 85.9 | – | 87.7 | TBD | TBD | TBD | TBD | TBD | TBD |

Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{zheng2026deepeyes,
  title     = {{DeepEyes}: Incentivizing ``Thinking with Images'' via Reinforcement Learning},
  author    = {Zheng, Ziwei and Yang, Michael and Hong, Jack and Zhao, Chenxiao and Xu, Guohai and Yang, Le and Shen, Chao and Yu, Xing},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2505.14362}
}
```
