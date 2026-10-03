# Evaluation

One-command evaluation of the base models and checkpoints produced by this repository.
The harness covers the benchmarks used by the reproduced papers (PAPO, VPPO, DVRP, ToR,
PGPO, PEPO, CFPO, VEPO, GRIT, DeepEyes) and CGPO, downloads them from public sources, runs
inference with vLLM on all visible GPUs, and scores every benchmark with a documented,
rule-based protocol (an LLM judge is only used for MM-Vet and, optionally, GQA).

```
eval/
├── run_all_benchmarks.py      # the runner (inference worker pool + serial scoring)
├── evaluate.py                # one-click wrapper used by scripts/eval.sh (checkpoint discovery/merging)
├── compare_tags.py            # paired significance test between two runs
├── viz_server.py              # browser UI for predictions, boxes and metrics
├── config/benchmarks.yaml     # benchmark registry
├── config/suites.yaml         # named benchmark lists that mirror each paper's tables
├── prepare/                   # data preparation (python -m eval.prepare ...)
└── easyr1_eval/               # loaders, scorers, backends, summaries, ...
```

## Quickstart

```bash
# 1. Download the benchmarks of a suite into data/eval/ (idempotent; re-run to resume).
bash scripts/prepare_eval_data.sh papo

# 2. Evaluate a Hugging Face model ...
bash scripts/eval.sh Qwen/Qwen2.5-VL-3B-Instruct --suite papo

# ... or a training run: the latest global_step_* is merged (FSDP -> HF) and evaluated.
bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo
bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo --all-steps

# Quick smoke test: first 32 samples of a few benchmarks.
bash scripts/eval.sh Qwen/Qwen2.5-VL-3B-Instruct --benchmarks geo3k,mathvista,pope,hallusionbench,vstar --limit 32
```

The run prints a score table and writes `eval/results/<run_name>/<step>/summary.csv`; every
run also becomes one row of `eval/results/summary.csv`. Use `--gpus 0,1` or
`CUDA_VISIBLE_DEVICES` to choose GPUs (default: all visible GPUs, one vLLM engine per GPU;
`--tp N` groups N GPUs per engine).

Requirements: the training environment (vLLM, transformers, `mathruler`), plus `flask` for
the visualization server. If huggingface.co is slow or blocked, set
`HF_ENDPOINT=https://hf-mirror.com` before preparing data.

## Data preparation

```bash
bash scripts/prepare_eval_data.sh --list                 # benchmarks, sources, sizes, suites
bash scripts/prepare_eval_data.sh geo3k pope vstar       # individual benchmarks
bash scripts/prepare_eval_data.sh vppo deepeyes          # suites
bash scripts/prepare_eval_data.sh all                    # every non-optional benchmark (~11 GB on disk)
bash scripts/prepare_eval_data.sh refcoco                # optional: RefCOCO family (~0.5 GB)
bash scripts/prepare_eval_data.sh seed_bench             # optional: SEED-Bench (27 GB download, ~25 GB on disk)
python -m eval.prepare papo --data-root /data/eval       # same CLI as a module (from the repo root)
```

- Data root: `--data-root`, else `$EVAL_DATA_ROOT`, else `$DATA_ROOT/eval`, else
  `<repo>/data/eval`. The runner, the wrapper and the viz server use the same rule.
- Each benchmark lands in `data/eval/<benchmark>/` (the RefCOCO splits share
  `data/eval/refcoco/`) in exactly the layout the loaders read, together with a
  `.prepared-<benchmark>.json` marker that records the source and the row count. Prepared
  benchmarks are skipped; `--force` re-downloads. Downloads go through `data/eval/.raw/` and
  are deleted after extraction unless `--keep-raw` is given.
- Hugging Face downloads retry with back-off and fewer workers (`--hf-workers`, default 8).
  COCO / Visual Genome images are fetched one by one (`--http-workers`, default 16), so only
  the ~4k images that the GRIT and RefCOCO sets reference are downloaded instead of the full
  archives. A failed benchmark does not stop the others: the command exits with a non-zero
  status and re-running it resumes.
- `all` leaves out the optional benchmarks (RefCOCO and SEED-Bench, which is ~27 GB);
  `--include-optional` adds them.
- Before a benchmark that needs more than 2 GB is downloaded, the free space of the data-root
  filesystem is checked against its requirement (temporary archives included; see `--list`)
  and the command stops with a clear message when it is short (`--skip-space-check` to
  override). Large downloads are announced with their size.
- Interrupted runs resume: Hugging Face downloads continue partial files, per-image downloads
  skip the images already on disk, archives are extracted member by member through temporary
  files, and SEED-Bench is converted shard by shard with a marker per shard (each parquet
  shard is deleted once converted, so the peak disk use stays close to the final size).
- The runner checks that every selected benchmark is prepared before it loads a model and
  prints the exact `prepare_eval_data.sh` command otherwise (`--skip-missing-data` evaluates
  the available ones instead).

## Suites

`--suite NAME` (several: `--suite papo,tor`) selects the benchmarks of a paper table.
`--benchmarks` adds benchmarks to the suite and `--skip-benchmarks` removes them. Some suites
also set prompt defaults (see [Prompts](#prompts)); explicit flags always win.

| Suite | Benchmarks | Paper table / notes |
|---|---|---|
| `papo` | geo3k, mathvista, wemath, mmk12, mathverse, logicvista, clevr_count, mmmu_pro, mathverse_v | PAPO Table 1 (same data and scoring as PAPO-Eval) |
| `vppo` | mathverse, dynamath, mmk12, geo3k, mathvision, wemath, logicvista, mmmu_pro | VPPO Table 1; the paper uses 4096 new tokens for 32B models (`--max-new-tokens 4096`) |
| `dvrp` | geo3k, mathvista, wemath, mathverse, mathverse_v, mmk12 | DVRP Table 1, general columns; medical columns not included; DVRP reports top_p 0.9/0.99, we use 1.0 |
| `pgpo` | geo3k, mmk12, mathverse, dynamath, mathvision, logicvista, mmmu_pro, mathverse_v | PGPO Table 1; PGPO evaluates DynaMath "sample variant1", we use the VPPO-Eval DynaMath file |
| `pepo` | geo3k, mathvista, wemath, mathverse, logicvista, clevr_count, mmmu_pro | PEPO Table 1 + Table 5 (ViRL39K scaling); Geometry3K val is not included |
| `pepo_geometry` | geo3k, mathvista, mathverse, logicvista | PEPO Table 1 with the prompt of the Geometry3K-trained models (`pepo.jinja`, `<answer>` tags) |
| `vepo` | geo3k, mmk12, hallusionbench, mathvista, wemath, mathverse, mathvision | VEPO Table 1; the paper uses greedy decoding and a gpt-4o-mini judge for 4 benchmarks, we use rule-based avg@8 |
| `tor` | mathverse, mathvision, mathvista, wemath, hallusionbench | ToR Table 4; the paper follows NoisyRollout (greedy + Gemini answer parsing), we use rule-based avg@8 |
| `cfpo` | cvqa_real, mars_bench, pope, textvqa, mmmu_pro, geo3k, wemath, mmk12, mathverse, logicvista | CFPO Table 1; CFPO reports POPE as pooled accuracy with avg@8 (same 9,000 questions; our primary is greedy macro F1, pooled accuracy in the details); CFPO's LogicVista file has 448 items, PAPO-Eval's 447 |
| `grit` | grit_vsr, grit_tallyqa, grit_gqa, ovdeval_position | GRIT Table 1 (GRIT judges answers with GPT-4o, we use relaxed exact match); default prompt: `grit_GR.txt` |
| `deepeyes` | vstar, hrbench_4k, hrbench_8k, mme_realworld_lite, pope | DeepEyes Tables 1-3; default: agentic DeepEyes inference (see [Agentic evaluation](#agentic-evaluation-deepeyes)) |
| `cgpo` | = `papo` | CGPO natural-image reproduction (trained on ViRL39K); default prompt `xml_grounded_reasoning.jinja` |
| `comparison` | `papo` + `vppo` + pope, hallusionbench | the benchmark set used by `examples/comparison/` |
| `perception` | pope, hallusionbench, mme, gqa, mm_vet | general perception / hallucination |
| `refcoco` | refcoco_val, refcoco_plus_val, refcocog_val | optional (COCO images) |
| `all` | every non-optional benchmark | |

The reasoning columns of all papers are scored with one protocol (PAPO-Eval, below), which
is what PAPO, VPPO, DVRP, PGPO, PEPO (Table 5) and CFPO use. Where a paper used another
protocol (VEPO and ToR: greedy decoding and an LLM answer parser; PEPO Table 1: the official
MathVista checker; GRIT: a GPT-4o judge; DeepEyes: a Qwen2.5-72B judge fallback), numbers
are comparable across methods within this harness but not digit-for-digit with the paper.
`--temperature 0 --num-samples 1` switches any run to greedy decoding.

## Benchmarks

Sample counts are the rows actually evaluated. "avg@8" = mean accuracy over 8 samples at
temperature 1.0, top_p 1.0 and 2048 new tokens; "greedy" = 1 sample at temperature 0.

| Key | Benchmark | Source (HF) | #Samples | Primary metric | Decoding | Used by |
|---|---|---|---:|---|---|---|
| `geo3k` | Geometry3K test | `PAPO-Galaxy/PAPO_eval` | 601 | boxed exact match, mean acc@k | avg@8 | PAPO, VPPO, DVRP, PGPO, PEPO, CFPO, VEPO, CGPO |
| `mathvista` | MathVista testmini | `PAPO-Galaxy/PAPO_eval` | 1000 | boxed exact match | avg@8 | PAPO, DVRP, PEPO, VEPO, ToR, CGPO |
| `wemath` | We-Math | `PAPO-Galaxy/PAPO_eval` | 1740 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PEPO, CFPO, VEPO, ToR, CGPO |
| `mmk12` | MMK12 test | `PAPO-Galaxy/PAPO_eval` | 2000 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PGPO, CFPO, VEPO, CGPO |
| `mathverse` | MathVerse testmini (multi-choice) | `PAPO-Galaxy/PAPO_eval` | 2180 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PGPO, PEPO, CFPO, VEPO, ToR, CGPO |
| `mathverse_v` | MathVerse vision-dependent | `PAPO-Galaxy/PAPO_eval` | 1308 | boxed exact match | avg@8 | PAPO, DVRP, PGPO, CGPO |
| `logicvista` | LogicVista | `PAPO-Galaxy/PAPO_eval` | 447 | boxed exact match | avg@8 | PAPO, VPPO, PGPO, PEPO, CFPO, CGPO |
| `clevr_count` | Counting (PAPO's "SuperClevr counting" column) | `PAPO-Galaxy/PAPO_eval` | 200 | boxed exact match | avg@8 | PAPO, PEPO, CGPO |
| `mmmu_pro` | MMMU-Pro (vision setting) | `PAPO-Galaxy/PAPO_eval` | 1730 | boxed exact match | avg@8 | PAPO, VPPO, PGPO, PEPO, CFPO, CGPO |
| `dynamath` | DynaMath (verifiable subset) | `chamber111/VPPO-Eval` | 3666 | boxed exact match | avg@8 | VPPO, PGPO |
| `mathvision` | MathVision (verifiable subset) | `chamber111/VPPO-Eval` | 2907 | boxed exact match | avg@8 | VPPO, PGPO, VEPO, ToR |
| `pope` | POPE random/popular/adversarial | `lmms-lab/POPE` | 9000 | macro F1 over the 3 splits (accuracy in details) | greedy | CFPO, DeepEyes |
| `hallusionbench` | HallusionBench (image questions) | `lmms-lab/HallusionBench` | 951 | question accuracy aAcc (fAcc, qAcc in details) | greedy | VEPO, ToR |
| `mme` | MME | `lmms-lab/MME` | 2374 | total score (acc + acc+); /2800 in averages | greedy | PGPO |
| `gqa` | GQA testdev-balanced | `lmms-lab/GQA` | 12578 | exact match (EM→judge cascade with a judge) | greedy | general |
| `mm_vet` | MM-Vet | `lmms-lab/MMVet` | 218 | LLM-judge score; **skipped without a judge** | greedy | general |
| `cvqa_real` | C-VQA-Real (counterfactual VQA on COCO) | `RavenInJuly/CFPO_Datasets` + COCO val2014 | 6288 | CFPO match, mean acc@k | avg@8 | CFPO |
| `mars_bench` | MARS-Bench | `RavenInJuly/CFPO_Datasets` + COCO val2014 | 5110 | CFPO match, mean acc@k | avg@8 | CFPO |
| `textvqa` | TextVQA val | `RavenInJuly/CFPO_Datasets` | 5000 | CFPO match, mean acc@k (VQA accuracy in details) | avg@8 | CFPO |
| `seed_bench` | SEED-Bench (optional, ~27 GB) | `lmms-lab/SEED-Bench` | 17990 | option accuracy (image / video split in details) | greedy | general |
| `vstar` | V* Bench | `craigwu/vstar_bench` | 191 | accuracy over all questions | greedy | DeepEyes |
| `hrbench_4k` | HR-Bench 4K | `DreamMr/HR-Bench` | 800 | mean of FSP and FCP accuracy | greedy | DeepEyes |
| `hrbench_8k` | HR-Bench 8K | `DreamMr/HR-Bench` | 800 | mean of FSP and FCP accuracy | greedy | DeepEyes |
| `mme_realworld_lite` | MME-RealWorld-Lite | `yifanzhang114/MME-RealWorld-lite-lmms-eval` | 1919 | accuracy over all questions (Perception / Reasoning and per-subtask in details) | greedy | DeepEyes |
| `grit_vsr` | GRIT VSR | `yfan1997/GRIT_data` + COCO 2017 | 288 | answer accuracy (GRIT grounding IoU in details) | greedy | GRIT |
| `grit_tallyqa` | GRIT TallyQA (official file) | `yfan1997/GRIT_data` + Visual Genome | 491 | answer accuracy (+ GRIT grounding IoU) | greedy | GRIT |
| `grit_gqa` | GRIT GQA | `yfan1997/GRIT_data` + Visual Genome | 509 | answer accuracy (+ GRIT grounding IoU) | greedy | GRIT |
| `ovdeval_position` | OVDEval position | `yfan1997/GRIT_data` + `omlab/OVDEval` | 2146 | GRIT grounding IoU (Acc@0.5 in details) | greedy | GRIT |
| `refcoco_val` | RefCOCO val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 10834 | Acc@0.5 IoU | greedy | DeepEyes, PEPO |
| `refcoco_plus_val` | RefCOCO+ val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 10758 | Acc@0.5 IoU | greedy | DeepEyes |
| `refcocog_val` | RefCOCOg val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 4896 | Acc@0.5 IoU | greedy | DeepEyes |

Not included: the medical benchmarks of DVRP, LISA grounding and the few-shot / puzzle tasks
of PEPO, and GRIT's GPT-judged MathVista/MME subsets.

### Scoring protocols

- **Reasoning (`boxed_exact_match`)** reproduces PAPO-Eval (`papo_eval/eval_utils.py`):
  whitespace around `<`, `>` and `/` is removed with `re.sub(r"\s*(<|>|/)\s*", r"\1", ...)`,
  the last `\boxed{}` is extracted with `mathruler.grader.extract_boxed_content` and compared
  to the reference with `mathruler.grader.grade_answer`. The score is the mean over the k
  samples of each question, averaged over questions (mean acc@k; `pass_at_k` is in the
  details). A response without any `\boxed{` falls back to its last `<answer>...</answer>`
  block (for models trained with an `<answer>`-tag prompt); this never changes the score of a
  response that has a box. The PAPO-Eval MathVista split keeps some free-form answers that
  exact match cannot always credit; like PAPO and VPPO we still use exact match.
- **POPE / MME / HallusionBench** parse yes/no from the final answer (`\boxed{}`,
  `<answer>`, "answer is ..." or the first yes/no word). HallusionBench reports question
  accuracy (aAcc, primary), figure accuracy (fAcc: every question about a figure correct) and
  question-pair accuracy (qAcc) as defined by the official evaluation.
- **V\* / HR-Bench (`mcq`)** extract the chosen option letter from the final answer (`C`,
  `(C)`, `C.`, `C. text`, `Option C`, ...) and fall back to a unique option-text match.
  V\* reports accuracy over all 191 questions; HR-Bench reports the mean of the FSP
  (`single`) and FCP (`cross`) accuracies, as DeepEyes does. DeepEyes additionally asks an
  LLM judge when its rules fail; we do not (the unparsed rate is in the details).
- **MME-RealWorld-Lite (`mcq`)** uses the lmms-eval prompt (question, "The choices are listed
  below:", the `(A)`-`(E)` options and "Select the best answer ... Respond with only the
  letter (A, B, C, D, or E) of the correct option."; the trailing completion cue "The best
  answer is:" of the base-model prompt is dropped) and the same letter extraction as V\*. The
  primary metric is accuracy over all 1,919 questions (lmms-eval's overall score); the details
  hold `accuracy_by_task` (Perception, Reasoning) and `accuracy_by_category`
  (`Perception/OCR`, `Perception/RS`, `Perception/DT`, `Perception/MO`, `Perception/AD`,
  `Reasoning/OCR`, `Reasoning/DT`, `Reasoning/MO`, `Reasoning/AD`), the breakdown of the
  DeepEyes table.
- **C-VQA-Real / MARS-Bench / TextVQA (`cfpo_match`)** reproduce CFPO's
  `Counterfactual-Eval/inference_eval.py`: the last `\boxed{}` is extracted with
  `mathruler.grader.extract_boxed_content` (the whole response when there is no box) and
  compared with `grade_answer`; if that fails, the part before a `:` is tried, and `No.`
  counts as `No`. TextVQA references are the 10 human answers and any match counts. The
  score is the mean over the 8 samples (avg@8, as CFPO). The details add accuracy by
  counterfactual / non-counterfactual question and by question type (`cf`, `ncf`, `<type>`,
  `<type>_cf`) and, for TextVQA, the standard VQA accuracy `min(#matching humans / 3, 1)` on
  the extracted answer.
- **SEED-Bench** asks for the option letter ("Answer with the option letter from the given
  choices directly.") and scores option accuracy over all 17,990 questions. The 3,757 video
  questions (dimensions 10-12) are evaluated as multi-image prompts with the 8 frames that
  lmms-eval extracted; `seed_image_accuracy`, `seed_video_accuracy` and
  `accuracy_by_question_type` are in the details.
- **GQA** uses the official normalized exact match. With a judge configured, exact-match-wrong
  answers are judged for semantic equivalence (synonyms, number words, morphology; prompt in
  `scorers.py`); the exact-match score is always kept as `exact_match_accuracy`.
- **MM-Vet** uses the official few-shot grading prompt with the configured judge model.
- **GRIT sets (`answer_bbox`)** report the two GRIT-paper metrics. The primary is answer
  accuracy: the final answer is checked with relaxed exact match (articles, number words and
  punctuation normalized; GRIT itself uses a GPT-4o judge). The GRIT grounding IoU
  (`grounding/grit_iou`, also the `<key>_giou` column of the global summary) mirrors
  `extract_eval_results.py` of UCSB-AI/GRIT: per sample, the IoU between the union of *all*
  boxes written in the response and the union of the GT boxes; samples with an empty GT list
  are skipped and samples without a predicted box score 0. Box precision/recall/F1 at IoU 0.5
  (one-to-one matching) and box counts are also reported. For agentic runs the predicted boxes
  are the zoom-in regions the agent committed.
- **OVDEval** reports only the GRIT grounding IoU (as GRIT does), with Acc@0.5 IoU in the
  details. **RefCOCO** reports Acc@0.5 IoU of the best box of the final answer
  (`\boxed{}` / `<answer>`, falling back to the whole response).
- **Box coordinates** (`--box-format`, default `auto`): `norm1000` reads `[x1, y1, x2, y2]` in
  0-1000 (values in 0-1 are also accepted), the convention of Qwen3-VL and of every
  grounding prompt in this repository. `pixel` reads absolute pixels of the image the model
  actually saw (Qwen2-VL / Qwen2.5-VL convention): the backend records each image's size
  after the runner's `--min-pixels`/`--max-pixels` resize and the processor's own resize
  (multiples of patch size x merge size), and boxes are divided by that size, which maps them
  back onto the original image. In pixel mode the grounding prompts ask for "absolute pixel
  coordinates" instead of "coordinates normalized to 0-1000". `auto` picks `pixel` for
  `qwen2_vl` / `qwen2_5_vl` checkpoints (read from `config.json`, or the model name) unless the
  system or format prompt asks for 0-1000 coordinates (CGPO, GRIT-JSON and DeepEyes prompts),
  and `norm1000` otherwise. The decision is printed at start-up and stored as `box_format` in
  the summaries; pass `--box-format norm1000` for a Qwen2.5-VL model trained to emit 0-1000
  boxes with a prompt that does not say so.
- Responses that hit the token limit without a final answer count as wrong (the PAPO-Eval
  reasoning scorer, like the original, only looks for the box). Truncation rates are
  reported for every benchmark (`generation/*` details and `truncated_rate` in `summary.csv`).

## Prompts

Loaders return the bare question; the runner renders it the way training does:

- `--format-prompt` (Jinja template with `{{ content }}`, same as `data.format_prompt`).
  Default for one-shot runs: `examples/format_prompt/math_perception.jinja`, i.e. the PAPO
  instruction ("... enclosed within <think> </think> tags. Then, provide your final answer
  enclosed within \boxed{}."), which makes the reasoning prompts identical to PAPO-Eval's.
  `--format-prompt none` sends the bare question.
- `--system-prompt FILE` (same as `data.system_prompt`; `none` for no system prompt) and
  `--prompt-mode chat` (default: the model's chat template) or `raw`.
- `--plain-think-tokens auto|true|false` (default `auto`) tokenizes `<think>` / `</think>` as plain
  text for models in which they are untrained added tokens (Qwen3-VL Instruct), as training does
  (`worker.actor.model.plain_think_tokens`); evaluate a checkpoint with the setting it was trained with.
- `--chat-template FILE` replaces the processor's chat template (same as
  `data.override_chat_template`); recorded in `summary.csv`.

Evaluate a model with the prompt it was trained with:

| Model | Flags (or suite default) |
|---|---|
| PAPO, VPPO, DVRP, PGPO, ToR, GRPO/DAPO baselines | default (`math_perception.jinja`) |
| CFPO | `--format-prompt none --system-prompt examples/system_prompt/cfpo.txt` (suite `cfpo`) |
| VEPO | `--format-prompt none --system-prompt examples/system_prompt/vepo.txt` (suite `vepo`) |
| PEPO (Geometry3K setting) | `--format-prompt examples/format_prompt/pepo.jinja` (suite `pepo_geometry`) |
| CGPO | `--format-prompt examples/format_prompt/xml_grounded_reasoning.jinja` (suite `cgpo`) |
| GRIT | `--format-prompt none --system-prompt examples/system_prompt/grit_GR.txt` (suite `grit`) |
| DeepEyes | `--interaction-mode agentic --system-prompt examples/system_prompt/deepeyes.txt` (suite `deepeyes`; `deepeyes_pixel.txt` for Qwen2-VL / Qwen2.5-VL) |

## LLM judge

The judge is off by default (`--judge-provider none`): MM-Vet is skipped with a warning and
GQA is scored with exact match. No other benchmark needs a judge.

```bash
# any OpenAI-compatible endpoint (OpenAI, a local vLLM/SGLang server, ...)
export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini
bash scripts/eval.sh <model> --suite perception --judge-provider openai

# DeepSeek (default model deepseek-v4-flash; override with --judge-model)
export DEEPSEEK_API_KEY=...
bash scripts/eval.sh <model> --suite perception --judge-provider deepseek
```

`--judge-model`, `--judge-base-url` and `--judge-api-key` override the environment;
`EVAL_JUDGE_PROVIDER` sets the default provider. `--judge-concurrency` (default 4) bounds the
parallel requests and transient HTTP errors are retried. Verdicts are cached (MM-Vet per run
in `metrics/mm_vet_judge_cache.jsonl`, GQA across runs in `eval/results/_judge_cache/`), so
re-scoring is free. Only compare models scored with the same judge.

Scoring is fault tolerant: if one benchmark's scorer fails (a judge outage, a malformed file,
...) the error is logged, the remaining benchmarks are still scored and summarized, and the
command exits with status 1 listing the failed benchmarks. Re-running the same command
resumes (finished inference and scoring are skipped).

## One-click wrapper and checkpoints

`bash scripts/eval.sh <model_or_ckpt> [wrapper options] [runner options]`

| `<model_or_ckpt>` | What is evaluated | Results directory |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` (HF id) | the model | `eval/results/Qwen2.5-VL-3B-Instruct/` |
| merged HF directory | the directory | `eval/results/<dir name>/` |
| `.../<run>/global_step_N/actor` | the merged `actor/huggingface`, or `actor/` itself after `scripts/finalize_run.py` | `eval/results/<run>/global_step_N/` |
| `.../<run>/global_step_N` | its `actor/` | same |
| `.../<run>` (contains `global_step_*`) | the latest step (`--all-steps`: every step) | one directory per step |

FSDP shards (`model_world_size_*_rank_*.pt`) are merged with `scripts/model_merger.py` into
`<actor>/huggingface` the first time (the shards are kept; `--no-merge` refuses instead).
Wrapper options: `--all-steps`, `--run-name NAME`, `--results-root DIR` (default
`eval/results`), `--no-merge`. All other options go to the runner (`--help-runner` lists them).

Common runner options:

| Option | Meaning |
|---|---|
| `--suite`, `--benchmarks`, `--skip-benchmarks` | benchmark selection |
| `--limit N` | only the first N samples of every benchmark (smoke tests) |
| `--temperature`, `--num-samples`, `--top-p`, `--max-new-tokens` | override the registry decoding settings for every benchmark |
| `--gpus`, `--tp`, `--gpu-memory-utilization`, `--max-model-len` | engine placement and size |
| `--min-pixels`, `--max-pixels` | image resizing (default 200704 to 1003520 pixels, as in training) |
| `--box-format auto\|norm1000\|pixel` | how predicted boxes are read (see [Scoring protocols](#scoring-protocols)) |
| `--batch-size`, `--max-batch-images` | request batching per engine |
| `--data-root`, `--skip-missing-data` | data location / tolerate unprepared benchmarks |
| `--resume` / `--no-resume`, `--score-only`, `--summary-only` | reuse or rebuild results |
| `--dry-run` | print the plan (benchmarks, decoding, missing data) without loading a model |
| `--backend transformers` | slow Hugging Face backend for debugging (`dummy` echoes the reference) |

Inference runs one vLLM engine per GPU group. As soon as every shard of a benchmark is done
it is scored on a single scoring thread while the GPUs continue with the next benchmark. Set
`EASYR1_WORKER_PORT_BASE` to different values when several runners share a machine.

## Agentic evaluation (DeepEyes)

```bash
bash scripts/eval.sh <deepeyes checkpoint> --suite deepeyes
# the same with explicit flags, for any benchmark:
bash scripts/eval.sh <ckpt> --benchmarks vstar,hrbench_4k --interaction-mode agentic \
    --agent-profile deepeyes --system-prompt examples/system_prompt/deepeyes.txt
```

The agent zooms into image regions with `image_zoom_in_tool` (at most
`--agent-max-tool-calls`, default 6) and answers in `<answer>` tags, which the same scorers
as in one-shot runs evaluate. Agent diagnostics (tool calls, failures, stop reasons) are
stored with every prediction and summarized as `agent/*` metric details. DeepEyes feeds
images of up to 16384*28*28 pixels; pass `--max-pixels 12845056` to match. To evaluate a
non-agent model on the DeepEyes suite, add `--interaction-mode one_shot --format-prompt
examples/format_prompt/math_perception.jinja --system-prompt none`.

Tool coordinates follow training (`worker.rollout.agent_bbox_format`). In agentic runs
`--box-format auto` resolves from the model alone: Qwen2-VL / Qwen2.5-VL write `bbox_2d` in
absolute pixels of the image as they see it (after the `--min-pixels`/`--max-pixels` resize
and the processor's `smart_resize`), which the tool maps back to the source image; other
models write 0-1000. For a pixel run the `deepeyes` suite swaps its system prompt for
`examples/system_prompt/deepeyes_pixel.txt`, and Qwen2-VL / Qwen2.5-VL runs load
`examples/chat_template/qwen2_5_vl_tool_call.jinja`, because their stock chat template ignores
tool definitions (override with `--chat-template`). An explicit `--system-prompt` that asks
for the other convention is rejected. Pass `--box-format norm1000` for a Qwen2.5-VL checkpoint
trained with 0-1000 tool boxes. Committed crops are stored per prediction as
`committed_tool_regions` with `bbox_norm1000` (the crop on the source image in 0-1000, the
same for both conventions, used by the grounding scorers) next to the model's raw `bbox_2d`.

## Output layout

```
eval/results/<run_name>/<step>/
├── summary.csv                                  # one row per benchmark + group averages + overall
├── predictions/<benchmark>/predictions.jsonl    # prompt, messages, image paths, responses, metadata
├── metrics/<benchmark>.json                     # primary score, details, generation diagnostics
├── metrics/<benchmark>_per_sample*.jsonl        # per-sample scores / extracted answers
└── state/                                       # resume fingerprints
eval/results/summary.csv                         # one row per run, one column per benchmark
```

`normalized_score_0_100` is the primary metric on a 0-100 scale (MME total / 2800 * 100);
group and overall averages use it. Benchmarks that were skipped (no judge), or whose primary
metric does not apply, are listed
but left out of the averages. For the GRIT sets the global summary also has a `<key>_giou`
column with the GRIT grounding IoU.

## Comparing runs

```bash
python eval/compare_tags.py <run_a> <run_b> [--benchmarks pope,mathvista] [--json report.json]
```

Runs are result directories or paths under `eval/results/` (e.g. `exp/global_step_200`).
Per-sample scores are recomputed from the stored predictions with the current scorers. Each
benchmark gets a paired bootstrap (10k replicates) delta with a 95% CI and p-value, a
McNemar test for binary metrics and Benjamini-Hochberg q-values; group and overall averages
are tested the same way. Judge-based scores are read from the cached verdicts only.

## Visualization server

```bash
pip install flask   # if it is not installed
python eval/viz_server.py [--results eval/results] [--data-root data/eval] [--port 7777]
```

Browse predictions per benchmark with images, predicted and ground-truth boxes, extracted
answers and per-sample metrics, and diff two runs. The server listens on 127.0.0.1 (use an
SSH tunnel, or `--host 0.0.0.0` on a trusted network only) and only serves files from the
results directory and the data root.

## Robustness evaluation (optional)

`--perturbation.type random_patch|gaussian_noise|vp_diffusion` (pixel space; `random_patch`
and `vp_diffusion` use the training functions, while `gaussian_noise` adds its `std` to [0, 1]
pixels and clips, unlike the training-side `gaussian_noise`, which adds it to the image
processor's normalized `pixel_values`) and `feature_divergence|feature_consistency`
(vision-feature space) re-run the
evaluation on perturbed inputs. Comma-separated values and `--seeds 1,2,3` expand into one
sub-directory per configuration plus a `perturbation_summary.csv` with deltas against the
clean run of the same backend. See `python eval/run_all_benchmarks.py --help` for the
parameters.

## Adding a new benchmark

A benchmark is a registry entry plus up to four small pieces of code. The example adds a
yes/no benchmark `mybench` hosted at `org/MyBench` as a parquet file with `question`,
`answer`, `image` and `category` columns.

1. **Preparation** (`eval/prepare/`): add a source to a module (or create a module and list it
   in `eval/prepare/sources.py`) that writes the files to `<data_root>/mybench/`:

   ```python
   # in eval/prepare/lmms_lab.py: download one file and rename it
   BenchmarkSource(
       key="mybench",
       target="mybench",
       outputs=("mybench/test.parquet",),
       source="org/MyBench",
       approx_size="50 MB",
       prepare=prepare_renamed,
       options={"repos": ["org/MyBench"], "files": {"data/test-00000-of-00001.parquet": "test.parquet"}},
   )
   ```

   A custom `prepare(ctx, source)` function can use `hf_download`, `hf_snapshot`,
   `http_download_many`, `extract_zip` / `extract_tar` and `place_referenced_files` from
   `eval/prepare/common.py`, and returns a dict with at least `rows`.
2. **Loader** (`eval/easyr1_eval/loaders.py`): reuse one (`sharegpt` for ShareGPT /
   LLaMA-Factory rows, `easy_parquet` for EasyR1 training-format parquet with `problem`,
   `answer`, `images`, `pope` for yes/no parquet, ...) or write
   `load_mybench(spec, data_root, limit)` that returns `EvalSample(benchmark, sample_id,
   prompt=<bare question>, target, images=[path or {"bytes": ...}], metadata={...})`, and
   register it in `load_samples`. Do not append an output-format instruction:
   `--format-prompt` adds it.
3. **Scorer** (`eval/easyr1_eval/scorers.py`): reuse one (`boxed_exact_match`, `pope`,
   `hallusionbench`, `mcq`, `answer_bbox`, `grounding_iou`, `refcoco`, ...) or write
   `score_mybench(spec, rows, judge_config, output_dir) -> MetricResult` and add it to
   `SCORERS`. For `compare_tags.py`, add a per-sample extractor to `SCORER_EXTRACTORS` in
   `eval/easyr1_eval/compare.py`.
4. **Registry entry** (`eval/config/benchmarks.yaml`):

   ```yaml
   - key: mybench
     label: MyBench
     group: Perception          # summary group
     loader: pope               # reuse the yes/no parquet loader
     scorer: pope
     primary_metric: macro_f1
     source: org/MyBench
     path: mybench/test.parquet # relative to the data root
     max_new_tokens: 2048
     temperature: 0.0
     num_samples: 1
     # requires_judge: true     # skip unless a judge is configured
     # optional: true           # leave out of `all`
   ```
5. **Suite** (optional, `eval/config/suites.yaml`): add the key to a suite or create one.
6. **Test**: add a loader/scorer test on a tiny synthetic file (see
   `tests/test_eval_loaders.py`, `tests/test_eval_new_scorers.py`, `tests/test_eval_prepare.py`);
   `tests/test_eval_registry_summary.py` already checks that every registry entry has a
   preparation recipe, a known scorer and a relative path. Then run
   `bash scripts/prepare_eval_data.sh mybench` and
   `bash scripts/eval.sh <model> --benchmarks mybench --limit 16`.
