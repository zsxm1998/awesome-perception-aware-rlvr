# Evaluation

One-command evaluation of the base models and checkpoints produced by this repository.
The harness covers the benchmarks used by the reproduced papers (PAPO, VPPO, DVRP, ToR,
PGPO, PEPO, CFPO, VEPO, NoisyRollout, VGPO, GRIT, DeepEyes, VA-OPD, VGS, VCSD, Vision-OPD) and CGPO, downloads
them from public sources, runs inference with vLLM on all visible GPUs, and scores every
benchmark with a documented, rule-based protocol (an LLM judge is only used for MM-Vet and,
optionally, GQA).

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
bash scripts/prepare_eval_data.sh all                    # every non-optional benchmark (~16 GB on disk)
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
also set prompt defaults for models without a training record; a checkpoint trained in this
repository is prompted as it was trained, and explicit flags always win (see [Prompts](#prompts)).

| Suite | Benchmarks | Paper table / notes |
|---|---|---|
| `papo` | geo3k, mathvista, wemath, mmk12, mathverse, logicvista, clevr_count, mmmu_pro, mathverse_v | PAPO Table 1 (same data and scoring as PAPO-Eval) |
| `vppo` | mathverse, dynamath, mmk12, geo3k, mathvision, wemath, logicvista, mmmu_pro | VPPO Table 1; the paper uses 4096 new tokens for 32B models (`--max-new-tokens 4096`) |
| `dvrp` | geo3k, mathvista, wemath, mathverse, mathverse_v, mmk12 | DVRP Table 1, general columns; medical columns not included; DVRP reports top_p 0.9/0.99, we use 1.0 |
| `pgpo` | geo3k, mmk12, mathverse, dynamath, mathvision, logicvista, mmmu_pro, mathverse_v | PGPO Table 1; PGPO evaluates DynaMath "sample variant1", we use the VPPO-Eval DynaMath file |
| `pepo` | geo3k, mathvista, wemath, mathverse, logicvista, clevr_count, mmmu_pro | PEPO Table 1 + Table 5 (ViRL39K scaling); Geometry3K val is not included |
| `pepo_geometry` | geo3k, mathvista, mathverse, logicvista | PEPO Table 1 with the prompt of the Geometry3K-trained models (`pepo.jinja`, `<answer>` tags), their image size, and PEPO's answer protocol (`--answer-protocol pepo`: MathVista, LogicVista and MathVerse are asked as PEPO's evaluation scripts ask them, the question, a newline and the `<think>`/`<answer>` instruction, which MathVerse follows with the request for the option letter; LogicVista reads the last standalone letter of the answer) |
| `vepo` | geo3k, mmk12, hallusionbench, mathvista, wemath, mathverse, mathvision | VEPO Table 1; the paper uses greedy decoding and a gpt-4o-mini judge for 4 benchmarks, we use rule-based avg@8 |
| `noisyrollout` | geo3k, mathverse, mathvision, mathvista, wemath, hallusionbench | NoisyRollout Table 1 (and Geometry3K test, in-domain); the paper uses greedy decoding and Gemini-2.0-Flash answer parsing for 4 benchmarks, we use rule-based avg@8 (HallusionBench greedy); the paper's MathVision is the full 3,040-item test set |
| `vgpo` | mathvista, mathverse, wemath, mmk12, geo3k, logicvista, clevr_count, mmmu_pro, mathverse_v | VGPO Table 1 without GeoMath (no public data); the paper decodes greedily, the default here is avg@8 |
| `vapo` | mathverse_v, mathvista, mathvision, logicvista, wemath, geo3k, mmmu_val, mmstar, hallusionbench, mm_vet | VAPO Tables 1-2; the paper decodes greedily with VLMEvalKit and GPT answer extraction, our math sets are rule-based avg@8 (the other four greedy); the paper's MathVerse is testmini Vision Only (here `mathverse_v`, which includes it), its MathVision the full 3,040-item test set, its We-Math the strict score; MM-Vet needs a judge |
| `tor` | mathverse, mathvision, mathvista, wemath, hallusionbench | ToR Table 4; the paper follows NoisyRollout (greedy + Gemini answer parsing), we use rule-based avg@8 |
| `cfpo` | cvqa_real, mars_bench, pope, textvqa, mmmu_pro, geo3k, wemath, mmk12, mathverse, logicvista | CFPO Table 1; CFPO reports POPE as pooled accuracy with avg@8 (same 9,000 questions; our primary is greedy macro F1, pooled accuracy in the details); CFPO's LogicVista file has 448 items, PAPO-Eval's 447 |
| `grit` | grit_vsr, grit_tallyqa, grit_gqa, ovdeval_position | GRIT Table 1 (GRIT judges answers with GPT-4o, we use relaxed exact match); defaults follow GRIT's evaluation, for models trained with GRIT's prompt: its prompt (`grit.jinja`, no system prompt), the bare question (`--grounding-instruction none`; elsewhere a box instruction is appended to these sets) and 3,136–200,704 pixels. Evaluate other models on these sets with `--benchmarks grit_vsr,grit_tallyqa,grit_gqa,ovdeval_position` and their own prompt flags |
| `deepeyes` | vstar, hrbench_4k, hrbench_8k, mme_realworld_lite, pope | DeepEyes Tables 1-3; default: agentic DeepEyes inference (see [Agentic evaluation](#agentic-evaluation-deepeyes)) |
| `va_opd` | wemath, mathvista, mathverse, hallusionbench, ai2d, mmmu_val, mmstar | VA-OPD Table 1 without OCRBench (not available); the paper reports avg@8 at T=1.0 with official scoring and a GPT-4o judge where applicable, for the best checkpoint; our math sets are avg@8, the other four greedy by default (`--temperature 1.0 --num-samples 8` for avg@8; HallusionBench scores the first sample only); our MathVerse holds the 2,180 multiple-choice testmini questions with an image; defaults: the training prompt (`math.jinja`) and image size |
| `vgs` | mathvision, mathverse_v, logicvista, mmmu_pro | VGS Table 1, the 4 of 7 benchmarks available (no VisualPuzzles, VlmsAreBlind); the paper reports Acc@1 (greedy) and Acc@16 (T=1); our MathVerse-V and MMMU-Pro come from PAPO-Eval and differ from the subsets the VGS authors use (MathVerse VD / VO multiple choice, 4-option MMMU-Pro); defaults: the training prompt (VGS' system prompt), image size and answer reading (`--answer-protocol vgs`) |
| `vcsd` | blink, mmstar, vstar, mathvista, hrbench_4k, hrbench_8k, hallusionbench | VCSD Table 1; the paper's HallusionBench score is (aAcc+fAcc+qAcc)/3 (`aqf_mean` in the details) and its Acc. the mean of the seven scores; decoding and scoring are not given in the paper (we use greedy rule-based scoring, MathVista avg@8); defaults: the training prompt (the bare problem), chat template, tokenizer and image size |
| `vision_opd` | vstar, zoombench, hrbench_4k, hrbench_8k, mme_realworld_lite, mmstar, pope | Vision-OPD Table 1 + holdout set of Table 2; the paper uses the full MME-RealWorld EN/CN (we have the lite version) and a gpt-oss-120b judge; MMVP is not available and CV-Bench is not in the suite (`--benchmarks cvbench` adds it); defaults: the training prompt (the dataset prompt, no system prompt), the non-thinking chat template and the image size (65,536-16,777,216 pixels) |
| `cgpo` | = `papo` | CGPO natural-image reproduction (trained on ViRL39K); default prompt `xml_grounded_reasoning.jinja` |
| `comparison` | math reasoning: geo3k, mathvista, wemath, mmk12, mathverse, mathvision, dynamath; vision-dependent reasoning: mathverse_v, mmmu_pro, logicvista, clevr_count, ai2d, mme_cognition; perception and hallucination: pope, hallusionbench, mmstar, blink, mme_perception, cvbench | the benchmark set used by `examples/comparison/`, in three groups (see below) |
| `opd` | = `comparison` (same groups) | the benchmark set used by `examples/comparison/opd_qwen3_vl_2b` |
| `perception` | pope, hallusionbench, mme, gqa, mm_vet | general perception / hallucination |
| `refcoco` | refcoco_val, refcoco_plus_val, refcocog_val | optional (COCO images) |
| `all` | every non-optional benchmark | |

A suite can split its benchmarks into groups (`groups:` in `eval/config/suites.yaml`, each benchmark
in exactly one group). The summaries then add, for every grouped suite with a scored benchmark in the
run, each group's mean, the mean of the group means and the mean over all of the suite's benchmarks
(columns `comparison: Math reasoning`, ..., `comparison: Group mean`, `comparison: Benchmark mean`).
Benchmarks weigh equally inside a group and groups weigh equally in the group mean, so the 7 math
sets of `comparison` count as much as its 6 perception sets. A mean is written only once every
benchmark it covers is scored. These columns do not depend on `--suite`: a run evaluated in several
invocations gets them as soon as its results are complete.

The reasoning columns of all papers are scored with one protocol (PAPO-Eval, below), which
is what PAPO, VPPO, DVRP, PGPO, PEPO (Table 5) and CFPO use. Where a paper used another
protocol (VEPO, NoisyRollout and ToR: greedy decoding and an LLM answer parser; VGPO: greedy decoding; PEPO Table 1: the official
MathVista checker; GRIT: a GPT-4o judge; DeepEyes: a Qwen2.5-72B judge fallback; VA-OPD: a
GPT-4o judge where applicable; Vision-OPD: a gpt-oss-120b judge; VGS and VCSD do not describe
their scoring), numbers are comparable across methods within this harness but not
digit-for-digit with the paper.
`--temperature 0 --num-samples 1` switches any run to greedy decoding.

Sampling uses no top-k limit and images of 200,704-1,003,520 pixels by default, the settings
the training scripts use, so that a model is evaluated as it was trained. The PAPO-Eval code
itself (LLaMA-Factory) samples with top-k 50 and images of 1,024-589,824 pixels; to evaluate
with those settings, add

```bash
--top-k 50 --min-pixels 1024 --max-pixels 589824
```

## Benchmarks

Sample counts are the rows actually evaluated. "avg@8" = mean accuracy over 8 samples at
temperature 1.0, top_p 1.0 and 2048 new tokens; "greedy" = 1 sample at temperature 0.

| Key | Benchmark | Source (HF) | #Samples | Primary metric | Decoding | Used by |
|---|---|---|---:|---|---|---|
| `geo3k` | Geometry3K test | `PAPO-Galaxy/PAPO_eval` | 601 | boxed exact match, mean acc@k | avg@8 | PAPO, VPPO, DVRP, PGPO, PEPO, CFPO, VEPO, NoisyRollout, CGPO, VGPO |
| `mathvista` | MathVista testmini | `PAPO-Galaxy/PAPO_eval` | 1000 | boxed exact match | avg@8 | PAPO, DVRP, PEPO, VEPO, NoisyRollout, ToR, CGPO, VA-OPD, VCSD, VGPO |
| `wemath` | We-Math | `PAPO-Galaxy/PAPO_eval` | 1740 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PEPO, CFPO, VEPO, NoisyRollout, ToR, CGPO, VA-OPD, VGPO |
| `mmk12` | MMK12 test | `PAPO-Galaxy/PAPO_eval` | 2000 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PGPO, CFPO, VEPO, CGPO, VGPO |
| `mathverse` | MathVerse testmini (multi-choice) | `PAPO-Galaxy/PAPO_eval` | 2180 | boxed exact match | avg@8 | PAPO, VPPO, DVRP, PGPO, PEPO, CFPO, VEPO, NoisyRollout, ToR, CGPO, VA-OPD, VGPO |
| `mathverse_v` | MathVerse vision-dependent | `PAPO-Galaxy/PAPO_eval` | 1308 | boxed exact match | avg@8 | PAPO, DVRP, PGPO, CGPO, VGS, VGPO |
| `logicvista` | LogicVista | `PAPO-Galaxy/PAPO_eval` | 447 | boxed exact match | avg@8 | PAPO, VPPO, PGPO, PEPO, CFPO, CGPO, VGS, VGPO |
| `clevr_count` | Counting (PAPO's "SuperClevr counting" column) | `PAPO-Galaxy/PAPO_eval` | 200 | boxed exact match | avg@8 | PAPO, PEPO, CGPO, VGPO |
| `mmmu_pro` | MMMU-Pro (vision setting) | `PAPO-Galaxy/PAPO_eval` | 1730 | boxed exact match | avg@8 | PAPO, VPPO, PGPO, PEPO, CFPO, CGPO, VGS, VGPO |
| `dynamath` | DynaMath (verifiable subset) | `chamber111/VPPO-Eval` | 3666 | boxed exact match | avg@8 | VPPO, PGPO |
| `mathvision` | MathVision (verifiable subset) | `chamber111/VPPO-Eval` | 2907 | boxed exact match | avg@8 | VPPO, PGPO, VEPO, NoisyRollout, ToR, VGS |
| `pope` | POPE random/popular/adversarial | `lmms-lab/POPE` | 9000 | macro F1 over the 3 splits (accuracy in details) | greedy | CFPO, DeepEyes, Vision-OPD |
| `hallusionbench` | HallusionBench (image questions) | `lmms-lab/HallusionBench` | 951 | question accuracy aAcc (fAcc, qAcc and their mean `aqf_mean` in details) | greedy | VEPO, NoisyRollout, ToR, VA-OPD, VCSD |
| `mme` | MME | `lmms-lab/MME` | 2374 | total score (acc + acc+); /2800 in averages | greedy (PGPO averages 8 samples) | PGPO |
| `mme_perception` | MME perception (the 10 perception subtasks; reads the `mme` data) | `lmms-lab/MME` | 2114 | perception score (acc + acc+); /2000 in averages | greedy | comparison suite |
| `mme_cognition` | MME cognition (the 4 cognition subtasks; reads the `mme` data) | `lmms-lab/MME` | 260 | cognition score (acc + acc+); /800 in averages | greedy | comparison suite |
| `gqa` | GQA testdev-balanced | `lmms-lab/GQA` | 12578 | exact match (EM→judge cascade with a judge) | greedy | general |
| `mm_vet` | MM-Vet | `lmms-lab/MMVet` | 218 | LLM-judge score; **skipped without a judge** | greedy | general |
| `cvqa_real` | C-VQA-Real (counterfactual VQA on COCO) | `RavenInJuly/CFPO_Datasets` + COCO val2014 | 6288 | CFPO match, mean acc@k | avg@8 | CFPO |
| `mars_bench` | MARS-Bench | `RavenInJuly/CFPO_Datasets` + COCO val2014 | 5110 | CFPO match, mean acc@k | avg@8 | CFPO |
| `textvqa` | TextVQA val | `RavenInJuly/CFPO_Datasets` | 5000 | CFPO match, mean acc@k (VQA accuracy in details) | avg@8 | CFPO |
| `seed_bench` | SEED-Bench (optional, ~27 GB) | `lmms-lab/SEED-Bench` | 17990 | option accuracy (image / video split in details) | greedy | general |
| `mmstar` | MMStar | `Lin-Chen/MMStar` | 1500 | accuracy over all questions (per category in details) | greedy | VA-OPD, VCSD, Vision-OPD |
| `blink` | BLINK val (14 subtasks, 1-4 images) | `BLINK-Benchmark/BLINK` | 1901 | accuracy over all questions (per subtask in details) | greedy | VCSD |
| `cvbench` | CV-Bench (2D: ADE20K and COCO count and relation; 3D: Omni3D depth and distance) | `nyu-visionx/CV-Bench` | 2638 | the dataset card's overall score: mean of the 2D accuracy (mean of ADE20K and COCO) and the 3D accuracy | greedy | Vision-OPD (holdout) |
| `ai2d` | AI2D test | `lmms-lab/ai2d` | 3088 | accuracy | greedy | VA-OPD |
| `mmmu_val` | MMMU validation (847 multiple-choice, 53 open) | `lmms-lab/MMMU` | 900 | accuracy over all questions (by question type, discipline and subject in details) | greedy | VA-OPD |
| `vstar` | V* Bench | `craigwu/vstar_bench` | 191 | accuracy over all questions | greedy | DeepEyes, VCSD, Vision-OPD |
| `hrbench_4k` | HR-Bench 4K | `DreamMr/HR-Bench` | 800 | mean of FSP and FCP accuracy | greedy | DeepEyes, VCSD, Vision-OPD |
| `hrbench_8k` | HR-Bench 8K | `DreamMr/HR-Bench` | 800 | mean of FSP and FCP accuracy | greedy | DeepEyes, VCSD, Vision-OPD |
| `mme_realworld_lite` | MME-RealWorld-Lite | `yifanzhang114/MME-RealWorld-lite-lmms-eval` | 1919 | accuracy over all questions (Perception / Reasoning and per-subtask in details) | greedy | DeepEyes, Vision-OPD (full EN/CN in the paper) |
| `zoombench` | ZoomBench (621 multiple-choice, 224 counting; full image) | `inclusionAI/ZoomBench` | 845 | accuracy over all questions (multiple-choice and counting accuracy in details) | greedy | Vision-OPD |
| `grit_vsr` | GRIT VSR | `yfan1997/GRIT_data` + COCO 2017 | 288 | answer accuracy (GRIT grounding IoU in details) | greedy | GRIT |
| `grit_tallyqa` | GRIT TallyQA (official file) | `yfan1997/GRIT_data` + Visual Genome | 491 | answer accuracy (+ GRIT grounding IoU) | greedy | GRIT |
| `grit_gqa` | GRIT GQA | `yfan1997/GRIT_data` + Visual Genome | 509 | answer accuracy (+ GRIT grounding IoU) | greedy | GRIT |
| `tallyqa_relabeled` | GRIT TallyQA, relabeled: one box per counted instance, corrected answers | `zsxm1998/GRIT-TallyQA-Relabeled` (with images) | 491 | answer accuracy (+ GRIT grounding IoU) | greedy | general |
| `ovdeval_position` | OVDEval position | `yfan1997/GRIT_data` + `omlab/OVDEval` | 2146 | GRIT grounding IoU (Acc@0.5 in details) | greedy | GRIT |
| `refcoco_val` | RefCOCO val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 10834 | Acc@0.5 IoU | greedy | DeepEyes, PEPO |
| `refcoco_plus_val` | RefCOCO+ val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 10758 | Acc@0.5 IoU | greedy | DeepEyes |
| `refcocog_val` | RefCOCOg val (optional) | `PaDT-MLLM/RefCOCO` + COCO train2014 | 4896 | Acc@0.5 IoU | greedy | DeepEyes |

Not included: the medical benchmarks of DVRP, LISA grounding and the few-shot / puzzle tasks
of PEPO, GRIT's GPT-judged MathVista/MME subsets, OCRBench (VA-OPD), VisualPuzzles and
VlmsAreBlind (VGS), MMVP and the full MME-RealWorld EN/CN sets (Vision-OPD).

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
  `<answer>`, "answer is ..." or the first yes/no word). MME scores each subtask out of 200
  (accuracy over questions plus accuracy+ over images with both questions right); `mme` sums the
  14 subtasks (out of 2,800) and `mme_perception` the 10 perception subtasks (existence, count,
  position, color, posters, celebrity, scene, landmark, artwork, OCR; out of 2,000), MME's
  perception score; `mme_cognition` the 4 cognition subtasks (commonsense reasoning, numerical
  calculation, text translation, code reasoning; out of 800), its cognition score. HallusionBench
  reports question
  accuracy (aAcc, primary), figure accuracy (fAcc: every question about a figure correct) and
  question-pair accuracy (qAcc) as defined by the official evaluation (questions grouped by
  category, subcategory and set; VS questions without a figure are left out of fAcc), and
  `aqf_mean` = (aAcc + fAcc + qAcc) / 3, the HallusionBench score that VCSD reports. The details
  also hold the three accuracies as `aAcc`, `fAcc` and `qAcc`. Only the first sample of each
  question is scored.
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
- **MMStar / BLINK / AI2D (`mcq`)** ask the released question with its options (MMStar's
  question text, which contains the options; BLINK's `prompt` field, the question followed by
  `(A)` options, with the 1-4 images of a question in dataset order; AI2D's question with
  `(A)`-`(D)` option lines), followed by "Answer with the option's letter from the given
  choices." (lmms-eval's instruction without "directly", as for V\*, so that it does not
  contradict a reasoning format prompt), and use the letter extraction of V\*. The primary
  metric is the accuracy over all questions; `accuracy_by_category` holds MMStar's six
  categories (250 questions each, so their mean equals the overall accuracy) and BLINK's 14
  subtasks.
- **CV-Bench (`mcq`)** asks the dataset's `prompt` (the question and `(A)` option lines) with the
  same letter instruction and letter extraction. The primary metric is the overall score of the
  dataset card: the 2D accuracy is the mean of the ADE20K and COCO accuracies, the 3D accuracy is
  the Omni3D accuracy, and the overall score is their mean (`accuracy_2d`, `accuracy_3d`,
  `accuracy_by_category` per source and `accuracy_by_task` in the details).
- **MMMU validation (`mmmu`)** asks the multiple-choice questions with `(A)` option lines and
  the letter instruction, and the open questions with "Answer the question using a single word
  or phrase." (lmms-eval's instruction). As in lmms-eval, the images are those that the question
  and the options reference with `<image n>`, in that order, placed before the text, which keeps
  the placeholders (four questions have an extra image that no placeholder references; it is
  not passed). Multiple-choice answers use the letter extraction of V\*; an answer without a
  recognizable letter is wrong (the official `parse_multi_choice_response` picks a random letter
  instead). Open answers are scored with the rule-based matching of the official
  `mmmu/utils/eval_utils.py` (`parse_open_response` and `eval_open`: the shortest tails after
  "is", "so", "therefore", ... of each line and the numbers in them, numbers compared after
  rounding to 2 decimals, text by containment) applied to the final answer (`\boxed{}`,
  `<answer>` or "answer is", else the whole response); the three questions with several
  accepted answers keep them as a list, as the official answer file does. The primary metric is
  the accuracy over all 900 questions; the details hold the accuracy by question type, by
  discipline (6) and by subject (30).
- **ZoomBench (`zoombench`)** asks the dataset's `query` (the question, the `A.`-`D.` options and,
  for most multiple-choice questions, "Answer with the option's letter from the given choices.";
  counting questions end with "Please answer using Arabic numerals.") on the full image, as
  Vision-OPD does. The dataset also provides the crop of the key region (the region view of the
  ZoomBench paper); the prepare step keeps it in `data/eval/zoombench/crops/`, but it is not
  evaluated. The official evaluation (Vision-OPD's `eval/judge_qwenlm.py`; the ZoomBench
  repository has the same two stages with another judge model) extracts the `<answer>` span or
  the text from `Answer:` on, accepts the answer when `mathruler`'s
  `grade_answer(reference, answer)` matches, and otherwise asks an LLM judge (gpt-oss-120b in
  Vision-OPD) whether the response means the same as the reference. We keep the first stage and
  replace the judge with rules, applied to the text after the last `</think>` with markdown
  emphasis removed:
  - multiple choice: the letter of the last `<answer>` span, else of the last `\boxed{}`, else of
    the line after "Answer:", "answer is" or "Final Answer:" (then after "option" / "choice"),
    else of a response that starts with a letter (`B`, `(B)`, `B.`, `B)`, `B: text`), else the
    option whose text the answer equals or the only option text it mentions; this maps "Yes" /
    "No" in the 15 two-option questions, whose option order varies. A letter must stand alone
    and be an option of the question; it is never the first capital letter of the response
    (which would read "Answer: D" as A), and an answer that names two different letters
    ("A or C") is wrong.
  - counting: the first number of the `<answer>` span, the `\boxed{}` or the answer line, else a
    first line that is only a number, else the only number of the last line, else the last
    number of the response; digits ("4.", "4.0", "1,200"; a minus sign is kept, "-4", "- 4", but
    not the dash of a range, "3-4", "3 - 4", nor a list item "- 4" opening a line) and English
    number words are read and compared with the integer reference.

  The primary metric is the accuracy over all 845 questions; the details hold `mcq_accuracy`,
  `counting_accuracy` and `answer_source_counts` (how many answers `mathruler` accepted, the
  rules decided, named two letters, could not be read or were truncated). The rules cannot
  credit paraphrased option texts or other answers that an LLM judge accepts by meaning, and
  the judge itself is not deterministic (about one point between two judging runs of the same
  outputs, reported in a Vision-OPD issue), so ZoomBench numbers are comparable within this
  harness rather than digit-for-digit with the papers. One multiple-choice item has the gold
  letter D for the options A and B.
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
  accuracy: the final answer is correct or wrong by relaxed exact match (articles, number words
  and punctuation normalized), while GRIT has GPT-4o score the answer between 0 and 1, so the two
  are not comparable. `answer/grit_rule_accuracy` (details) checks the `<answer>` text with the
  rule of GRIT's training reward in this repository (`examples/reward_function/grit.py`); it is
  not GRIT's metric either. The
  GRIT grounding IoU (`grounding/grit_iou`, also the `<key>_giou` column of the global summary)
  follows `extract_eval_results.py` of UCSB-AI/GRIT: per sample, the IoU between the union of
  *all* boxes written in the response and the union of the GT boxes; samples with an empty GT
  list are skipped and samples without a predicted box score 0. It reads boxes written in
  square brackets; GRIT's script takes any four comma-separated integers (also in parentheses or
  JSON), which `grounding/grit_iou_grit_pattern` uses. Both map the boxes onto the original image
  and clip them to it (GRIT's script compares Qwen pixel boxes with the original image's boxes
  without rescaling or clipping).
  Box precision/recall/F1 at IoU 0.5 (one-to-one matching) and box counts are also reported.
  For agentic runs the predicted boxes are the zoom-in regions the agent committed.
- **OVDEval** reports only the GRIT grounding IoU (as GRIT does), with Acc@0.5 IoU and
  `grounding/grit_iou_grit_pattern` in the details. **RefCOCO** reports Acc@0.5 IoU of the best box of the final answer
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
  Built-in default for one-shot runs: `examples/format_prompt/math_perception.jinja`, i.e. the PAPO
  instruction ("... enclosed within <think> </think> tags. Then, provide your final answer
  enclosed within \boxed{}."), which makes the reasoning prompts identical to PAPO-Eval's.
  `--format-prompt none` sends the bare question.
- `--system-prompt FILE` (same as `data.system_prompt`; `none` for no system prompt) and
  `--prompt-mode chat` (default: the model's chat template) or `raw`.
- `--plain-think-tokens auto|true|false` (built-in default `auto`) tokenizes `<think>` / `</think>` as
  plain text for models in which they are untrained added tokens (Qwen3-VL Instruct), as training
  does (`worker.actor.model.plain_think_tokens`).
- `--chat-template FILE` replaces the processor's chat template (same as
  `data.override_chat_template`; `none` keeps the model's); recorded in `summary.csv`.

Each of these options, the image size (`--min-pixels`, `--max-pixels`), `--interaction-mode` and
`--agent-prompt-style` is taken from the first source that sets it:

1. the command line;
2. the training record of a checkpoint trained in this repository (`<run>/global_step_N/actor`,
   merged or not): the run's `experiment_config.json`, which the trainer writes, gives the format and
   system prompt, the image size, `plain_think_tokens`, the interaction mode and the agent prompt
   style, and the checkpoint carries the chat template and tokenizer it was trained with (saved with
   its processor); a run trained with the VGS reward also gets `--answer-protocol vgs`, the answer
   reading of its reward. A flag that differs from the training setting is used, with a warning. A
   prompt file that is no longer at its recorded path is looked up under this repository's
   `examples/`; if it is not there either, the run stops and asks for the flag (a flag for that
   option is enough);
3. the defaults of the selected suites, for models without a training record (the released models
   of a paper's tables, or a model directory copied without its run);
4. the built-in defaults.

The runner prints where each setting came from (`[info] prompt settings from ...`), and
`summary.csv` names the training record it used. Checkpoints trained with the scripts of this
repository therefore need no prompt flags, whatever the suite. Other models get the prompt of each
method's training from these flags or suite defaults:

| Model | Flags (or suite default) |
|---|---|
| PAPO, VPPO, DVRP, PGPO, ToR, GRPO/DAPO baselines, the OPD comparison | default (`math_perception.jinja`) |
| CFPO | `--format-prompt none --system-prompt examples/system_prompt/cfpo.txt` (suite `cfpo`) |
| VEPO | `--format-prompt none --system-prompt examples/system_prompt/vepo.txt` (suite `vepo`) |
| NoisyRollout | `--format-prompt none --system-prompt examples/system_prompt/noisyrollout.txt --min-pixels 262144 --max-pixels 1000000` (suite `noisyrollout`) |
| VGPO | default prompt, `--min-pixels 262144 --max-pixels 4194304` (suite `vgpo`) |
| VAPO | default prompt, `--min-pixels 262144 --max-pixels 4194304` (suite `vapo`) |
| PEPO (Geometry3K setting) | `--format-prompt examples/format_prompt/pepo.jinja` (suite `pepo_geometry`) |
| CGPO | `--format-prompt examples/format_prompt/xml_grounded_reasoning.jinja` (suite `cgpo`) |
| GRIT (`examples/reproduction/grit`) | `--format-prompt examples/format_prompt/grit.jinja --system-prompt none --min-pixels 3136 --max-pixels 200704` (suite `grit`) |
| GRIT in the comparison (`examples/comparison/qwen3_vl_4b/grit.sh`) | `--format-prompt none --system-prompt examples/system_prompt/grit_GR.txt`; on the GRIT sets with `--benchmarks grit_vsr,grit_tallyqa,grit_gqa,ovdeval_position`, not `--suite grit` |
| DeepEyes, Qwen2.5-VL (`qwen2_5_vl_7b_grpo_deepeyes.sh`, DeepEyes' own prompts) | `--interaction-mode agentic --agent-prompt-style official` (suite `deepeyes` plus `--agent-prompt-style official`) |
| VA-OPD (`examples/reproduction/va_opd`) | `--format-prompt examples/format_prompt/math.jinja --min-pixels 262144 --max-pixels 4194304` (suite `va_opd`) |
| VGS (`examples/reproduction/vgs`) | `--format-prompt none --system-prompt examples/system_prompt/vgs.txt --min-pixels 262144 --max-pixels 4194304` (suite `vgs`) |
| VCSD (`examples/reproduction/vcsd`) | `--format-prompt none --chat-template examples/chat_template/qwen_no_thinking.jinja --plain-think-tokens false --min-pixels 262144 --max-pixels 4194304` (suite `vcsd`) |
| Vision-OPD (`examples/reproduction/vision_opd`) | `--format-prompt none --system-prompt none --chat-template examples/chat_template/qwen_no_thinking.jinja --min-pixels 65536 --max-pixels 16777216` (suite `vision_opd`) |
| DeepEyes, rewritten prompt (Qwen3-VL and `*_native.sh`) | `--interaction-mode agentic --system-prompt examples/system_prompt/deepeyes.txt` (suite `deepeyes`; `deepeyes_pixel.txt` is picked for Qwen2-VL / Qwen2.5-VL) |

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
| `--temperature`, `--num-samples`, `--top-p`, `--top-k`, `--max-new-tokens` | override the registry decoding settings for every benchmark (`--top-k` is unset by default: no limit) |
| `--gpus`, `--tp`, `--gpu-memory-utilization`, `--max-model-len` | engine placement and size |
| `--min-pixels`, `--max-pixels` | image resizing (default: the suite's, else 200704 to 1003520 pixels, as in training; `pepo_geometry` uses 3136 to 12845056, PEPO's training range) |
| `--max-dynamic-patch N` | InternVL only: at most N tiles per image (default: the model config's), as `worker.actor.model.max_dynamic_patch` in training |
| `--box-format auto\|norm1000\|pixel` | how predicted boxes are read (see [Scoring protocols](#scoring-protocols)) |
| `--grounding-instruction append\|none` | whether the box instruction is appended to the questions of the grounding sets (default `append`; the `grit` suite asks the bare question, as GRIT) |
| `--answer-protocol default\|pepo\|vgs` | `pepo`: MathVista, LogicVista and MathVerse are asked and LogicVista is read as in PEPO's evaluation scripts (suite `pepo_geometry`); `vgs`: the `\boxed{}`-scored benchmarks read the answer as the VGS training reward does, with the same code (`verl/utils/vgs_answer.py`: the last `\boxed{}`, else the text after `</reason>`; option letters for multiple-choice answers, mathruler otherwise). It only changes the scoring: a checkpoint gets it from its training record when trained with that reward (the `vgs` suite's default applies to models without a record), and `--score-only --answer-protocol ...` rescores existing predictions with the given reading; without a protocol from a flag, the training record or a suite, predictions are read as recorded when they were generated |
| `--agent-prompt-style native\|official`, `--agent-observation-min-pixels N` | agentic runs: DeepEyes' own prompts instead of the rewritten one; the lower pixel bound of tool crops (default `--min-pixels`; the image processor's own lower bound still applies) |
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

`--agent-prompt-style official` evaluates a model trained with DeepEyes' own prompts
(`worker.rollout.agent_prompt_style=official`): DeepEyes' system prompt (tool schema written
out) replaces `--system-prompt`, its format instruction ("Think first, call
**image_zoom_in_tool** if needed, then answer. ...") follows the question, and crops come back
as `<tool_response><image>` + the same instruction + `</tool_response>`, as in its training data
and evaluation scripts. The setting enters the result fingerprint only when used.

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
│                                                #   + the means of grouped suites (row_type suite_*)
├── predictions/<benchmark>/predictions.jsonl    # prompt, messages, image paths, responses, metadata
├── metrics/<benchmark>.json                     # primary score, details, generation diagnostics
├── metrics/<benchmark>_per_sample*.jsonl        # per-sample scores / extracted answers
└── state/                                       # resume fingerprints
eval/results/summary.csv                         # one row per run, one column per benchmark,
                                                 #   group averages, grouped-suite means, Overall Avg
```

A resumed run reuses predictions and scores whose fingerprint is unchanged. The score fingerprint holds
the scorer version (`SCORER_VERSION`, all benchmarks) and, for a scorer that changed alone, its
revision (`SCORER_REVISIONS`, its benchmarks only), so a change of the scoring rescores the
predictions without generating them again.

`normalized_score_0_100` is the primary metric on a 0-100 scale (MME total / 2800 * 100, MME
perception / 2000 * 100, MME cognition / 800 * 100);
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
