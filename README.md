<div align="center">

# Awesome Perception-Aware RLVR

**Unified reproductions, controlled comparisons and one-click evaluation for perception-aware reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) of large vision-language models, plus a curated paper list.**

RLVR: PAPO · VPPO · DVRP · ToR · PGPO · PEPO · CFPO · VEPO · NoisyRollout · VGPO · VAPO · GRIT · DeepEyes · CGPO<br>
OPD: VA-OPD · VGS · VCSD · Vision-OPD<br>
in one EasyR1 codebase with 30+ benchmarks

🌟 **Official repository of [CGPO](#-cgpo-acm-mm-2026-oral) (ACM MM 2026 Oral)** 🌟

[English](README.md) | [简体中文](README_zh.md)

[![Awesome](https://awesome.re/badge.svg)](https://github.com/sindresorhus/awesome)
[![Papers](https://img.shields.io/badge/Papers-125-blue)](#-paper-list)
[![Reproduced](https://img.shields.io/badge/Reproduced-18-brightgreen)](#-reproduced-methods)
[![CGPO](https://img.shields.io/badge/CGPO-ACM%20MM%202026%20Oral-8A2BE2)](https://doi.org/10.1145/3767308.3835969)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Built on EasyR1](https://img.shields.io/badge/built%20on-EasyR1-orange)](https://github.com/hiyouga/EasyR1)

</div>

RLVR makes large vision-language models better reasoners, but its reward only checks the final answer: a
model can be right for the wrong reasons and rely on language priors instead of the image.
**Perception-aware RLVR** methods put visual perception back into the objective, e.g. by contrasting
the policy on counterfactual views of the image, by giving credit to the tokens that actually depend
on the image, or by letting the model ground or zoom into visual evidence while it reasons. The same
ideas reach **OPD**, where a teacher scores the student's own rollouts token by token: weighting the
tokens the image decides, distilling the teacher's visual gain rather than its language prior, or
letting the model teach itself with a privileged view of the image.

<p align="center">
  <img src="docs/assets/architecture.svg" width="100%" alt="Framework architecture: one RL step in three stages. Rollout in vLLM: noised rollout images (NoisyRollout), a zoom-in tool (DeepEyes), online filtering (DAPO), and the reward inputs the policy must generate before vLLM sleeps: grounding consistency (CGPO) and visual-claim probes (VAPO). Before the update, without gradient: rollout-policy and reference log-probs, and the frozen teacher's for sampled-token OPD and VA-OPD; an auxiliary forward on a changed input (masked, noised, pixelated or removed image, or a change inside the model) that gives each token's visual dependence; the reward function with each method's extra term; the advantages. Actor update: the five steps of compute_perception_reasoning_policy_loss (control signals, advantage modulation, token-wise PPO terms, token selection, auxiliary losses), or a distillation loss for OPD with a student and a no-gradient teacher forward on each micro-batch, then the optimizer step.">
</p>

This repository provides

- the **official implementation of [CGPO](#-cgpo-acm-mm-2026-oral)** (ACM MM 2026 Oral);
- **[Reproductions](#-reproduced-methods)** of 17 methods (13 RLVR and 4 OPD) with each paper's own data,
  model and hyper-parameters, plus the baselines they compare against;
- **[Controlled comparisons](#-controlled-comparison)** under one shared setting each: the RLVR methods
  except DeepEyes on Qwen3-VL-4B, and the OPD methods on a Qwen3-VL-2B student with a GRPO-trained
  Qwen3-VL-4B as the teacher;
- **[One-click evaluation](eval/README.md)** on the union of the benchmarks used by these papers, with
  per-benchmark download scripts and per-paper suites;
- a **[paper list](#-paper-list)** of perception-aware RLVR (grounded reasoning and thinking with images
  included) and OPD, grouped by training stage, and the benchmarks of this line of work.

All methods share one training framework (a fork of [EasyR1](https://github.com/hiyouga/EasyR1)),
so they can be combined, compared and extended with a few configuration switches.

> [!NOTE]
> Except for CGPO, the methods are **unofficial re-implementations**. Please refer to and cite the
> original papers and repositories. Differences from the official recipes are documented in each
> method's README. Our numbers come from single runs of these re-implementations and may not
> reproduce the gains reported in the papers; see
> [About the results](examples/comparison/README.md#about-the-results).

## 📖 Contents

- [News](#-news)
- [Roadmap](#-roadmap)
- [CGPO (ACM MM 2026 Oral)](#-cgpo-acm-mm-2026-oral)
- [Reproduced methods](#-reproduced-methods)
- [Quick start](#-quick-start)
- [Repository layout](#-repository-layout)
- [Controlled comparison](#-controlled-comparison)
- [Evaluation](#-evaluation)
- [Configuration](#-configuration)
- [Paper list](#-paper-list)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

## 🔥 News

- **2026-10**: Results of both [controlled comparisons](#-controlled-comparison), every method at its last
  and at its best validation step, and 30 perception-aware OPD papers in the [paper list](#-paper-list).
- **2026-10**: OPD: a teacher model (a frozen model or an EMA of the policy), OPD from sampled tokens and on
  the full next-token distributions, reproductions of [VA-OPD](examples/reproduction/va_opd/README.md),
  [VGS](examples/reproduction/vgs/README.md), [VCSD](examples/reproduction/vcsd/README.md) and
  [Vision-OPD](examples/reproduction/vision_opd/README.md), an
  [OPD comparison](examples/comparison/opd_qwen3_vl_2b/README.md), and ZoomBench, BLINK, MMStar, AI2D and
  MMMU in the evaluation.
- **2026-10**: 🎉 First release: the official implementation of [CGPO](#-cgpo-acm-mm-2026-oral) (ACM MM
  2026 Oral), 13 reproduced RLVR methods (PAPO, VPPO, DVRP, ToR, PGPO, PEPO, CFPO, VEPO, NoisyRollout, VGPO,
  VAPO, GRIT and DeepEyes), a controlled comparison on Qwen3-VL-4B, and one-click evaluation.

## 🚧 Roadmap

- [x] Results of the two controlled comparisons (RLVR on Qwen3-VL-4B, OPD on Qwen3-VL-2B).
- [x] OPD next to GRPO and DAPO (a teacher model, OPD from sampled tokens and on the full distributions),
  with VA-OPD, VGS, VCSD and Vision-OPD.
- [x] More single-turn methods in the controlled comparison: NoisyRollout, VGPO and VAPO.
- [ ] A grounded-reasoning comparison on data with evidence boxes: TreeVGR, DeFacto and iVGR next to
  GRIT and CGPO.
- [ ] More thinking-with-images methods in the line of DeepEyes: MGPO, Mini-o3 and Pixel Reasoner, and
  MED's evaluation with and without the tool.

Suggestions are welcome: open an issue to propose a method, or see [Contributing](#-contributing).

## 🌟 CGPO (ACM MM 2026 Oral)

This repository is the official code release of

> **CGPO: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning**<br>
> Shengxuming Zhang, Linyun Zhou, Hengrui Lou, Zhenyang Wang, Xiuming Zhang, Zunlei Feng<br>
> *Proceedings of the 34th ACM International Conference on Multimedia (MM '26)*, Rio de Janeiro, Brazil, 2026 · **Oral presentation**<br>
> [[Paper]](https://doi.org/10.1145/3767308.3835969) · [[Code and scripts]](examples/reproduction/cgpo/README.md) · [[BibTeX]](#-citation)

CGPO teaches a vision-language model *evidence-sensitive reasoning*: key reasoning steps ground the
visual evidence they rely on, and the conclusion should change when that evidence is removed. The
policy grounds its evidence inline in the chain of thought; masking the grounded regions gives a
counterfactual image, and the shift of the policy's output distribution between the original and
the counterfactual image measures how much each token depends on the evidence. Responses that
depend more on their evidence receive larger advantages on perception-critical tokens, and a
grounding-consistency reward (the policy re-detects every grounded entity) keeps the evidence
boxes from being inflated. Only answer-level supervision is needed.

```bash
bash scripts/prepare_data.sh cgpo
bash examples/reproduction/cgpo/qwen3_vl_8b_cgpo.sh
bash scripts/eval.sh checkpoints/CGPO-Reproduce/qwen3_vl_8b_cgpo --suite cgpo
```

## 🧪 Reproduced methods

Each method is trained with the scripts in `examples/reproduction/<method>`, which follow its paper's
data, model and hyper-parameters (the README of each directory gives the setting, the baselines, the
differences from the official code and the reported numbers), and all except DeepEyes and Vision-OPD
also with the scripts of the [controlled comparisons](#-controlled-comparison). Except for CGPO these
are unofficial re-implementations.

Methods (and the papers of the [paper list](#-paper-list)) are grouped by the training stage where they make
their perception-aware change, in the order of one training step:

| Stage | RLVR | OPD |
| --- | --- | --- |
| Inputs | **Training data and inputs**: the tasks and images the policy is trained on, and how the images are changed for sampling | **Inputs and privileged information**: what the teacher and the student see (a crop, a higher resolution or a symbolic state for the teacher, a degraded view for the student) |
| Response | **Response structure**: what the policy writes or does in its response (perception text, grounding, visual operations and tools) | **Response structure**: what the student generates (a description before the reasoning, visual operations) |
| Supervision | **Reward**: per-response scores that check perception beyond the correctness of the answer | **Distillation target**: the distribution the student is pulled toward (contrast-sharpened, visual gain, a composed teacher) |
| Credit | **Credit assignment**: how the update is spread over tokens and responses (weights, selection, token-level penalties) | **Credit assignment**: which tokens and responses are distilled, and how strongly |
| Auxiliary | **Auxiliary objectives**: loss terms beside the policy gradient (KL to a changed view, consistency, attention supervision) | **Auxiliary objectives**: student-side loss terms beside the distillation loss |

A method that changes several stages lists all of them in the column *Training stages with perception-aware
changes*, the main one first and in bold; the tables are sorted by the main stage.

### 🎯 RLVR

The policy learns from a verifiable reward on its own rollouts.

| Method | Training stages with perception-aware changes | Paper | Venue | Code | Key idea | Reproduction setting |
| --- | --- | --- | --- | --- | --- | --- |
| [**NoisyRollout**](examples/reproduction/noisyrollout/README.md) | **Training data and inputs** | [NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](https://arxiv.org/abs/2504.13055)<br>Xiangyan Liu et al. · `2504.13055` | NeurIPS 2025 | [GitHub](https://github.com/real-absolute-AI/NoisyRollout) | Half of each group's rollouts are sampled from a diffusion-noised image whose noise anneals to almost zero; all rollouts are trained on the clean image. | Qwen2.5-VL-7B/32B, Geometry3K / K12 |
| [**GRIT**](examples/reproduction/grit/README.md) | **Response structure** · Reward | [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)<br>Yue Fan et al. · `2505.15879` | NeurIPS 2025 | [GitHub](https://github.com/UCSB-AI/GRIT) | Single-turn reasoning chains interleaved with boxes; GRPO-GR rewards the structure, the boxes (with a counting bonus) and the answer, without supervising the boxes. | Qwen2.5-VL-3B / InternVL3-2B, 20 GRIT samples |
| [**DeepEyes**](examples/reproduction/deepeyes/README.md) | **Response structure** · Reward | [DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning](https://arxiv.org/abs/2505.14362)<br>Ziwei Zheng et al. · `2505.14362` | ICLR 2026 | [GitHub](https://github.com/Visual-Agent/DeepEyes) | Multi-turn agentic RL with a zoom-in tool: the crop of the original image returns as a new observation, and a tool bonus goes to correct answers that used the tool. | Qwen2.5-VL-7B / Qwen3-VL-8B, DeepEyes-47k, multi-turn zoom-in tool |
| [**VAPO**](examples/reproduction/vapo/README.md) | **Reward** | [More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models](https://arxiv.org/abs/2509.25848)<br>Xinyu Tian et al. · `2509.25848` | ICLR 2026 | [GitHub](https://github.com/xytian1008/VAPO) | At random points of a correct response the policy judges a claim about the image (Yes / No); the weighted rate of right answers is added to the reward. | Qwen2.5-VL-3B/7B / Qwen3-VL-8B, ViRL39K (single-image) with visual claims |
| [**VEPO**](examples/reproduction/vepo/README.md) | **Credit assignment** | [Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection](https://arxiv.org/abs/2606.03937)<br>Senjie Jin et al. · `2606.03937` | arXiv | [GitHub](https://github.com/Leonnnnnn929/VEPO) | Selects tokens by the JS divergence and entropy gap between the original and a perturbed image, combined with token entropy; only the top 20% of each response are optimized. | Qwen2.5-VL-7B, Geometry3K |
| [**VGPO**](examples/reproduction/vgpo/README.md) | **Credit assignment** | [Visually-Guided Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2604.09349)<br>Zengbin Wang et al. · `2604.09349` | ACL 2026 | [GitHub](https://github.com/wzb-bupt/VGPO) | The last-layer similarity to the mean image token reweights the advantage per token and per response within each group, boosting visual tokens late in a response. | Qwen2.5-VL-3B/7B/32B, ViRL39K |
| [**PGPO**](examples/reproduction/pgpo/README.md) | **Credit assignment** | [Not All Tokens See Equally: Perception-Grounded Policy Optimization for Large Vision-Language Models](https://arxiv.org/abs/2604.01840)<br>Zekai Ye et al. · `2604.01840` | arXiv | - | Visual dependency from a forward with the attention to all visual tokens masked; threshold-gated token weights that keep the response total multiply the advantage. | Qwen2.5-VL-3B/7B, ViRL39K |
| [**ToR**](examples/reproduction/tor/README.md) | **Credit assignment** | [Bridging Perception and Reasoning: Token Reweighting for RLVR in Multimodal LLMs](https://arxiv.org/abs/2603.25077)<br>Jinda Lu et al. · `2603.25077` | arXiv | - | Separate weights for the 30% highest-entropy tokens (reasoning) and the 30% tokens whose log-probability changes most without the image (perception); the other tokens are left out. | Qwen2.5-VL-7B, Geometry3K |
| [**PEPO**](examples/reproduction/pepo/README.md) | **Credit assignment** | [Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought](https://arxiv.org/abs/2603.22847)<br>Yunheng Li et al. · `2603.22847` | arXiv | [GitHub](https://github.com/xzxxntxdy/PEPO) | A perception prior without an extra forward: the hidden-state similarity to the vision tokens, gated by token entropy, is mixed into the token advantages with a growing coefficient. | Qwen2.5-VL-3B / InternVL3-2B, Geometry3K |
| [**VPPO**](examples/reproduction/vppo/README.md) | **Credit assignment** | [Spotlight on Token Perception for Multimodal Reinforcement Learning](https://arxiv.org/abs/2510.09285)<br>Siyuan Huang et al. · `2510.09285` | ICLR 2026 | [GitHub](https://github.com/huaixuheqing/VPPO-RL) | Visual dependency of each token (KL with a perturbed image): only the 40% most dependent tokens of a response get gradients, and its advantage is scaled by its mean dependency. | Qwen2.5-VL-7B / Qwen3-VL-8B, ViRL39K |
| [**CGPO**](examples/reproduction/cgpo/README.md) (ours) | **Credit assignment** · Response structure · Reward | [CGPO: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning](https://doi.org/10.1145/3767308.3835969)<br>Shengxuming Zhang et al. | ACM MM 2026 (Oral) | this repo | Grounds its evidence inline; masking the grounded regions gives a counterfactual image whose KL scales the advantages of perception-critical tokens, and a grounding-consistency reward keeps the boxes tight. | Qwen2.5-VL-7B / Qwen3-VL-8B, ViRL39K (natural images) |
| [**CFPO**](examples/reproduction/cfpo/README.md) | **Auxiliary objectives** | [CFPO: Counterfactual Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2606.23206)<br>Zhangyuan Yu et al. · `2606.23206` | ICML 2026 | [GitHub](https://github.com/Raven-July/CFPO) | Counterfactual inside the model: in every layer the image-token values with the highest text-to-image attention are replaced by their mean, and the KL to this counterfactual is maximized. | Qwen2.5-VL-3B, ViRL39K |
| [**DVRP**](examples/reproduction/dvrp/README.md) | **Auxiliary objectives** | [Thinking with Deltas: Incentivizing Reinforcement Learning via Differential Visual Reasoning Policy](https://arxiv.org/abs/2601.06801)<br>Shujian Gao et al. · `2601.06801` | arXiv | - | Three views per image: maximizes the KL to a patch-masked view, minimizes the KL to a diffusion-noised view and penalizes the entropy on both. | Qwen2.5-VL-3B/7B, ViRL39K |
| [**PAPO**](examples/reproduction/papo/README.md) | **Auxiliary objectives** | [Perception-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2507.06448)<br>Zhenhailong Wang et al. · `2507.06448` | ICLR 2026 | [GitHub](https://github.com/MikeWangWZHL/PAPO) | Implicit perception loss: maximizes the KL between the policy on the original and on a patch-masked image; a double entropy loss keeps this KL from growing in degenerate ways. | Qwen2.5-VL-3B/7B, ViRL39K |

### 🧑‍🏫 OPD

A teacher (another model or the student itself) scores the student's own rollouts token by token.

| Method | Training stages with perception-aware changes | Paper | Venue | Code | Key idea | Reproduction setting |
| --- | --- | --- | --- | --- | --- | --- |
| [**Vision-OPD**](examples/reproduction/vision_opd/README.md) | **Inputs and privileged information** | [Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation](https://arxiv.org/abs/2605.18740)<br>Qianhao Yuan et al. · `2605.18740` | NeurIPS 2026 | [GitHub](https://github.com/VisionOPD/Vision-OPD) | The student sees the full image, an EMA teacher a zoomed crop of the question's region; JS divergence on the student's top-100 tokens. | Qwen3.5-4B / 9B, EMA self-teacher on region crops, Vision-OPD-6K |
| [**VCSD**](examples/reproduction/vcsd/README.md) | **Distillation target** | [Visual Contrastive Self-Distillation](https://arxiv.org/abs/2607.21556)<br>Yijun Liang et al. · `2607.21556` | arXiv | [GitHub](https://github.com/joliang17/VCSD) | Self-distillation from an EMA teacher: the original-image distribution sharpened by its contrast with a black image is the forward-KL target; no answers or rewards. | Qwen3-VL-2B/4B/8B, Qwen3.5-2B/4B/9B, EMA self-teacher, ViRL39K |
| [**VGS**](examples/reproduction/vgs/README.md) | **Distillation target** · Auxiliary objectives | [Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](https://arxiv.org/abs/2606.00564)<br>Hee Suk Yoon et al. · `2606.00564` | ICML 2026 (Spotlight) | - | Splits the teacher into a language prior and a visual gain: besides the reverse KL, the student matches its own text-only distribution times the teacher's image-over-text ratio. | Qwen3-VL-2B / 4B students, GRPO-trained Qwen3-VL-8B teacher, Vision-SR1-47K |
| [**VA-OPD**](examples/reproduction/va_opd/README.md) | **Credit assignment** | [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924)<br>Ruiqi Liu et al. · `2605.21924` | arXiv | - | Weights the distillation by the teacher's visual advantage (its log-probability with the image beyond a pixelated copy), per response and per token. | Qwen3-VL-2B student, 4B / 8B / 32B teachers, Geometry3K / ViRL39K |

The upstream EasyR1 algorithms (GRPO, DAPO, GSPO, CISPO, SAPO, REINFORCE++, RLOO, ReMax) remain
available. The OPD methods learn from a teacher instead of a verifiable reward; they share a teacher role
(`worker.teacher`) and two base objectives, OPD from sampled tokens (`algorithm.adv_estimator=teacher_log_ratio`)
and OPD on the full next-token distributions (`algorithm.distill_loss_coef`), see
[docs/algorithm_parameters.md](docs/algorithm_parameters.md#on-policy-distillation).

## 🚀 Quick start

**1. Install** (Linux, CUDA 12.8 drivers, conda):

```bash
git clone https://github.com/zsxm1998/awesome-perception-aware-rlvr.git
cd awesome-perception-aware-rlvr
bash scripts/install_env.sh          # creates the conda env "parlvr" (torch 2.10, vLLM 0.19, flash-attn 2.8.3)
conda activate parlvr
QWEN35_FASTPATH_ONLY=1 bash scripts/install_env.sh   # optional, Qwen3.5 models only: adds their fast kernels
```

A [Dockerfile](Dockerfile) based on `vllm/vllm-openai:v0.19.0` is also provided.

**2. Prepare data** (downloaded into `./data`, see [data/README.md](data/README.md)):

```bash
bash scripts/prepare_data.sh papo            # training/validation data of one method
bash scripts/prepare_data.sh all             # or everything
bash scripts/prepare_eval_data.sh papo       # benchmarks of the PAPO evaluation suite
```

If `huggingface.co` is slow or blocked, set `HF_ENDPOINT=https://hf-mirror.com`.

**3. Train**:

```bash
bash examples/reproduction/papo/qwen2_5_vl_7b_grpo_papo.sh                       # PAPO-G, paper setting
bash examples/comparison/rlvr_qwen3_vl_4b/cgpo.sh                                 # CGPO in the controlled comparison
N_GPUS_PER_NODE=4 bash examples/reproduction/papo/qwen2_5_vl_7b_grpo.sh trainer.total_epochs=1   # any override
```

Checkpoints go to `checkpoints/<project>/<experiment>/global_step_*` (the scripts keep the latest
save and the one with the best validation reward, `trainer.save_limit=1`); logs go to the console and
to `experiment_log.jsonl` in the same folder (`LOGGER='["console","wandb"]'` or `swanlab` for
online tracking).

**4. Evaluate**:

```bash
bash scripts/eval.sh checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo --suite papo
bash scripts/eval.sh Qwen/Qwen2.5-VL-7B-Instruct --suite papo         # base model
```

The wrapper merges FSDP checkpoints into Hugging Face format, runs every benchmark of the suite
with vLLM on all visible GPUs and writes a summary table. See [eval/README.md](eval/README.md).

**5. Free disk space** (after training has finished):

```bash
python3 scripts/finalize_run.py checkpoints/PAPO-Reproduce/qwen2_5_vl_7b_grpo_papo   # --keep last|best|all|N,..., --dry-run
```

A saved step holds the optimizer states and is 3 to 4 times the size of the model (34 GB for a 4B
model). This keeps the last step and the step with the highest validation reward as Hugging Face
weights in `global_step_N/actor` (about 9 GB each for a 4B model) and deletes the optimizer states and
the other steps; the run can no longer be resumed. `--keep` takes a comma-separated set instead:
`last` or `best` alone, `all` (every saved step, e.g. for the training dynamics; train with
`trainer.save_limit=-1` to save them) or step numbers (`--keep last,50,100`). The controlled
comparisons report the last step, and the best step in a second table without the validation benchmark
(the validation set is often also an evaluated benchmark, and choosing the step on it biases that
score); the best step also serves as a teacher. A single
`.../global_step_N` is finalized alone, without touching the other steps. `scripts/eval.sh` accepts
finalized runs.

## 📁 Repository layout

```
.
├── examples/
│   ├── reproduction/<method>/   # paper settings: common.sh (shared args) + one script per run
│   ├── comparison/              # controlled comparisons: an index and one directory per comparison
│   │   ├── rlvr_qwen3_vl_4b/    #   RLVR: one script per method, shared common.sh
│   │   └── opd_qwen3_vl_2b/     #   OPD: one script per method, shared common.sh
│   ├── config.yaml              # base training config
│   ├── reward_function/         # reward functions
│   └── format_prompt/, system_prompt/, chat_template/
├── eval/                    # one-click evaluation (registry, loaders, scorers, prepare/)
├── scripts/                 # install_env.sh, prepare_data.sh, prepare_eval_data.sh, eval.sh, finalize_run.py, launcher.sh
├── verl/                    # training framework (EasyR1 fork) with all method implementations
├── docs/                    # algorithm parameters, adding a method, BibTeX, figures (assets/)
├── data/                    # downloaded data (git-ignored)
└── tests/                   # unit tests
```

The perception-aware methods are implemented as composable blocks in `verl/trainer/`:
auxiliary (counterfactual) image views, token-level visual-sensitivity signals, token selection,
advantage scaling, auxiliary losses and extra rewards; OPD adds a teacher and the
distillation losses (`distillation.py`). See
[docs/algorithm_parameters.md](docs/algorithm_parameters.md) for every switch,
[docs/implementation_notes.md](docs/implementation_notes.md) for behaviors several methods share
(e.g. the gradient of the k3 KL estimator, loss averaging) and
[docs/add_method.md](docs/add_method.md) for adding a new method.

## 📊 Controlled comparison

Two comparisons train the reproduced methods in one shared setting each and evaluate them on the same 19
benchmarks in three groups (math reasoning, vision-dependent reasoning, and perception and hallucination).
[examples/comparison/](examples/comparison/README.md) ([中文](examples/comparison/README_zh.md)) lists them and
explains [what the results can and cannot say](examples/comparison/README.md#about-the-results).

- **[RLVR](examples/comparison/rlvr_qwen3_vl_4b/README.md)**: every RLVR method except DeepEyes
  ([why](examples/comparison/rlvr_qwen3_vl_4b/README.md#why-deepeyes-is-not-included)) on Qwen3-VL-4B-Instruct,
  with the same data (ViRL39K / MMK12) and the same GRPO hyper-parameters; only the method-specific arguments
  differ, and each method keeps the image perturbation of its paper.
- **[OPD](examples/comparison/opd_qwen3_vl_2b/README.md)**: the two base OPD objectives, VA-OPD, VGS and VCSD on a
  Qwen3-VL-2B student with a GRPO-trained Qwen3-VL-4B as the teacher (VCSD distills from an EMA of the student
  instead), next to GRPO on the same student, with the same data, budget and evaluation; Vision-OPD needs
  region annotations, which this data does not have.

The README of each comparison gives every method's scores per group and per benchmark, at the last and at the
best validation step, and its training time.

## 📈 Evaluation

The evaluation harness covers the union of the benchmarks used by the reproduced papers:
math and multimodal reasoning (Geometry3K, MathVista, We-Math, MMK12, MathVerse, MathVerse-V,
LogicVista, CLEVR counting, MMMU-Pro, MMMU, DynaMath, MathVision), perception and hallucination (POPE,
HallusionBench, MME, MMStar, BLINK, AI2D, TextVQA, CFPO's counterfactual C-VQA-Real and MARS-Bench,
SEED-Bench, ...), grounded reasoning (GRIT's VSR / TallyQA / GQA / OVDEval, refCOCO) and high-resolution
perception (V*, HR-Bench, MME-RealWorld-Lite, ZoomBench). Each benchmark has a download script, and each paper
has a suite (`--suite papo`, `--suite deepeyes`, ...). A checkpoint trained here is prompted as it was trained
(prompt, image size, chat template and tokenizer, read from its run;
[details](eval/README.md#prompts)), whatever the suite. Adding a benchmark takes a loader, a scorer and a
registry entry; see the tutorial in [eval/README.md](eval/README.md#adding-a-new-benchmark).

> [!IMPORTANT]
> The papers use different evaluation protocols (rule-based avg@8 at temperature 1.0 for
> PAPO/VPPO/PGPO/DVRP/CFPO; greedy decoding with an LLM judge for ToR/VEPO/NoisyRollout/VAPO/GRIT/DeepEyes/Vision-OPD). This
> repository evaluates every model with one harness, so compare methods against baselines run in
> this repository rather than against numbers copied across papers.

## 🔧 Configuration

Environment variables understood by every training script:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | per script | Hugging Face id or local path of the policy |
| `TEACHER_PATH` | per script | teacher of the OPD scripts |
| `DATA_ROOT` | `./data` | prepared training data |
| `LOGGER` | `["console","file"]` | add `"wandb"`, `"swanlab"`, `"tensorboard"` or `"mlflow"` |
| `LOGGER_FINISH_TIMEOUT` | `300` | seconds SwanLab may take to close when training ends; after that the run ends anyway, and SwanLab data not yet uploaded may be missing online (`swanlab sync` uploads the local backup) |
| `N_GPUS_PER_NODE` | per script | GPUs per node |
| `NNODES` | `1` | number of nodes (start a Ray cluster first for multi-node runs) |
| `EXPERIMENT_NAME` | script name | run name; checkpoints in `checkpoints/<project>/<experiment>` |

Any config key can be appended as `key=value` (see [examples/config.yaml](examples/config.yaml)
and [verl/trainer/config.py](verl/trainer/config.py)). Set `USE_MODELSCOPE_HUB=1` to download models
from ModelScope.

**`<think>` in Qwen3-VL Instruct models.** These checkpoints ship `<think>` and `</think>` as added
tokens that were never trained (only the Thinking checkpoints use them). A format prompt that asks
for `<think> ... </think>` is then encoded with two untrained tokens and is not followed: on 256
ViRL39K prompts the format reward of Qwen3-VL-4B-Instruct is 0.00 with the released tokenizer and
0.47 when the tags are plain text, at the same accuracy. By default
(`worker.actor.model.plain_think_tokens=auto`) such tokens are therefore tokenized as plain text:
the tokenizer is loaded as usual (Hugging Face hub, ModelScope or a local path), saved with the two
added tokens removed under `~/.cache/parlvr/tokenizers` (`PARLVR_CACHE_DIR`), and used by training,
rollout and evaluation; checkpoints save this tokenizer. Models whose chat template uses the tokens
(Qwen3-VL Thinking) or that do not have them (Qwen2.5-VL, InternVL) are not changed. Set
`worker.actor.model.plain_think_tokens=false` to keep the released tokenizer (the evaluation reads the
setting from a checkpoint's training record; `--plain-think-tokens false` for other models).

## 📚 Paper list

Verified against arXiv, OpenReview and the official repositories (on 2026-10-01; the perception-aware OPD
papers on 2026-10-15); venues are only listed when confirmed by a primary source (sources:
[docs/paper_list_sources.md](docs/paper_list_sources.md), BibTeX: [docs/paper_list.bib](docs/paper_list.bib)).
Newest first within each table. Each paper is listed under the training stage of its main perception-aware
change (see [Reproduced methods](#-reproduced-methods)); the methods implemented in this repository are listed
there and not repeated here. Pull requests that add papers are welcome.

### 🎯 Perception-aware RLVR

RLVR for large vision-language models with a perception-aware change in one of the training stages;
grounded reasoning and thinking with images belong to the response structure.

#### Training data and inputs

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2025-12 | [Boosting RL-Based Visual Reasoning with Selective Adversarial Entropy Intervention](https://arxiv.org/abs/2512.10414)<br>Yang Yu et al. · `2512.10414` | arXiv | - | **SaEI**: Entropy-guided adversarial perturbation of visual inputs during RL sampling to enlarge the explored answer space. |
| 2025-09 | [Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models](https://arxiv.org/abs/2509.13031)<br>Yan Chen et al. · `2509.13031` | arXiv | [GitHub](https://github.com/cythu/PeBR-R1) | **PeBR-R1**: Two-stage RL: first perception (coarse and fine visual understanding), then reasoning, with dataset-level sampling. |
| 2025-06 | [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/abs/2506.10128)<br>Xiyao Wang et al. · `2506.10128` | NeurIPS 2025 | [GitHub](https://github.com/si0wang/ViCrit) | **ViCrit**: RL proxy task: localize one injected visual hallucination in a human-written caption; exact-match verifiable reward. |
| 2025-06 | [Revisiting Visual Understanding in Multimodal Reasoning through a Lens of Image Perturbation](https://arxiv.org/abs/2506.09736)<br>Yuting Li et al. · `2506.09736` | arXiv | [GitHub](https://github.com/YutingLi0606/Vision-Matters) | **Vision-Matters**: Distractor concatenation, dominance-preserving mixup and random rotation perturbations plugged into SFT, DPO or GRPO. |
| 2025-05 | [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](https://arxiv.org/abs/2505.14231)<br>Sule Bai et al. · `2505.14231` | arXiv | [GitHub](https://github.com/AMAP-ML/UniVG-R1) | **UniVG-R1**: Universal visual grounding: CoT cold-start SFT then rule-based RL with difficulty-aware weight adjustment. |
| 2025-04 | [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](https://arxiv.org/abs/2504.07954)<br>En Yu et al. · `2504.07954` | NeurIPS 2025 | [GitHub](https://github.com/linkangheng/PR1) | **Perception-R1 (Yu et al.)**: Rule-based GRPO for perception policies (grounding, counting, OCR, detection); studies perceptual complexity and reward design. |
| 2025-04 | [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/abs/2504.07615)<br>Haozhan Shen et al. · `2504.07615` | arXiv | [GitHub](https://github.com/om-ai-lab/VLM-R1) | **VLM-R1**: R1-style rule-based RL framework for VLMs on visual understanding tasks; better generalization than SFT. |
| 2025-03 | [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)<br>Ziyu Liu et al. · `2503.01785` | ICCV 2025 | [GitHub](https://github.com/Liuziyu77/Visual-RFT) | **Visual-RFT**: GRPO with verifiable perception rewards (e.g., IoU) for detection, reasoning grounding and fine-grained classification. |

#### Response structure

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-08 | [ReGround: Restoring Visual Grounding in Multi-Step Reasoning through Self-Diagnosis and Visual Re-Examination](https://arxiv.org/abs/2608.04385)<br>Lei Peng et al. · `2608.04385` | ACM MM 2026 | [GitHub](https://github.com/sespoir/ReGround) | **ReGround**: SFT then GRPO teach self-diagnosis of grounding failures and selective image re-injection during multi-step reasoning. |
| 2026-07 | [Perceive-to-Reason: Decoupling Perception and Reasoning for Fine-Grained Visual Reasoning](https://arxiv.org/abs/2607.01191)<br>Hongxing Li et al. · `2607.01191` | arXiv | [GitHub](https://github.com/ZJU-REAL/Perceive-to-Reason) | **P2R**: Perceiver localizes evidence, Reasoner answers from annotated image and crops; alternating perception/reasoning GRPO. |
| 2026-06 | [Thinking with Visual Grounding](https://arxiv.org/abs/2606.16122)<br>Junkai Zhang et al. · `2606.16122` | arXiv | [GitHub](https://github.com/Jun-Kai-Zhang/visually_grounded_thinking) | **Visually grounded thinking**: Interleaves thoughts with point/box groundings; grounding-aware RL adds dense grounding rewards to answer rewards. |
| 2026-05 | [iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](https://arxiv.org/abs/2605.31096)<br>Chang-Bin Zhang et al. · `2605.31096` | ICML 2026 | [GitHub](https://github.com/Visual-AI/iVGR) | **iVGR**: Dual-stream RL with a consistency reward internalizes localization into textual CoT; no explicit boxes needed at inference. |
| 2026-03 | [Seeing with You: Perception-Reasoning Coevolution for Multimodal Reasoning](https://arxiv.org/abs/2603.28618)<br>Ziqi Miao et al. · `2603.28618` | arXiv | [GitHub](https://github.com/Dtc7w3PQ/PRCO) | **PRCO**: One shared policy plays Observer (question-tailored evidence caption, utility reward) and Solver (answer, outcome reward). |
| 2026-02 | [What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom](https://arxiv.org/abs/2602.01334)<br>Yan Ma et al. · `2602.01334` | ICML 2026 | [GitHub](https://github.com/GAIR-NLP/Med) | **MED**: Analysis of crop-and-zoom tool-use RL: gains are mostly intrinsic; RL mainly reduces tool-induced harm. |
| 2025-12 | [Deep But Reliable: Advancing Multi-turn Reasoning for Thinking with Images](https://arxiv.org/abs/2512.17306)<br>Wenhao Yang et al. · `2512.17306` | arXiv | - | **DRIM**: Multi-turn tool-based thinking with images; redundancy-penalized policy optimization rewards self-reflection and multi-scale exploration. |
| 2025-12 | [Thinking with Images via Self-Calling Agent](https://arxiv.org/abs/2512.08511)<br>Wenxi Yang et al. · `2512.08511` | arXiv | [GitHub](https://github.com/YWenxi/think-with-images-through-self-calling) | **sCoT**: Recasts interleaved multimodal CoT as language-only CoT that calls parameter-sharing subagents; optimized with GRPO. |
| 2025-12 | [Thinking with Programming Vision: Towards a Unified View for Thinking with Images](https://arxiv.org/abs/2512.03746)<br>Zirun Guo et al. · `2512.03746` | CVPR 2026 | [GitHub](https://github.com/ByteDance-BandAI/CodeVision) | **CodeVision**: Code as a universal tool for image operations; SFT then RL with a dense process reward. |
| 2025-11 | [CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](https://arxiv.org/abs/2511.19820)<br>Miguel Carvalho et al. · `2511.19820` | CVPR 2026 Workshop (GRAIL-V) | [GitHub](https://github.com/miguelscarv/cropvlm) | **CropVLM**: RL-trained external cropping model (no box labels) that zooms into relevant regions for any target VLM. |
| 2025-11 | [DeepEyesV2: Toward Agentic Multimodal Model](https://arxiv.org/abs/2511.05271)<br>Jack Hong et al. · `2511.05271` | ICLR 2026 | [GitHub](https://github.com/Visual-Agent/DeepEyesV2) | **DeepEyesV2**: Agentic MLLM invoking code execution and web search; cold-start SFT then RL; introduces RealX-Bench. |
| 2025-09 | [Mixture-of-Visual-Thoughts: Exploring Context-Adaptive Reasoning Mode Selection for General Visual Reasoning](https://arxiv.org/abs/2509.22746)<br>Zejun Li et al. · `2509.22746` | ICLR 2026 | [GitHub](https://github.com/Future-Living-Lab/mixture-of-visual-thoughts) | **MoVT / AdaVaR**: Unifies multiple visual reasoning modes via SFT; AdaGRPO induces context-adaptive mode selection. |
| 2025-09 | [Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search](https://arxiv.org/abs/2509.07969)<br>Xin Lai et al. · `2509.07969` | ICLR 2026 | [GitHub](https://github.com/Mini-o3/Mini-o3) | **Mini-o3**: Deep multi-turn visual search: Visual Probe dataset, diverse cold-start trajectories, over-turn masking in RL. |
| 2025-09 | [Reinforced Visual Perception with Tools](https://arxiv.org/abs/2509.01656)<br>Zetong Zhou et al. · `2509.01656` | arXiv | [GitHub](https://github.com/ls-kelvin/REVPT) | **ReVPT**: GRPO-based RL that teaches MLLMs to reason with a suite of four visual perception tools. |
| 2025-08 | [Thyme: Think Beyond Images](https://arxiv.org/abs/2508.11630)<br>Yi-Fan Zhang et al. · `2508.11630` | ICLR 2026 | [GitHub](https://github.com/yfzhang114/Thyme) | **Thyme**: Generates and executes code for image manipulation and computation; SFT then RL with GRPO-ATS. |
| 2025-07 | [Look-Back: Implicit Visual Re-focusing in MLLM Reasoning](https://arxiv.org/abs/2507.03019)<br>Shuo Yang et al. · `2507.03019` | arXiv | [GitHub](https://github.com/PKU-YuanGroup/Look-Back) | **Look-Back**: SFT then RL teach MLLMs to autonomously re-focus on visual inputs late in reasoning, without image re-injection. |
| 2025-07 | [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](https://arxiv.org/abs/2507.07999)<br>Haochen Wang et al. · `2507.07999` | ICLR 2026 | [GitHub](https://github.com/Haochen-Wang409/TreeVGR) | **TreeVGR**: RL jointly supervising localization and reasoning for traceable evidence; introduces the TreeBench benchmark. |
| 2025-07 | [High-Resolution Visual Reasoning via Multi-Turn Grounding-Based Reinforcement Learning](https://arxiv.org/abs/2507.05920)<br>Xinyu Huang et al. · `2507.05920` | ACL 2026 Findings | [GitHub](https://github.com/EvolvingLMMs-Lab/MGPO) | **MGPO**: Multi-turn RL cropping sub-images at model-predicted grounding coordinates, using only answer-correctness reward. |
| 2025-06 | [Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing](https://arxiv.org/abs/2506.09965)<br>Junfei Wu et al. · `2506.09965` | NeurIPS 2025 | [GitHub](https://github.com/AntResearchNLP/ViLaSR) | **ViLaSR**: Drawing to reason in space (boxes, auxiliary lines); cold start, reflective rejection sampling, then RL. |
| 2025-06 | [Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning](https://arxiv.org/abs/2506.04034)<br>Qing Jiang et al. · `2506.04034` | ICLR 2026 | [GitHub](https://github.com/IDEA-Research/Rex-Thinker) | **Rex-Thinker**: Object referring as CoT over candidate instances; cold-start SFT then GRPO; learns to abstain when nothing matches. |
| 2025-05 | [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.14677)<br>Jiaer Xia et al. · `2505.14677` | TMLR | [GitHub](https://github.com/maifoundations/Visionary-R1) | **Visionary-R1**: Caption-reason-answer output format trained with pure RL to mitigate shortcut learning. |
| 2025-05 | [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/abs/2505.23678)<br>Gabriel Sarch et al. · `2505.23678` | NeurIPS 2025 | [GitHub](https://github.com/Gabesarch/grounded-rl) | **ViGoRL**: RL anchors each reasoning step to image coordinates; multi-turn variant zooms into predicted coordinates. |
| 2025-05 | [ACTIVE-o3: Empowering MLLMs with Active Perception via Pure Reinforcement Learning](https://arxiv.org/abs/2505.21457)<br>Muzhi Zhu et al. · `2505.21457` | ICML 2026 | [GitHub](https://github.com/aim-uofa/Active-o3) | **ACTIVE-o3**: GRPO framework giving MLLMs active perception (region selection) with a dual-form reward, no region supervision. |
| 2025-05 | [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.20272)<br>Meng Cao et al. · `2505.20272` | arXiv | [GitHub](https://github.com/zzzhhzzz/Ground-R1) | **Ground-R1**: Thinking-with-images trained with Scale Relative Policy Optimization to de-bias reward learning across evidence-region sizes. |
| 2025-05 | [Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning](https://arxiv.org/abs/2505.19702)<br>Minheng Ni et al. · `2505.19702` | NeurIPS 2025 | [GitHub](https://github.com/kodenii/Point-RFT) | **Point-RFT**: Format finetuning on 71K point-grounded rationales, then reinforcement finetuning for visual document understanding. |
| 2025-05 | [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/abs/2505.19255)<br>Mingyuan Wu et al. · `2505.19255` | ICLR 2026 | [GitHub](https://github.com/VTool-R1/VTool-R1) | **VTool-R1**: RL with Python visual-editing tools yields interleaved text-image CoT using only outcome rewards. |
| 2025-05 | [VLM-R³: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192)<br>Chaoya Jiang et al. · `2505.16192` | NeurIPS 2025 | - | **VLM-R³**: Region-Conditioned RPO rewards choosing regions, crop/zoom transformations, and weaving sub-images into interleaved CoT. |
| 2025-05 | [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966)<br>Haozhe Wang et al. · `2505.15966` | NeurIPS 2025 | [GitHub](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) | **Pixel Reasoner**: Pixel-space operations (zoom-in, select-frame); instruction tuning then curiosity-driven RL. |
| 2025-05 | [Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs](https://arxiv.org/abs/2505.15436)<br>Xintong Zhang et al. · `2505.15436` | arXiv | [GitHub](https://github.com/xtong-zhang/Chain-of-Focus) | **Chain-of-Focus**: Adaptive search-and-zoom on key regions; SFT on MM-CoF, then RL with outcome and format rewards. |
| 2025-05 | [Visual Agentic Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.14246)<br>Ziyu Liu et al. · `2505.14246` | arXiv | [GitHub](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT) | **Visual-ARFT**: Agentic RFT for web search and code-based image manipulation (crop, rotate); introduces MAT benchmark. |
| 2025-05 | [OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning](https://arxiv.org/abs/2505.08617)<br>Zhaochen Su et al. · `2505.08617` | arXiv | [GitHub](https://github.com/zhaochen0110/OpenThinkIMG) | **OpenThinkIMG**: Open framework for tool-augmented LVLMs; V-ToolRL learns adaptive vision-tool invocation from task success. |

#### Reward

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-08 | [Evidence-RL: Towards Evidence-intensive Visual Reasoning](https://arxiv.org/abs/2608.08021)<br>Haojie Huang et al. · `2608.08021` | NeurIPS 2026 | [GitHub](https://github.com/evidencerl/code) | **Evidence-RL (CED)**: Neutralizes object evidence regions vs. matched non-evidence regions; GRPO rewards correct answers that causally rely on evidence. |
| 2026-05 | [Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning](https://arxiv.org/abs/2605.14054)<br>Haozhe Wang et al. · `2605.14054` | ICML 2026 | - | **MoCA**: Perception Verification via a blindfolded text reasoner rewards perception separately; credit routed to bad seeing vs. bad thinking. |
| 2026-04 | [Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization](https://arxiv.org/abs/2604.08476)<br>Sai Srinivas Kancheti et al. · `2604.08476` | COLM 2026 | - | **Faithful GRPO**: GRPO with logical-consistency and visual-grounding constraints enforced via Lagrangian dual ascent. |
| 2026-02 | [Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs](https://arxiv.org/abs/2602.08241)<br>Siqu Ou et al. · `2602.08241` | arXiv | - | **SAYO**: RL with a region-level visual-attention reward that aligns optimization with visually grounded reasoning steps. |
| 2025-12 | [Learning When to Look: A Disentangled Curriculum for Strategic Perception in Multimodal Reasoning](https://arxiv.org/abs/2512.17227)<br>Siqi Yang et al. · `2512.17227` | CVPR 2026 Findings | [GitHub](https://github.com/gaozilve-max/learning-when-to-look) | **Learning When to Look**: Disentangled SFT curriculum, then RL with a Pivotal Perception Reward teaching when to look. |
| 2025-09 | [VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding](https://arxiv.org/abs/2509.24776)<br>Yizhuo Ding et al. · `2509.24776` | arXiv | - | **VTPerception-R1**: Perception-augmented SFT, then RL with visual, textual and consistency perception rewards. |
| 2025-09 | [DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning](https://arxiv.org/abs/2509.20912)<br>Tianrun Xu et al. · `2509.20912` | ICML 2026 | [GitHub](https://github.com/tinnel123666888/defacto) | **DeFacto**: Positive, counterfactual (evidence masked) and random-masking training with GRPO rewards for evidence-answer consistency. |
| 2025-08 | [Self-Rewarding Vision-Language Model via Reasoning Decomposition](https://arxiv.org/abs/2508.19652)<br>Zongxia Li et al. · `2508.19652` | arXiv | [GitHub](https://github.com/zli12321/Vision-SR1) | **Vision-SR1**: Self-reward checks that generated visual descriptions suffice to answer without the image; decoupled visual and answer advantages. |
| 2025-06 | [Perception-R1: Advancing Multimodal Reasoning Capabilities of MLLMs via Visual Perception Reward](https://arxiv.org/abs/2506.07218)<br>Tong Xiao et al. · `2506.07218` | ICLR 2026 | [GitHub](https://github.com/tongxiao2002/Perception-R1) | **Perception-R1 (Xiao et al.)**: Judging LLM scores consistency between responses and visual annotations as a visual perception reward in RLVR. |

#### Credit assignment

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-09 | [Reinforcing Multimodal Reasoning via Token-Level Perception-Grounded Advantage Estimation](https://arxiv.org/abs/2609.39168)<br>Zhihan Zhang et al. · `2609.39168` | ACM MM 2026 | [GitHub](https://github.com/Zhihan72/TPAE) | **TPAE**: Scores each token's consistency with vision-dependency/entropy patterns of correct rollouts to modulate sequence-level advantages. |
| 2026-09 | [Anchoring What Matters: A Dual-Level Learning Framework for Visually-Grounded Multimodal Reasoning](https://arxiv.org/abs/2609.18057)<br>Xinxin Song et al. · `2609.18057` | arXiv | - | **PIVOT**: Replays visually-grounded past trajectories as anchors; allocates extra advantage by local visual support and downstream impact. |
| 2026-06 | [PRPO: Perception-Reinforced Policy Optimization via Token-Level Dynamic Advantage Reshaping](https://arxiv.org/abs/2606.08708)<br>Qiming Li et al. · `2606.08708` | arXiv | - | **PRPO**: Robust Visual Dependency finds grounded, perturbation-stable tokens; Perceptual Advantage Reshaping amplifies their token-level advantages. |
| 2026-06 | [DyCo-RL: Dynamic Cross-Modal Coordination for Visual Reasoning](https://arxiv.org/abs/2606.08035)<br>Hangui Lin et al. · `2606.08035` | arXiv | [GitHub](https://github.com/Sammy20207109/DyCo-RL) | **DyCo-RL**: Assigns tokens visual/text roles via Fisher-Rao attention shifts; reweights advantages by attention-role alignment. |
| 2026-05 | [Structured Role-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2605.07274)<br>Bingqing Jiang et al. · `2605.07274` | arXiv | - | **SRPO**: Role-aware token weights: perception tokens by original-vs-corrupted image dependency, reasoning tokens by consistency with perception. |
| 2026-04 | [Improving Vision-language Models with Perception-centric Process Reward Models](https://arxiv.org/abs/2604.24583)<br>Yingqian Min et al. · `2604.24583` | CVPR 2026 | [GitHub](https://github.com/RUCAIBox/Perceval) | **Perceval**: Perception-centric PRM flags hallucinated image claims; RL applies token-level penalties on those spans instead of sequence-level advantages. |
| 2025-08 | [COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](https://arxiv.org/abs/2508.04182)<br>Peizheng Guo et al. · `2508.04182` | CVPR 2026 | - | **COPO**: Token-level causal-completeness reward (sufficiency/necessity) builds causally informed GRPO advantages to reduce hallucination. |

#### Auxiliary objectives

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-07 | [SIVA-RL: Sensitivity-Invariance Visual Alignment for Multimodal Reinforcement Learning](https://arxiv.org/abs/2607.13931)<br>Cheng Tang et al. · `2607.13931` | arXiv | [GitHub](https://github.com/tchenglv520/SIVA-RL) | **SIVA-RL**: Within-image PatchSwap interventions; audited reward drop routes each pair to sensitivity or invariance alignment. |
| 2026-05 | [Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR](https://arxiv.org/abs/2605.30912)<br>Ruina Hu et al. · `2605.30912` | EMNLP 2026 | [GitHub](https://github.com/Nrich-sunny/Attend-to-Evidence) | **EASE**: On high-reward rollouts only, supervises response-to-image attention toward annotated evidence regions (training-time labels). |
| 2026-05 | [Reinforcing Multimodal Reasoning Against Visual Degradation](https://arxiv.org/abs/2605.09262)<br>Rui Liu et al. · `2605.09262` | arXiv | - | **ROMA**: Teacher-forced corrupted views of clean trajectories, worst-case token KL, correctness-conditioned regularization for degradation robustness. |
| 2026-01 | [CPPO: Contrastive Perception Policy Optimization for VLM Agents](https://arxiv.org/abs/2601.00501)<br>Ahmad Rezaei et al. · `2601.00501` | ICML 2026 Workshop (AIWILD) | [GitHub](https://github.com/vbdi/cppo) | **CPPO**: Contrastive Perception Loss applied to perception tokens detected by entropy shifts under perturbed images. |
| 2025-09 | [Perception-Consistency Multimodal Large Language Models Reasoning via Caption-Regularized Policy Optimization](https://arxiv.org/abs/2509.21854)<br>Songjun Tu et al. · `2509.21854` | arXiv | - | **CapPO**: Caption-based consistency regularization between image- and caption-conditioned responses, plus KL-weighted advantage estimation. |

### 🧑‍🏫 Perception-aware OPD

OPD for large vision-language models whose teacher, target or token weights come from visual evidence: a privileged view of the image for the teacher, a contrast between views, or the visual dependence of each token.

#### Inputs and privileged information

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [Distilling Routed 3D Privilege for Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2610.12355)<br>Hongxing Li et al. · `2610.12355` | arXiv | [GitHub](https://github.com/ZJU-REAL/GPD) | **GPD**: Depth, semantic and bird's-eye-view cues rendered as text are routed to the self-teacher as privilege for spatial reasoning; a privileged KL on incorrect trajectories augments GRPO. |
| 2026-10 | [Where-OPD: Spatially Guided On-Policy Self-Distillation of MLLMs with Synthetic Scenes](https://arxiv.org/abs/2610.02117)<br>Sophia Sirko-Galouchenko et al. · `2610.02117` | arXiv | [GitHub](https://github.com/sirkosophia/Where-OPD) | **Where-OPD**: The self-teacher receives textual, spatially grounded guidance on the visual elements a query needs, from procedurally generated scenes with known object positions. |
| 2026-09 | [RS-OPSD: Reliable Privileged On-Policy-Self-Distillation for Ultra-High-Resolution Remote Sensing VQA](https://arxiv.org/abs/2609.38072)<br>Chengjie Jiang et al. · `2609.38072` | arXiv | - | **RS-OPSD**: Zoomed evidence crops as the self-teacher's privilege for ultra-high-resolution remote sensing VQA, with safeguards against context loss and unreliable teachers (GeoEvidence-6K). |
| 2026-09 | [MM-OPD: Towards One More Bottleneck Between Perception and Reasoning](https://arxiv.org/abs/2609.32690)<br>Jintao Tong et al. · `2609.32690` | arXiv | - | **MM-OPD**: A frozen self-teacher also sees a caption or the generating code of the image (a symbolic view); residual token targets correct the image-conditioned prediction toward the right evidence. |
| 2026-09 | [Progressive-View On-Policy Distillation for Regional-to-Global Transfer in Multimodal LLMs](https://arxiv.org/abs/2609.32333)<br>Shanfeng Huang et al. · `2609.32333` | arXiv | - | **PVD**: Progressive views move the student from the region crop to the full image through a padded crop; a regional-advantage weight reallocates the token supervision. |
| 2026-08 | [ArmorOCR: Grounded Adversarial Visual Perception via Observation-Transferred Self-Distillation](https://arxiv.org/abs/2608.20122)<br>Linhan Cao et al. · `2608.20122` | arXiv | [GitHub](https://github.com/ant-research/ArmorOCR) | **ArmorOCR**: Self-distillation from privileged transformed observations for adversarial OCR, then GRPO for grounded OCR; introduces the AdvSpot benchmark. |
| 2026-08 | [Self-Supervised Visual On-Policy Distillation](https://arxiv.org/abs/2608.14144)<br>Yijiang Li et al. · `2608.14144` | arXiv | - | **S²VOPD**: Removes information from the student instead of adding it to the teacher: the teacher sees the original image, the student a strongly augmented view; no annotations, rewards or stronger teacher. |
| 2026-08 | [ADOPD: Reference-Privileged On-Policy Distillation for MLLM-Based Industrial Anomaly Detection](https://arxiv.org/abs/2608.09789)<br>Jingtai He et al. · `2608.09789` | arXiv | [GitHub](https://github.com/withTai/ADOPD) | **ADOPD**: A reference-privileged teacher for industrial anomaly detection; its likelihood gap between matched and mismatched references calibrates each response's weight. |
| 2026-07 | [RP-OPSD: Resolution-Privileged On-Policy Self-Distillation for Multimodal Large Language Models](https://arxiv.org/abs/2607.24447)<br>Qihui Zhu et al. · `2607.24447` | arXiv | [GitHub](https://github.com/sansanyuchen/RP-OPSD) | **RP-OPSD**: Resolution as privilege: the student samples from a low-resolution image and the self-teacher scores with the original resolution. |
| 2026-07 | [Self-Boosting Vision-Language Models with Noisy Student On-Policy Self-Distillation](https://arxiv.org/abs/2607.23125)<br>Shuai Wang et al. · `2607.23125` | arXiv | - | **NOPD**: Noisy-student self-distillation: the model learns on corrupted inputs from its own clean-input predictions, without external models or answers. |
| 2026-06 | [Learning Visual Spatial Planning from Symbolic State via Modality-Gap-Aware Self-Distillation](https://arxiv.org/abs/2606.06076)<br>Haocheng Luo et al. · `2606.06076` | arXiv | [GitHub](https://github.com/Oranger-l/MGSD) | **MGSD**: Self-distillation guided by the symbolic state for visual spatial planning, after a grounding cold start; inference uses the image only. |
| 2026-06 | [ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation](https://arxiv.org/abs/2606.05718)<br>Kanghui Tian et al. · `2606.05718` | arXiv | [GitHub](https://github.com/tiankanghui/ViCuR) | **ViCuR**: Replaces answer-side privilege with query-related visual cues and adds a cue recovery module, so the student can recover the cues from its own input. |

#### Response structure

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-09 | [On-Policy Visual Evidence Distillation](https://arxiv.org/abs/2609.36838)<br>Shaohang Wei et al. · `2609.36838` | arXiv | [GitHub](https://github.com/sylvain-wei/ReVuE) | **ReVuE**: OPD for visual agents that interleave reasoning with image operations; corrections target the failing stage: acquiring, reading or grounding the evidence. |
| 2026-06 | [Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation](https://arxiv.org/abs/2606.19120)<br>Sihan Wang et al. · `2606.19120` | arXiv | - | **ViGOS**: The student describes the image before reasoning; an image-only perception teacher supervises the description and a privileged teacher the reasoning, against reference-answer shortcuts. |
| 2026-06 | [Thinking Without Images: Internalizing Visual Manipulation with On-Policy Self-Distillation](https://arxiv.org/abs/2606.08719)<br>Yishuo Cai et al. · `2606.08719` | arXiv | [GitHub](https://github.com/walkeralan123/Imagine-OPD) | **Imagine-OPD**: A teacher with zoomed evidence views supervises the student's imagined zoom-in reasoning, internalizing thinking with images without tool calls. |

#### Distillation target

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [CV-OPD: Contrastive Vision On-policy Distillation](https://arxiv.org/abs/2610.13264)<br>Haowei Wang et al. · `2610.13264` | arXiv | - | **CV-OPD**: Compares the teacher's distributions under a positive and a negative crop on the same student prefix and adds a contrastive term, so that the student relies on the visual evidence. |
| 2026-09 | [LEGO-OPD: Factorized Teacher Composition for Multimodal On-Policy Distillation](https://arxiv.org/abs/2610.00333)<br>Jaeyun Shin et al. · `2610.00333` | arXiv | - | **LEGO-OPD**: Composes a language-expert prior and a grounding-expert factor into one teacher distribution, so the visual grounding signal transfers without the VLM teacher's language prior. |
| 2026-09 | [Distill the Visual Evidence, Not Just the Answer: Cross-World On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2609.38777)<br>Yuanhao Sun et al. · `2609.38777` | arXiv | [GitHub](https://github.com/baokou-fw2/CWAD) | **CW-OPD**: Builds two visual worlds that differ in answer-critical evidence; the student distills the teacher's cross-world change of response, not only its answer. |
| 2026-09 | [OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation](https://arxiv.org/abs/2609.16459)<br>Chenhao Qiu et al. · `2609.16459` | arXiv | [GitHub](https://github.com/Echochef/OPD-Aha) | **OPD-Aha**: Reconstructs the target from the teacher's preference with the real image over a visual null, so that the student learns to reflect when its reasoning contradicts the image. |
| 2026-08 | [Distill What the Student Can See: Fisher-Projected On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2608.01263)<br>Leyan Xue et al. · `2608.01263` | arXiv | - | **FP-OPD**: Distills only the teacher corrections the student can realize, projecting the teacher-student gap onto the student's visual tangent space estimated from image perturbations. |
| 2026-07 | [VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation](https://arxiv.org/abs/2607.28590)<br>Kangning Zhang et al. · `2607.28590` | arXiv | [GitHub](https://github.com/DeepExperience/VAD_Multimodal_OPD) | **VAD**: The teacher scores each prefix with the evidence crop present and degraded; only the part of its correction along this visual direction forms the target. |

#### Credit assignment

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-10 | [VC-OPD: Visual Counterfactual On-Policy Distillation for Grounded Vision-Language Reasoning](https://arxiv.org/abs/2610.14939)<br>Jiahe Feng · `2610.14939` | arXiv | - | **VC-OPD**: The teacher scores each student prefix on the original and on a counterfactual image with the instance evidence corrupted; the gap reweights the distillation toward image-dependent tokens. |
| 2026-09 | [Evidence-Aligned Multimodal On-Policy Self-Distillation for Fine-Grained Visual Understanding](https://arxiv.org/abs/2609.34672)<br>Nanxing Hu et al. · `2609.34672` | arXiv | - | **EAD**: Keeps the crop-conditioned teacher as the target but weights its corrections by a separate evidence reference that removes model-state and lost-context effects. |
| 2026-09 | [BAS-OPD: Budget-Aware Selective On-Policy Self-Distillation for Fine-Grained Multimodal Perception](https://arxiv.org/abs/2609.25891)<br>Zihan Chen et al. · `2609.25891` | arXiv | - | **BAS-OPD**: Selects which rollouts query the privileged-view teacher under a budget (random, uncertainty-based or a learned utility) for fine-grained perception. |
| 2026-08 | [Perception Before Supervision: Self-Contained Visual Distillation from Counterfactual Blind Spots](https://arxiv.org/abs/2608.09931)<br>Shravan Venkatraman et al. · `2608.09931` | BMVC 2026 | [GitHub](https://github.com/mbzuai-oryx/CVPD) | **CVPD**: Finds visual blind spots in the model's own responses (zooming into a region sharpens the answer, removing it changes nothing) and distills them without external supervision. |
| 2026-08 | [OPD-V: Visual On-Policy Self-Distillation with Modality Balance](https://arxiv.org/abs/2608.05131)<br>Aniri et al. · `2608.05131` | arXiv | [GitHub](https://github.com/aniri15/OPD-V) | **OPD-V**: A positive (zoom-in) and a negative (masked) self-teacher; their logit margin defines a modality-balance trust region for selective distillation. |
| 2026-07 | [Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners](https://arxiv.org/abs/2607.28336)<br>Feng Xiong et al. · `2607.28336` | arXiv | - | **PCD**: Strengthens distillation only where downstream failure and teacher-student disagreement both point to a correctable perception error, with separated perception and reasoning rollouts. |
| 2026-07 | [Med-OPD: Improving Medical Vision-Language Models via Evidence-Aware On-Policy Distillation](https://arxiv.org/abs/2607.16303)<br>Yunhang Qian et al. · `2607.16303` | arXiv | [GitHub](https://github.com/yunhang8658/MedOPD) | **Med-OPD**: A medical evidence advantage from a teacher counterfactual focuses distillation on the diagnosis-critical, evidence-dependent tokens of medical VLMs. |
| 2026-06 | [V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning](https://arxiv.org/abs/2606.25319)<br>Haoxiang Sun et al. · `2606.25319` | arXiv | [GitHub](https://github.com/eVI-group-SCU/V-Zero) | **V-Zero**: Answer-label-free OPD for fine-grained visual reasoning: a question-relevant crop and a negative view gate the token-level distillation. |

#### Auxiliary objectives

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2026-09 | [Is Better Teacher Supervision Enough? Unlocking Student-side Learning in Multimodal On-Policy Distillation](https://arxiv.org/abs/2609.39120)<br>Siyuan Liu et al. · `2609.39120` | arXiv | [GitHub](https://github.com/Sirilaw/S-OPD) | **S-OPD**: Strengthens student-side perception: contrasts the student under the original and a masked image with teacher-based token gating, and aligns it under noise-perturbed images. |

### 📏 Benchmarks and datasets

| Date | Paper | Venue | Code | Key idea |
|---|---|---|---|---|
| 2025-04 | [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837)<br>Haozhe Wang et al. · `2504.08837` | NeurIPS 2025 | [GitHub](https://github.com/TIGER-AI-Lab/VL-Rethinker) | **VL-Rethinker / ViRL39K**: GRPO with Selective Sample Replay and Forced Rethinking; releases the ViRL39K RL training set. |
| 2025-03 | [MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning](https://arxiv.org/abs/2503.07365)<br>Fanqing Meng et al. · `2503.07365` | arXiv | [GitHub](https://github.com/ModalMinds/MM-EUREKA) | **MM-Eureka / MMK12**: Introduces MMK12, a multimodal K12 math dataset with human-verified answers, plus rule-based RL models. |
| 2024-10 | [DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models](https://arxiv.org/abs/2411.00836)<br>Chengke Zou et al. · `2411.00836` | ICLR 2025 | [GitHub](https://github.com/DynaMath/DynaMath) | **DynaMath**: Dynamic visual math benchmark generating question variants to test reasoning robustness. |
| 2024-09 | [MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2409.02813)<br>Xiang Yue et al. · `2409.02813` | ACL 2025 | [GitHub](https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu-pro) | **MMMU-Pro**: Harder MMMU: filters text-answerable questions, augments options, adds a vision-only (question-in-image) setting. |
| 2024-08 | [Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models](https://arxiv.org/abs/2408.15556)<br>Wenbin Wang et al. · `2408.15556` | AAAI 2025 | [GitHub](https://github.com/DreamMr/HR-Bench) | **HR-Bench**: Benchmark of 4K and 8K images for high-resolution perception; also proposes training-free DC². |
| 2024-08 | [MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?](https://arxiv.org/abs/2408.13257)<br>Yi-Fan Zhang et al. · `2408.13257` | ICLR 2025 | [GitHub](https://github.com/MME-Benchmarks/MME-RealWorld) | **MME-RealWorld**: 29,429 human-annotated QA pairs over 43 subtasks on high-resolution real-world images. |
| 2024-07 | [LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts](https://arxiv.org/abs/2407.04973)<br>Yijia Xiao et al. · `2407.04973` | arXiv | [GitHub](https://github.com/Yijia-Xiao/LogicVista) | **LogicVista**: 448 multiple-choice questions evaluating logical reasoning in visual contexts across 5 tasks. |
| 2024-07 | [We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](https://arxiv.org/abs/2407.01284)<br>Runqi Qiao et al. · `2407.01284` | ACL 2025 | [GitHub](https://github.com/We-Math/We-Math) | **We-Math**: 6.5K visual math problems over 67 knowledge concepts with a four-dimensional diagnostic metric. |
| 2024-03 | [Are We on the Right Way for Evaluating Large Vision-Language Models?](https://arxiv.org/abs/2403.20330)<br>Lin Chen et al. · `2403.20330` | NeurIPS 2024 | [GitHub](https://github.com/MMStar-Benchmark/MMStar) | **MMStar**: 1,500 human-selected vision-indispensable samples minimizing visual-free answering and data leakage. |
| 2024-03 | [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://arxiv.org/abs/2403.14624)<br>Renrui Zhang et al. · `2403.14624` | ECCV 2024 | [GitHub](https://github.com/ZrrSkywalker/MathVerse) | **MathVerse**: 2,612 diagram math problems, each in six versions varying text vs. diagram information (15K test samples). |
| 2024-02 | [Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset](https://arxiv.org/abs/2402.14804)<br>Ke Wang et al. · `2402.14804` | NeurIPS 2024 (Datasets and Benchmarks) | [GitHub](https://github.com/mathllm/MATH-V) | **MATH-Vision**: 3,040 competition math problems with visual contexts across 16 disciplines and 5 difficulty levels. |
| 2023-12 | [V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135)<br>Penghao Wu et al. · `2312.14135` | CVPR 2024 | [GitHub](https://github.com/penghao-wu/vstar) | **V\* / V\*Bench**: Introduces V\*Bench for fine visual detail search in high-resolution images, alongside the SEAL visual-search MLLM. |
| 2023-11 | [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502)<br>Xiang Yue et al. · `2311.16502` | CVPR 2024 | [GitHub](https://github.com/MMMU-Benchmark/MMMU) | **MMMU**: 11.5K college-level multimodal questions across 6 disciplines and 30 subjects. |
| 2023-10 | [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://arxiv.org/abs/2310.14566)<br>Tianrui Guan et al. · `2310.14566` | CVPR 2024 | [GitHub](https://github.com/tianyi-lab/HallusionBench) | **HallusionBench**: 346 images with 1,129 expert-written questions diagnosing language hallucination and visual illusion. |
| 2023-10 | [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://arxiv.org/abs/2310.02255)<br>Pan Lu et al. · `2310.02255` | ICLR 2024 | [GitHub](https://github.com/lupantech/MathVista) | **MathVista**: 6,141 examples from 28 existing datasets plus three new ones (IQTest, FunctionQA, PaperQA). |
| 2023-05 | [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)<br>Yifan Li et al. · `2305.10355` | EMNLP 2023 | [GitHub](https://github.com/RUCAIBox/POPE) | **POPE**: Polling-based object probing evaluation for object hallucination in LVLMs. |
| 2021-05 | [Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning](https://arxiv.org/abs/2105.04165)<br>Pan Lu et al. · `2105.04165` | ACL 2021 | [GitHub](https://github.com/lupantech/InterGPS) | **Inter-GPS / Geometry3K**: Introduces Geometry3K (3,002 geometry problems with formal-language annotations) and the Inter-GPS solver. |

## 🤝 Contributing

Contributions are welcome:

- **Add a paper** to the list: open a pull request that adds a row (newest first) with the arXiv
  link, venue and code link to both READMEs, then run `python scripts/check_docs.py --fix`.
- **Add a method**: follow [docs/add_method.md](docs/add_method.md) and add a
  `examples/reproduction/<method>/` directory with a README and baseline scripts.
- **Add a benchmark**: follow [eval/README.md](eval/README.md#adding-a-new-benchmark).
- **Report reproduction results**: open an issue with the "Reproduction results" form (the command,
  the logs and the evaluation summary).

Before opening a pull request, set up the checks once with `pip install pre-commit && pre-commit install`,
then run `pre-commit run --all-files` and `make test`. The [contributing guide](.github/CONTRIBUTING.md)
([中文](.github/CONTRIBUTING_zh.md)) explains each check, the commit message format and the pull request steps.

## 📝 Citation

If you find this repository useful, please cite CGPO and the papers of the methods you use:

```bibtex
@inproceedings{zhang2026cgpo,
  title     = {{CGPO}: Counterfactual Grounding Policy Optimization for Evidence-Sensitive Pathology Vision-Language Reasoning},
  author    = {Zhang, Shengxuming and Zhou, Linyun and Lou, Hengrui and Wang, Zhenyang and Zhang, Xiuming and Feng, Zunlei},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  location  = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {10},
  doi       = {10.1145/3767308.3835969},
  url       = {https://doi.org/10.1145/3767308.3835969}
}
```

BibTeX entries of the reproduced methods are in their READMEs and in [docs/references.bib](docs/references.bib).

## 🙏 Acknowledgements

This repository is built on [EasyR1](https://github.com/hiyouga/EasyR1) and
[veRL](https://github.com/volcengine/verl), and uses [vLLM](https://github.com/vllm-project/vllm)
for rollout and evaluation. We thank the authors of all reproduced methods for releasing their
papers, code and data; PAPO, VPPO, GRIT and DeepEyes in particular for their public datasets and
evaluation sets.

## 📄 License

Apache-2.0, see [LICENSE](LICENSE). Datasets and models downloaded by the scripts are subject to
their own licenses.
