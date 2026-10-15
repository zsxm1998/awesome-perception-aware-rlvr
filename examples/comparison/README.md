# Controlled comparisons

[English](README.md) | [简体中文](README_zh.md)

The papers reproduced in [examples/reproduction/](../reproduction/) use different backbones, training sets,
baseline definitions and evaluation protocols, and several of them reuse baseline numbers from other papers, so
their reported gains are hard to compare with each other. Each comparison below trains one family of methods
under **one** setting: the same model, data, hyper-parameters and evaluation. In each directory a script only
sets `ALGO_ARGS`, the switches that define its method; everything else comes from the directory's `common.sh`.

| Comparison | Methods | Shared setting | Results |
| --- | --- | --- | --- |
| [RLVR on Qwen3-VL-4B-Instruct](rlvr_qwen3_vl_4b/README.md) | GRPO and DAPO baselines; PAPO, VPPO, ToR, DVRP, PGPO, PEPO, CFPO, VEPO, NoisyRollout, VGPO, VAPO, GRIT, CGPO | ViRL39K, 202 steps of 384 prompts × 8 responses, GRPO hyper-parameters, 4 A800 GPUs | 19 benchmarks in three groups, at the last and at the best validation step; training time |
| [OPD on Qwen3-VL-2B-Instruct](opd_qwen3_vl_2b/README.md) | GRPO reference; OPD from sampled tokens, OPD on the full distributions, VA-OPD, VGS, VCSD, S-OPD | the data, prompt and lengths of the RLVR comparison; a GRPO-trained Qwen3-VL-4B-Instruct as the teacher; 606 updates of 1,024 responses, 4 A800 GPUs | the same 19 benchmarks, at the last and at the best validation step; training time |

DeepEyes (RLVR) and Vision-OPD (OPD) are not in these comparisons; the README of each comparison explains why.

## About the results

> [!IMPORTANT]
> **The results are a reference, not a verdict on the methods.** The numbers produced with this repository
> (the tables of the comparisons above and the "this repo" columns of the reproduction READMEs) come with
> these limits.

- **Re-implementations.** Except for CGPO, the methods are our re-implementations of the papers and, where
  available, the official code. Details a paper leaves open may differ from the authors' code; the known
  differences are listed in each method's README.
- **A different environment.** Hardware, software versions (PyTorch, vLLM, transformers, the RL framework) and
  data preprocessing differ from those of the original papers.
- **One run per method.** Training is not deterministic (sampling, kernels, hardware), and each configuration
  is trained once. Two runs of CFPO in the RLVR comparison with the same configuration (the second with a faster
  implementation of its counterfactual attention that differs only by rounding) ended 0.6 points apart in the
  mean of the 3 groups and up to 2.6 points apart on single benchmarks, in both directions, while two
  checkpoints of one run five steps apart differ by about 0.15 points. Differences of about a point between
  methods are within this variation, and many reported gains are of this size.
- **A shared setting that is not the methods' own.** One backbone, one dataset, one training budget, one set
  of hyper-parameters and one prompt are used for every method of a comparison. Most methods were designed and
  tuned for another setting (a different model, data, base algorithm, response length or schedule), and some of
  their components are left out (see the Methods section of each comparison); a method may not reach its best
  performance in this setting.
- **Our evaluation protocol.** All checkpoints are evaluated with this repository's harness and prompts,
  which differ from the papers' evaluations.

The tables describe how our re-implementations behave in these settings; they do not rank the methods and are
not evidence for or against any of them. The original papers remain the reference for each method's
performance. If you are an author and see a problem with our implementation or setting, please open an issue
or a pull request; we will correct it.
