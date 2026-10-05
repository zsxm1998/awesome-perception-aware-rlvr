# NoisyRollout — NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation

NeurIPS 2025 · [arXiv:2504.13055](https://arxiv.org/abs/2504.13055) ·
[official code](https://github.com/real-absolute-AI/NoisyRollout) ·
**Unofficial re-implementation in this repository.**

## Method

NoisyRollout changes only how GRPO collects rollouts. For every prompt, half of the rollouts are
sampled from the clean images and half from a noised copy of them; all of them form one GRPO group,
so the advantages of the noised rollouts are normalized together with the clean ones. The policy is
always updated on the clean images: the noised images only reach the rollout engine, and the old,
reference and current log-probabilities are all conditioned on the clean images, with no importance
correction for the noised sampling (Eq. 2; the released code does the same).

The noise is a step of a 1,000-step variance-preserving forward diffusion,
x_k = √ᾱ_k · x + √(1 − ᾱ_k) · ε on the [0, 1] pixels of the resized image, clipped and converted back
to 8 bits. The step k anneals with a sigmoid over training (Eq. 3):
k_t = ⌊α₀ · (1 − σ(λ (t − γ) / t_max))⌋, so the early steps use strong noise (k ≈ α₀, a noise std of
about 0.41–0.50 in pixel units for α₀ = 450–500) and the later steps almost none. Each prompt gets one
noised image per step, shared by its noised rollouts.

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct (Qwen2.5-VL-32B-Instruct for the `32b` scripts) |
| Training data | Geometry3K train (2,101; `bash scripts/prepare_data.sh noisyrollout`) or K12 (`xyliu6/k12-freeform` train, 6,457, an early subset of MMK12 in free-form) |
| Validation | the matching test split (Geometry3K 601, K12 808), greedy, every 5 steps |
| Prompt / reward | the official system prompt (`examples/system_prompt/noisyrollout.txt`: `<think>` + `\boxed{}`), the problem as the user message; 0.9 accuracy + 0.1 format (`math.py:compute_score`) |
| Rollout | 512 prompts per step, 12 rollouts per prompt (6 clean + 6 noised for NoisyRollout; 8 = 4 + 4 for 32B), T=1.0, top-p 1.0; vLLM tensor parallel 4 |
| RL | GRPO without KL; clip 0.2 / 0.2; entropy bonus 1e-3 on the full-vocabulary entropy (`algorithm.invariant_entropy_coef=-0.001`, `algorithm.entropy_loss_type=full`); token-level loss; update batch 128 prompts (4 updates per step) |
| Optimization | AdamW, lr 1e-6 constant, weight decay 1e-2, vision tower frozen; data seed 1 |
| Lengths / pixels | max prompt 2,048, max response 2,048; 262,144-1,000,000 pixels |
| Noise (7B) | Geometry3K: α₀ = 500, λ = 30, γ = 40 of t_max = 60 steps (15 epochs); K12: α₀ = 450, λ = 60, γ = 40 of t_max = 120 steps (10 epochs) |
| Noise (32B, Table 13) | Geometry3K: α₀ = 450, λ = 30, γ = 35 of 40 steps (10 epochs); K12: α₀ = 450, λ = 30, γ = 35 of 70 steps |
| Switches | `algorithm.rollout_image_transform=vp_diffusion`, `algorithm.rollout_image_transform_kwargs={"noise_t_init":α₀,"noise_gamma":λ,"noise_t_mid":γ/t_max,"noise_t_max":1000,"pixel_rounding":"floor"}`; `worker.rollout.n` is the total per prompt and must be even |
| GPUs | 8 |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#rollouts-from-transformed-images)
for the switches. Training logs `rollout_transform/noise_t` and `rollout_transform/noise_std` (the
current step and its noise std) and the accuracy of the clean and of the noised rollouts
(`rollout_transform/accuracy_clean`, `rollout_transform/accuracy_transformed`).

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_geo3k_grpo.sh` | GRPO baseline (12 rollouts), Geometry3K, 7B |
| `qwen2_5_vl_7b_geo3k_noisyrollout.sh` | NoisyRollout (6 + 6), Geometry3K, 7B |
| `qwen2_5_vl_7b_k12_grpo.sh` | GRPO baseline (12 rollouts), K12, 7B |
| `qwen2_5_vl_7b_k12_noisyrollout.sh` | NoisyRollout (6 + 6), K12, 7B |
| `qwen2_5_vl_32b_geo3k_grpo.sh` | GRPO baseline (8 rollouts), Geometry3K, 32B |
| `qwen2_5_vl_32b_geo3k_noisyrollout.sh` | NoisyRollout (4 + 4), Geometry3K, 32B |
| `qwen2_5_vl_32b_k12_grpo.sh` | GRPO baseline (8 rollouts), K12, 32B |
| `qwen2_5_vl_32b_k12_noisyrollout.sh` | NoisyRollout (4 + 4), K12, 32B |

```bash
bash scripts/prepare_data.sh noisyrollout
bash examples/reproduction/noisyrollout/qwen2_5_vl_7b_geo3k_noisyrollout.sh
bash scripts/prepare_eval_data.sh noisyrollout
bash scripts/eval.sh checkpoints/NoisyRollout-Reproduce/qwen2_5_vl_7b_geo3k_noisyrollout --suite noisyrollout
# greedy decoding, as the paper (which parses four of the benchmarks with Gemini-2.0-Flash)
bash scripts/eval.sh checkpoints/NoisyRollout-Reproduce/qwen2_5_vl_7b_geo3k_noisyrollout --suite noisyrollout --temperature 0 --num-samples 1
```

Checkpoints go to `checkpoints/NoisyRollout-Reproduce/<script name>`. Append `key=value` overrides to
the command and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in
the environment. Evaluation is described in [eval/README.md](../../../eval/README.md).

## Differences from the paper / official code

- **Rollout count.** `worker.rollout.n` is the number of rollouts per prompt in total, of which the
  second half comes from the noised images; the official `worker.rollout.n=6` with `is_noisy=true`
  means 6 clean + 6 noised, which is `worker.rollout.n=12` here. Unequal splits (Table 12) are not
  supported, as in the released code.
- **Reward and entropy.** As the official scripts, the reward is 0.9 accuracy + 0.1 format and the
  loss has an entropy bonus of 1e-3; the paper describes a 0/1 reward and no entropy term.
- **Loss aggregation.** The loss is averaged over all response tokens of an update batch (the
  token-mean of the paper, Table 13). The released code averages over the tokens of each micro-batch
  of 2 sequences and then over micro-batches, which depends on the micro-batch size.
- **Randomness.** The noise is drawn from a generator seeded by the data seed, the step and the
  prompt's position in the batch, so a run is reproducible; the released code uses the unseeded
  global torch generator. Noise strength, clipping and the 8-bit truncation (`pixel_rounding=floor`)
  follow the released code, and the annealed step matches it at every step.
- **K12 GRPO baseline.** `qwen2_5_vl_7b_k12_grpo.sh` trains on the 6,457-problem train split for 10
  epochs, the 6.4K MMK12 baseline of Table 1. The released `k12_grpo.sh` trains on `mini_train`
  (2,100) for 15 epochs, which matches the 2.1K setting of App. A (Table 11).
- **Geometry3K data.** We use `hiyouga/geometry3k`, which holds the same 2,101 train and 601 test
  problems as the authors' `xyliu6/geometry3k` in a different row order, so a data seed shuffles
  them differently.
- **32B.** The official repository has no 32B scripts; the `32b` scripts follow Table 13. Table 13
  lists t_max = 70 steps with 7 epochs of K12 (84 steps at 12 steps per epoch), so the K12 32B runs
  stop at step 70 (`trainer.max_steps=70`). The memory settings of these scripts (vLLM tensor
  parallel 8, micro-batches of 1 and 2) are ours and have not been run.
- **No reference model.** The released code computes reference log-probabilities with a zero KL
  coefficient; we skip them (`algorithm.disable_kl=true`). As every script in this repository, the
  policy loss keeps dual-clip PPO (C = 3), which only acts on negative advantages with a ratio above 3.
- Rotation and the other augmentations of App. A, the ablations (Tables 3-12) and the 2.1K K12
  setting are not scripted.
- **Evaluation** (suite `noisyrollout`) uses the PAPO-Eval protocol (rule-based `\boxed{}` match,
  avg@8, T=1.0) instead of greedy decoding with Gemini-2.0-Flash answer parsing; our MathVision is
  VPPO-Eval's 2,907 verifiable items instead of the full 3,040, and our HallusionBench prompt does not
  ask for a `\boxed{yes}` / `\boxed{no}` answer. Use `--temperature 0 --num-samples 1` for greedy
  decoding.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Tables 1 and 2 (greedy accuracy, %; HallusionBench with the reasoning prompt).

| Script | MathVerse | MathVision | MathVista | We-Math | HallusionBench | Paper (mean of 5) | This repo (Avg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_vl_7b_geo3k_grpo` | 50.8 | 27.3 | 70.5 | 67.4 | 69.8 | 57.2 | TBD |
| `qwen2_5_vl_7b_geo3k_noisyrollout` | 53.2 | 28.5 | 72.6 | 69.6 | 72.1 | 59.2 | TBD |
| `qwen2_5_vl_7b_k12_grpo` | 51.8 | 29.4 | 73.2 | 70.2 | 70.3 | 59.0 | TBD |
| `qwen2_5_vl_7b_k12_noisyrollout` | 53.0 | 30.6 | 74.5 | 70.3 | 72.2 | 60.1 | TBD |
| `qwen2_5_vl_32b_geo3k_grpo` | 58.9 | 39.2 | 77.0 | 76.1 | 72.3 | 64.7 | TBD |
| `qwen2_5_vl_32b_geo3k_noisyrollout` | 58.9 | 39.9 | 77.8 | 77.2 | 73.5 | 65.5 | TBD |
| `qwen2_5_vl_32b_k12_grpo` | 58.9 | 40.0 | 76.7 | 76.9 | 72.1 | 64.9 | TBD |
| `qwen2_5_vl_32b_k12_noisyrollout` | 59.3 | 41.6 | 77.4 | 77.6 | 73.2 | 65.8 | TBD |

The paper reports Geometry3K test accuracy (in-domain) for the 7B Geometry3K runs: 51.4 or 52.0 for
GRPO (depending on the table) and 54.9 for NoisyRollout. Because the evaluation protocol differs,
compare our runs with each other rather than with these numbers. Results of this repository will be
added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{liu2025noisyrollout,
  title     = {NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation},
  author    = {Liu, Xiangyan and Ni, Jinjie and Wu, Zijian and Du, Chao and Dou, Longxu and Wang, Haonan and Pang, Tianyu and Shieh, Michael Qizhe},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
