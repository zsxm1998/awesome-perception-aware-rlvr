# VAPO — More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models

ICLR 2026 · [arXiv:2509.25848](https://arxiv.org/abs/2509.25848) ·
[official code](https://github.com/xytian1008/VAPO) ·
**Unofficial re-implementation in this repository.**

## Method

VAPO (Vision-Anchored Policy Optimization) observes that long reasoning makes vision-language models
forget the image, and rewards responses that keep perceiving it. Every training image comes with 20
claims about its content, generated offline by GPT-5 without the question, half of them true and half
false. After the GRPO rollout, each response is cut at K = 20 random punctuation marks of its reasoning
(*visual anchors*). At each cut the policy reads the prompt, the response up to the cut and one claim,
asked as `<anchor>{claim} Is this claim correct? Answer (Yes/No): `, and answers with one greedy token;
the anchor is right when the answer matches the claim's label (Eq. 3). The perception score is the rate
of right answers weighted toward later cuts (Eq. 4), and it is added to the reward of correct answers
only (Eq. 5), so that the policy cannot gain it by giving up on the problem. The update itself is GRPO's.

In this repository the anchors are *claim probes* (`algorithm.claim_probe_count`, K): they run on the
rollout engine right after generation, one token per probe, and need no extra training forward pass.
The perception score of a response is

R_perc = max(0, (Σ_k w_k s_k / Σ_k w_k − 0.5) / 0.5), w_k = exp(β a_k / T),

with s_k ∈ {0, 1} the probe answers, a_k the character position of the cut, T the number of characters
of the response and β = 1.5 (`algorithm.claim_probe_late_emphasis`); chance (0.5) maps to 0. The reward
is 0.8 · accuracy + 0.1 · format + 0.1 · 1[accuracy = 1] · R_perc
(`worker.reward.reward_function_kwargs.perception_weight=0.1` of `examples/reward_function/math.py`).

## Setting

| | This repository |
| --- | --- |
| Base model | Qwen2.5-VL-7B-Instruct (3B and Qwen3-VL-8B scripts as well) |
| Training data | `xytian1008/VAPO-Thinker-train36k` (36,581 problems with 20 claims each) — `bash scripts/prepare_data.sh vapo`. These are exactly the single-image problems of ViRL39K as processed by PAPO, in the same order: we checked all 36,581 rows (problem, answer and image bytes) |
| Validation | `xytian1008/VAPO-Thinker-val1k` (1,000 problems sampled from the evaluation benchmarks; monitoring only), greedy, every 3 steps and before training |
| Prompt / reward | `<think>` + `\boxed{}` (`examples/format_prompt/math_perception.jinja`, the text of the official `vapo.jinja`), Qwen's default system prompt; 0.8 accuracy + 0.1 format + 0.1 gated perception (`math.py:compute_score`, `perception_weight=0.1`) |
| Rollout | 384 prompts x 5 rollouts per step, T=1.0, top-p 0.99; vLLM tensor parallel 1 |
| RL | GRPO: low-variance KL loss 1e-2, clip 0.2 / 0.2, token-level loss, no entropy term, no dynamic sampling; update batch 128 prompts (3 updates per step) |
| Optimization | AdamW, lr 5e-6 constant, weight decay 1e-2, gradient clipping 1.0, vision tower trainable, 2 epochs (about 190 steps); data seed 1 |
| Lengths / pixels | max prompt 4,096, max response 2,048; 262,144-4,194,304 pixels |
| GPUs | 8 |
| Claim probes | `algorithm.claim_probe_count=20` (K), `algorithm.claim_probe_late_emphasis=1.5` (β); `worker.rollout.mm_processor_cache_gb=4` so that the n x K probes of a prompt reuse its processed image |

See [docs/algorithm_parameters.md](../../../docs/algorithm_parameters.md#visual-claim-probes) for the
switches. The probes follow the released code (with one exception, `<answer>`, listed under the differences
below): the reasoning is the text before `</think>` (else before `<answer>`, else before `\boxed`, else the whole
response); a cut follows one of `, . \n ! ? ; :` at least two characters after
the previous cut; K cuts are drawn without replacement (with fewer cuts, all are used and the last one
repeated; with none, every probe reads the whole response); each probe asks a different claim; a
response whose prompt and response exceed prompt + response length − 144 tokens is not cut in its last
128 reasoning characters; the probe text is the decoded prompt, the response up to and including the cut
character and the question (`algorithm.claim_probe_question`), encoded again; the answer is the greedy token
among the candidates (`algorithm.claim_probe_yes_tokens` / `claim_probe_no_tokens`) `yes Yes True true 1`
(yes) and `0 no No false False` (no). The rollout engine gets 128 more tokens of context for the
question (the longest in the data has 77). On random responses without `<answer>`, our cut positions,
claim choices, perception scores and rewards equal those of the released functions.

Training logs `claim_probe/score` (mean R_perc of the probed responses), `claim_probe/accuracy` (and
`accuracy_early` / `accuracy_late` for cuts before / after the middle of the response), `yes_rate`,
`fallback_rate` (responses without a cut), `probed_fraction`, `max_prompt_tokens`, the time
`timing_s/claim_probe`, and `reward/perception` (0 for responses that were not probed).

## Scripts

| Script | Run |
| --- | --- |
| `qwen2_5_vl_7b_grpo.sh` | GRPO baseline, 7B (the paper's K = 0) |
| `qwen2_5_vl_7b_vapo.sh` | VAPO, 7B |
| `qwen2_5_vl_3b_grpo.sh` | GRPO baseline, 3B |
| `qwen2_5_vl_3b_vapo.sh` | VAPO, 3B |
| `qwen3_vl_8b_grpo.sh` | GRPO baseline, Qwen3-VL-8B |
| `qwen3_vl_8b_vapo.sh` | VAPO, Qwen3-VL-8B |

```bash
bash scripts/prepare_data.sh vapo
bash examples/reproduction/vapo/qwen2_5_vl_7b_vapo.sh
bash scripts/prepare_eval_data.sh vapo
bash scripts/eval.sh checkpoints/VAPO-Reproduce/qwen2_5_vl_7b_vapo --suite vapo
# greedy decoding, as the paper
bash scripts/eval.sh checkpoints/VAPO-Reproduce/qwen2_5_vl_7b_vapo --suite vapo --temperature 0 --num-samples 1
```

Checkpoints go to `checkpoints/VAPO-Reproduce/<script name>`. Append `key=value` overrides to the
command and set `N_GPUS_PER_NODE`, `MODEL_PATH`, `DATA_ROOT`, `LOGGER` or `EXPERIMENT_NAME` in the
environment. Evaluation is described in [eval/README.md](../../../eval/README.md). The controlled
comparison runs VAPO as [`examples/comparison/qwen3_vl_4b/vapo.sh`](../../comparison/README.md), on
ViRL39K with the claims added as a column (`bash scripts/prepare_data.sh comparison` builds
`virl39k_claims`); the 2,289 multi-image problems have no claims and get the GRPO reward.

## Differences from the paper / official code

The official code is licensed under Apache-2.0 (its LICENSE file; the README says MIT). The probes are
written in this repository and follow the released functions step by step (see above).

- **Recipe.** We follow the paper (Sec. 5, App. A.7): GRPO with a KL penalty of 1e-2, clip 0.2 and 2
  epochs. The released 7B script trains with clip-higher (0.2 / 0.28), without KL, with dynamic sampling
  on the overall reward and for 5 epochs. The paper's training times (Table 11: VAPO 19h14m, GRPO
  18h46m, DAPO 25h11m) and its K = 0 ablation (Table 29, equal to its GRPO baseline) agree with the paper's
  recipe. Settings the paper does not give follow the released script (batch sizes, lengths, pixels,
  top-p 0.99, optimizer, validation).
- **Reward.** As the released reward: 0.8 · accuracy + 0.1 · format + 0.1 · 1[accuracy = 1] · R_perc,
  with R_perc mapped by (x − 0.5) / 0.5 and clipped at 0, and positions counted in characters of the
  decoded response. Eq. 4–5 write accuracy + format + 0.1 · R_perc with the plain weighted rate and
  token positions.
- **Probed responses.** The probes run on responses with accuracy 1 only. The perception score enters
  the reward only for these, so the reward of every response is the same as with probes on all
  responses (0.1 · format for wrong answers either way); the probes of wrong answers are skipped.
- **Probe question.** The reproduction scripts keep the released question and candidates; the controlled
  comparison drops the trailing space of the question (see [Probe question](#probe-question)).
- **`<answer>`.** The released code ends the reasoning at `</think>`, else at `\boxed`. We also end it at
  `<answer>` (after `</think>`, before `\boxed`), so that formats with an `<answer>` tag never cut inside the
  answer when `</think>` is missing. This does not change the probes of the VAPO prompt, which has no
  `<answer>` tag.
- **Randomness.** The released code draws cuts and claims from an unseeded generator; here each
  response's probes are seeded by `data.seed`, the step and the response's row in the step's batch.
- **3B.** The released 3B script starts from `xytian1008/Qwen2.5-VL-3B_VISOR_39K` with the data
  `xytian1008/VISOR_train_39k` (K = 5, lr 1e-5), neither of which is public. Our 3B scripts start from
  Qwen2.5-VL-3B-Instruct with the 7B recipe, as the paper states its setting for both sizes.
- **Qwen3-VL-8B.** Table 16 reports GRPO and VAPO on Qwen3-VL-8B without a recipe; the scripts reuse
  the main recipe.
- **Saving.** Every 5 steps, keeping the last checkpoint (the released script saves every 20 steps and
  keeps all); we evaluate the last step.
- The ablations (K, β, γ, claim sources, anchor placement) are not scripted.
- **Evaluation** (suite `vapo`). The paper uses VLMEvalKit with greedy decoding and a GPT model
  extracting the answers. Our math benchmarks use the rule-based PAPO-Eval protocol (avg@8 at T=1.0 by
  default, `--temperature 0 --num-samples 1` for greedy) and the other four are greedy. The paper's
  MathVerse is the testmini Vision Only split (about 700 items); the closest set here is `mathverse_v`
  (1,308 vision-dependent multiple-choice items, Vision Only included). The paper uses the full 3,040-item
  MathVision test (here VPPO-Eval's 2,907 verifiable items), We-Math's strict score (here per-question
  accuracy) and Geometry3K val + test (here the PAPO-Eval test split). MM-Vet needs a judge model. V*
  and BLINK of the appendix tables are available as benchmarks (`--benchmarks vstar,blink`);
  RealWorldQA and MuirBench are not.

### Probe question

The released question ends with a space (`Answer (Yes/No): `). Qwen's tokenizer encodes that space as a token
of its own, and after it the digit candidates `1` / `0` win over `Yes` / `No` (which follow a space as the
token ` Yes`, not a candidate). We measured the probes of the first training step, with the same rollouts for
every row of a column, for three readings (the question and the candidates are options):

| Reading | Qwen3-VL-4B, comparison setting, data seeds 1 / 2 / 3 | Qwen2.5-VL-7B, reproduction setting, seed 1 |
| --- | --- | --- |
| Released: `": "`, candidates `yes Yes True true 1` / `0 no No false False` | 58.3 / 57.0 / 58.5% right, 89–92% "yes"; seed 1: every answer `1` or `0` | 57.8% right, 77% "yes"; every answer `1` or `0` |
| No trailing space: `":"`, the released candidates | 77.5 / 75.7 / 73.9% right, 53–58% "yes"; seed 1: answers `Yes` / `No` | 60.9% right, 23% "yes" |
| Released question, candidates `Yes` / `No` only | 72.6 / 72.2 / 70.9% right, 74–76% "yes" | 68.8% right, 41% "yes" |

About 54% of the claims are correct, so answering "yes" to every probe would be right on about 54% of them.
On seed 1 of the 4B model, the mean perception score of the probed responses is 0.19 with the released reading
and 0.56 without the trailing space. These are untrained models; what the policy answers changes during training.

The reproduction scripts keep the released reading. The comparison (`examples/comparison/qwen3_vl_4b/vapo.sh`)
uses the question without the trailing space, the best of the three on its model across the three data seeds:

```bash
'algorithm.claim_probe_question="\n<anchor>{claim} Is this claim correct? Answer (Yes/No):"'
```

### Dynamic sampling

The reproduction and the comparison do not use dynamic sampling, so the following only matters when
`algorithm.online_filtering=true` is added to a VAPO run.

1. The probes run after online filtering, on the correct responses of the kept groups, so no probe is
   spent on groups that are dropped. The filter therefore cannot use the perception score:
   `algorithm.filter_key` must be `accuracy` or `format`, and `overall` is rejected when the
   configuration is built.
2. The reason is that, at filtering time, `overall` would be the reward without the perception score,
   which is not the reward the policy is trained on, and its meaning changes: a group whose responses
   are all correct and well formatted has an overall reward of 1.0 without the perception score and is
   always dropped, while the released script filters on the overall reward with the perception score,
   which drops such a group only when its mean perception score is at least 0.9.
3. As a consequence, with dynamic sampling on accuracy the groups that are all correct are dropped, and
   the perception reward acts only in groups with both correct and wrong answers.
4. Filtering on the overall reward with the perception score would need the probes before the filter:
   probe the correct responses of every generated group, then filter. Groups without a correct answer
   need no probes either way; the extra cost is the probes of the all-correct groups that are dropped
   and of the groups beyond the batch size in the last round.
5. The scripts of this repository are not affected.

## Results

> [!NOTE]
> Columns marked as the paper's are copied from the original publication. Columns of this repository
> come from our unofficial re-implementation, trained once with a single seed on our hardware and
> software versions and evaluated with this repository's harness. They may differ from the official
> results, including smaller or no gains over the baselines, and do not replace them; corrections
> from the authors are welcome. See [About the results](../../../README.md#-about-the-results).

Paper Tables 28 and 29 (greedy decoding, %; MM-Vet is a judged score). The 7B GRPO row is the paper's
K = 0 ablation; the paper's 3B model starts from a non-public SFT model (see above).

| Script | MathVerse | MathVista | MathVision | LogicVista | We-Math | Geo3K | MMMU | MMStar | HallBench | MM-Vet | Avg | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | 40.7 | 62.3 | 23.2 | 42.6 | 33.1 | 38.5 | 52.7 | 54.9 | 50.0 | 64.8 | 46.3 | TBD |
| `qwen2_5_vl_7b_grpo` | 48.2 | 70.6 | 26.1 | 45.5 | 39.1 | 47.3 | 56.6 | 58.9 | 53.2 | 69.9 | 51.5 | TBD |
| `qwen2_5_vl_7b_vapo` | 53.3 | 75.6 | 31.9 | 50.9 | 43.6 | 51.3 | 60.2 | 63.0 | 57.4 | 71.9 | 55.9 | TBD |
| `qwen2_5_vl_3b_grpo` | – | – | – | – | – | – | – | – | – | – | – | TBD |
| `qwen2_5_vl_3b_vapo` | 35.8 | 67.1 | 23.9 | 39.7 | 35.4 | 44.2 | 55.6 | 59.4 | 49.5 | 64.6 | 47.5 | TBD |

Paper Table 16 (Qwen3-VL-8B, six benchmarks):

| Script | MathVerse | LogicVista | Geo3K | MMMU | MMStar | HallBench | Avg | This repo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-8B | 58.8 | 56.7 | 59.3 | 68.4 | 67.3 | 59.2 | 61.8 | TBD |
| `qwen3_vl_8b_grpo` | 63.5 | 60.1 | 64.4 | 72.7 | 68.3 | 60.4 | 64.9 | TBD |
| `qwen3_vl_8b_vapo` | 67.6 | 65.8 | 68.7 | 74.2 | 72.5 | 63.9 | 68.8 | TBD |

Because the evaluation protocol differs, compare our runs with each other rather than with these
numbers. Results of this repository will be added after the reproduction runs finish.

## Citation

```bibtex
@inproceedings{tian2026more,
  title     = {More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models},
  author    = {Tian, Xinyu and Zou, Shu and Yang, Zhaoyuan and He, Mengqi and Waschkowski, Fabian and Wesemann, Lukas and Tu, Peter and Zhang, Jing},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
