# Implementation Notes

Behaviors of the shared implementation that several methods depend on. The per-method READMEs in
[`examples/reproduction/`](../examples/reproduction/) state where a method's paper or code differs
from them; parameter names are listed in [algorithm_parameters.md](algorithm_parameters.md).

## The k3 estimator used as a loss

PAPO (KL_prcp), DVRP (λ_nec and λ_rob) and CFPO (KL_cf) put a KL term between the policy on the
original input, p = π_θ(· | s), and on an auxiliary view, q (masked, noised or counterfactual), into
the loss. As in PAPO's and CFPO's code, the KL is estimated at the sampled token y with k3:

    r = q(y) / p(y),    k3 = r − log r − 1.

The auxiliary view is scored before the update, so q carries no gradient. For y ∼ p (one on-policy
update) the value is an unbiased estimate of KL(p ‖ q). Its gradient is not that of KL(p ‖ q):

    ∂k3 / ∂log p(y) = 1 − q(y) / p(y)

    E_{y∼p}[(1 − q/p) ∇log p] = Σ_y p ∇log p − Σ_y q ∇log p
                             = ∇Σ_y p − ∇Σ_y q log p
                             = 0 + ∇KL(q ‖ p),

using Σ_y p = 1 and that the entropy of q does not depend on θ. So the estimate reports KL(p ‖ q),
while minimizing or maximizing it moves the policy along the gradient of the reverse divergence
KL(q ‖ p). A numerical check on random categorical distributions gives an expected k3 gradient equal
to ∇KL(q ‖ p) and different from ∇KL(p ‖ q). Two more details: within PPO mini-batches the tokens
come from the rollout policy, so the expectation holds only approximately after the first update;
and k3 is clipped (to [−10, 10] with the current policy as reference, to [0, 10] with the rollout
policy), so clipped tokens contribute no gradient.

## Auxiliary views carry no gradient

Every auxiliary view (PAPO's masked image, DVRP's masked and noised images, VPPO's and PGPO's
perturbed passes, CFPO's counterfactual, VEPO's noised image) is scored once with the rollout policy
before the update, as in PAPO's default (`RECOMPUTE_AUG_LOG_PROBS=False`). Gradients therefore
reach the policy only through the original-input branch. In particular, the entropy terms that PAPO
(Double Entropy) and DVRP (λ_ent) put on their auxiliary views are constants for the update and do
not change training; the PAPO authors confirmed this and found that recomputing the masked view with
gradient does not help ([PAPO issue #20](https://github.com/MikeWangWZHL/PAPO/issues/20)).

## Dual-clip

For tokens with a negative advantage, the PPO ratio is additionally bounded by
`worker.actor.clip_ratio_dual` (default 3.0, EasyR1's), as in dual-clip PPO. All scripts keep the
default.

## seq and token loss averaging

`worker.actor.loss_avg_mode` decides how token losses are combined into the loss of a mini-batch
(one optimizer step), which is split into micro-batches and across data-parallel ranks:

- `token`: the mean over every response token of the mini-batch on all ranks; each micro-batch's
  token mean is rescaled by its share of the mini-batch's tokens.
- `seq`: each response's loss is the mean over its own tokens, and every response with at least one
  token is weighted equally across micro-batches, dynamic batches and ranks; each micro-batch's
  response mean is rescaled by its share of the mini-batch's responses. This is verl's
  `seq-mean-token-mean`.

Upstream EasyR1 rescales by the share of tokens in both modes, so there a response's weight in `seq`
mode grows with the mean response length of the micro-batch it lands in. The PEPO GRPO arms and GRIT
use `seq`, as their trainers (ms-swift's GRPO loss and GRIT's TRL trainer) average per response.

## Where top-quantile thresholds are taken

Token selection by entropy (`top_entropy_quantile`) or visual sensitivity (`top_perception_quantile`)
keeps the top fraction of tokens; `*_thr_granularity` decides among which tokens:

- `micro_batch` (default): within each update micro-batch, from the entropy of the policy being
  updated. CGPO uses it, as in its paper experiments.
- `batch`: over every response token of the rollout batch (after online filtering), from the rollout
  policy, computed on the driver before the update; `entropy_top_p` < 1 uses the entropy of the
  renormalized top-p distribution. ToR uses it with top-p 0.95 (paper Eq. 5-9).
- `response`: within each response (VPPO, VEPO).

The selection keeps exactly the top fraction. With `batch`, the tokens tied at the threshold are
drawn uniformly at random (seeded by `data.seed` and the step): with top-p 0.95 most tokens have a
truncated entropy of exactly 0, and taking them in sort order would select them in runs of
consecutive responses. With `micro_batch` and `response` the scores are continuous, and tied
tokens are taken in sort order.

## Noise on normalized pixel values

`corrupt_image=gaussian_noise` adds N(0, std²) to the image processor's normalized pixel values,
without clipping, as VEPO's code. A normalized value is (pixel − mean) / image_std, so a std of σ in
normalized units is σ · image_std in [0, 1] pixel units: 2.0 is about 0.54 for Qwen2.5-VL (image_std
about 0.27 per channel) and 1.08 is 0.54 for Qwen3-VL (image_std 0.5). The evaluation perturbation
`--perturbation.type gaussian_noise` works on [0, 1] pixels (with clipping), because vLLM receives
images, not features.

## Online filtering

DAPO-style online filtering keeps the prompt groups whose rewards vary and generates more rounds until
the batch is full:

- `filter_criterion=mean_range` keeps a group whose mean `filter_key` lies strictly in
  (`filter_low`, `filter_high`); `std` keeps a group whose values differ (unbiased std > 0), as
  ms-swift's dynamic sampling. For a 0/1 key such as accuracy the two agree.
- `online_filtering_fallback` decides what happens when a round keeps no group and when
  `trainer.max_try_make_batch` rounds do not fill the batch: `error` raises in both cases;
  `keep_round` keeps such a round whole and still raises after the last round, as PAPO's code;
  `first_round` drops such a round and trains on the first round unfiltered once the rounds run out,
  as ms-swift (`max_resample_times=3` checks the first round and two resamples); it needs whole
  rounds, so `data.mini_rollout_batch_size` must be unset or at least `data.rollout_batch_size`.
