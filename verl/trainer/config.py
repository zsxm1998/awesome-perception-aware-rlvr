# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


MODEL_LEVEL_VISUAL_CORRUPTIONS = frozenset({"cross_modal_attention_value_mean"})


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


def _parse_optional_json_dict(value: Mapping[str, Any] | str | None, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must decode to a JSON object, but got {type(parsed).__name__}.")
        return parsed
    raise TypeError(f"{field_name} must be a mapping, JSON string, or None, but got {type(value).__name__}.")


def _validate_choice(field_name: str, value: str | None, choices: set[str], allow_none: bool = False) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{field_name} must be one of {sorted(choices)}, but got None.")
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}, but got {value!r}.")


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_key: Optional[str] = None
    """column holding a per-row system prompt; a row with a non-empty value uses it instead of `system_prompt`
    (DeepEyes: the official ThinkLite prompt, or every row's official prompt)."""
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.system_prompt = get_abs_path(self.system_prompt, prompt="System prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    filter_criterion: str = "mean_range"
    """which groups online filtering keeps. `mean_range`: the group mean of `filter_key` lies strictly between
    `filter_low` and `filter_high`. `std`: the `filter_key` values of the group differ (unbiased std > 0), as
    ms-swift's dynamic sampling. For a 0/1 key such as accuracy the two agree."""
    online_filtering_fallback: str = "error"
    """what online filtering does when a round keeps no group, and when `trainer.max_try_make_batch` rounds do not
    fill the batch. `error`: raise in both cases. `keep_round`: keep such a round whole, still raise after the last
    round (PAPO's code). `first_round`: drop such a round, and after the last round train on the first round
    unfiltered (ms-swift's dynamic sampling)."""
    log_entropy: bool = False
    """log current-policy token entropy statistics."""
    top_entropy_quantile: float = 1.0
    """fraction of highest-entropy tokens to keep for entropy-mask modulation. Used by ToR and CGPO; VPPO/PAPO leave it at 1.0."""
    entropy_thr_granularity: str = "micro_batch"
    """where the top-entropy threshold is taken. `batch`: over every response token of the rollout batch, before the update, from the rollout (old) policy's entropy (ToR). `micro_batch`: within each update micro-batch, from the entropy of the policy being updated (CGPO, as in its paper). `response`: within each response."""
    entropy_top_p: float = 1.0
    """entropy of the top-p truncated (renormalized) distribution for `entropy_thr_granularity=batch`; 1.0 is the full vocabulary. ToR uses 0.95 (paper Eq. 5)."""
    corrupt_image: Optional[str] = None
    """auxiliary visual corruption. `random_patch` is used by PAPO/VPPO/DVRP, `no_image` by ToR, `cgpo_flat|cgpo_hierarchical` by CGPO, `mask_visual_attention` by PGPO, `pixelation` removes fine image details while preserving image size, and `cross_modal_attention_value_mean` applies a scoped model-level intervention."""
    corrupt_image_kwargs: Any = None
    """JSON/mapping kwargs passed to the selected `corrupt_image` transform, e.g. patch size, black probability, pixelation ratio, CGPO fill type, or saliency threshold multiplier. `random_patch` takes `patch_size`, `black_prob` and `mask_before_resize` (default false: mask the resized image, so the patches line up with the vision encoder's; true: mask the original image, then resize it, as PAPO's code). `gaussian_noise` takes `std` (default 2.0), added to the image processor's normalized pixel_values without clipping, as VEPO's code: one draw per prompt, or per response with `corrupt_image_position=response`."""
    corrupt_image_position: str = "prompt"
    """Whether corruption is built from the prompt side or from the generated response side. VPPO/CGPO mostly use response; PAPO/DVRP uses prompt."""
    visual_sensitivity_loss_coef: float = 0.0
    """coefficient for KL-based decremental-view sensitivity regularization (KL_prcp in PAPO). Used by PAPO/DVRP, not needed by VPPO/ToR/CGPO default scripts."""
    visual_sensitivity_reference: str = "current"
    """which branch is compared against decremental log-probs when computing visual sensitivity / KL_prcp. CGPO and VPPO/ToR use `old`; PAPO and DVRP use `current`."""
    visual_sensitivity_metric: str = "sampled_low_var_kl"
    """token-level visual sensitivity signal. `sampled_low_var_kl` preserves existing PAPO/VPPO/ToR/CGPO behavior; `sampled_boxcox` is the bounded Box-Cox power gap (p^α - q^α)/α (PowerOPD-style); `sampled_abs_log_ratio` is |log p - log q| (ToR); `full_vocab_*` are generic distribution measures between the original and corrupted views; `vepo` is VEPO's fused recipe (JSD ⊕ abs entropy gap via soft-or)."""
    visual_sensitivity_boxcox_alpha: float = 1.0
    """Box-Cox α for `visual_sensitivity_metric='sampled_boxcox'`. Scores are bounded to [-1/α, 1/α]; α→0 approaches the sampled log-ratio. The 1/α denominator is kept so scores stay comparable across α sweeps."""
    visual_sensitivity_log_metrics: Any = None
    """extra sampled sensitivity estimators to log for diagnosis (detached, never trained on), e.g. `sampled_low_var_kl` while training with `sampled_boxcox`. Accepts a list or a comma-separated string; only sampled-family metrics are allowed. Each entry also emits comparison diagnostics against the active signal: top-quantile token overlap (`diag_topq_overlap_{active}_vs_{extra}`, needs `top_perception_quantile < 1`) and response-level Spearman rank correlation (`diag_response_rank_corr_...`). Takes effect when decremental log-probs are available."""
    visual_sensitivity_hidden_metric: str = "cosine"
    """hidden-state visual sensitivity metric for single-forward visual-token/response-token similarity."""
    visual_token: str = "auto"
    """visual payload token used by hidden-state visual sensitivity. `auto` resolves common Qwen/InternVL image/video placeholder ids from the tokenizer."""
    visual_sensitivity_jsd_weight: float = 0.5
    """VEPO sub-parameter: JSD weight in the soft-or fusion; the absolute entropy-gap signal gets `1 - visual_sensitivity_jsd_weight`. Only used when `visual_sensitivity_metric='vepo'`."""
    visual_sensitivity_entropy_gate: str = "none"
    """optional full-vocab sensitivity gate. `normal_entropy` multiplies the fused visual signal by normalized original-image entropy."""
    invariant_entropy_coef: float = 0.0
    """coefficient for entropy regularization on the original/current view (positive: penalty, negative: bonus). Used by VPPO, PAPO, VEPO and the shared comparison scripts. Both forms carry gradient: `entropy_loss_type=sampled` uses -log p of the sampled token, `full` the full-vocabulary entropy."""
    decremental_entropy_coef: float = 0.0
    """coefficient for entropy regularization on the decremental auxiliary view. Introduced by PAPO and also reused by DVRP."""
    entropy_loss_type: str = "sampled"
    """how entropy regularization terms are computed: sampled `-log_prob` (PAPO/VPPO/DVRP defaults) or full entropy (research/debug setting)."""
    top_perception_quantile: float = 1.0
    """fraction of highest-perception-shift tokens kept for perception-mask modulation. Used by VPPO, ToR, and CGPO."""
    perception_thr_granularity: str = "micro_batch"
    """where the top-perception threshold is taken: `batch` (every response token of the rollout batch, before the update; ToR), `micro_batch` (within each update micro-batch; CGPO) or `response` (within each response; VPPO, VEPO)."""
    advantage_scaling_method: Optional[str] = None
    """advantage-scaling formula. `None` disables scaling, `vppo` and `cgpo` preserve the existing response-level rules as token matrices, and `pgpo` applies token-level perception-grounded scaling."""
    response_advantage_scaling_method: Optional[str] = None
    """legacy alias for `advantage_scaling_method` with `vppo|cgpo`; kept so existing VPPO/CGPO scripts are unchanged."""
    vppo_response_scaling_min: float = 0.9
    """lower bound used by the VPPO response-scaling rule. Only meaningful when `response_advantage_scaling_method='vppo'`."""
    cgpo_response_scaling_coef: float = 0.1
    """scale factor used by the CGPO response-scaling rule `1 + coef * tanh(normalized_score)`. Only used when `response_advantage_scaling_method='cgpo'`."""
    pgpo_token_scaling_threshold: float = 0.4
    """PGPO token-scaling threshold; corresponds to tau in the paper."""
    pgpo_token_scaling_boost: float = 2.0
    """PGPO token-scaling boost factor; corresponds to beta in the paper."""
    pgpo_threshold_mode: str = "absolute"
    """what `pgpo_token_scaling_threshold` is. `absolute`: a value of the min-max normalized score (paper Eq. 6). `quantile`: a fraction of each response's tokens; the threshold is the response's ascending normalized score at index floor(length * threshold) (the authors' development code, Yzk1114/EasyR1)."""
    pgpo_low_weight_floor: float = 0.0
    """lower bound of the weight of the tokens below the threshold. The paper has none (0.0); the authors' development code uses 0.1."""
    pgpo_mass_normalization: bool = True
    """rescale each response's token weights to sum to its length (paper Eq. 7). The authors' development code does not (false)."""
    advantage_scaling_schedule: str = "none"
    """generic interpolation for token/response advantage scaling. `linear` applies 1 + progress * (factor - 1)."""
    pepo_gate_alpha: float = 0.05
    """PEPO entropy-gated visual-weight alpha; only used when `advantage_scaling_method='pepo'`."""
    pepo_gate_temperature: float = 1.8
    """PEPO softmax temperature for visual-token similarity weights; only used when `advantage_scaling_method='pepo'`."""
    incremental_image_transform: Optional[str] = None
    """incremental auxiliary-view transform. Configuring this, or non-zero incremental-view loss coefficients, enters DVRP-style dual-view mode. Currently only `vp_diffusion` is supported."""
    incremental_image_kwargs: Any = None
    """JSON/mapping kwargs for the incremental auxiliary transform. Supplying them implies DVRP-style incremental-view training is intended."""
    visual_robustness_loss_coef: float = 0.0
    """coefficient for the incremental-view robustness KL term. Non-zero values enable the DVRP-style incremental loss path."""
    incremental_entropy_coef: float = 0.0
    """coefficient for entropy regularization on the incremental auxiliary view. Non-zero values enable the DVRP-style incremental loss path."""
    noise_t_init: float = 500.0
    """initial VP-diffusion timestep for DVRP's incremental view; the step t anneals as noise_t_init * sigmoid(noise_gamma * (0.5 - progress)), and the view is sqrt(alpha_bar_t) x + sqrt(1 - alpha_bar_t) eps on NoisyRollout's 1000-step schedule (DVRP follows NoisyRollout)."""
    noise_gamma: float = 10.0
    """sigmoid annealing sharpness for DVRP VP-diffusion scheduling."""
    noise_t_max: float = 1000.0
    """largest VP-diffusion timestep for DVRP, at most 1000 (the length of NoisyRollout's schedule)."""
    tor_use_token_weighting: bool = False
    """enable ToR-style token weighting instead of binary masking when entropy/perception masks are available."""
    tor_rsn_weight: float = 1.0
    """weight assigned to entropy-selected reasoning tokens by ToR."""
    tor_prcp_weight: float = 0.5
    """weight assigned to perception-selected tokens by ToR."""
    normalize_pg_loss_by_selected_tokens: bool = False
    """if enabled, average the effective perception-reasoning PG loss over selected tokens when a selection mask exists."""
    use_grounding_consistency_reward: bool = False
    """enable trainer-side grounding consistency reward via second-pass detection prompting. Used by CGPO only."""
    grounding_consistency_reward_weight: float = 0.1
    """multiplier applied to the grounding consistency reward before it is added to the training reward. Used by CGPO."""
    grounding_consistency_detector: str = "self"
    """detector backend for grounding consistency reward. `self` preserves the existing rollout-model detector; `grounding-dino` uses IDEA-Research/grounding-dino-base."""
    grounding_dino_device: str = "worker"
    """device for the Grounding DINO detector. `worker` runs one detector per rollout worker GPU; `auto` uses local cuda:0 when available."""
    grounding_dino_batch_size: int = 4
    """mini-batch size for Grounding DINO detector inference."""
    include_region_tokens_in_perception_mask: bool = False
    """OR tokens inside `<region>...</region>` spans into the perception mask. Used by CGPO; ignored elsewhere."""

    def post_init(self):
        self.corrupt_image_kwargs = _parse_optional_json_dict(self.corrupt_image_kwargs, "corrupt_image_kwargs")
        self.incremental_image_kwargs = _parse_optional_json_dict(
            self.incremental_image_kwargs, "incremental_image_kwargs"
        )

        _validate_choice(
            "corrupt_image",
            self.corrupt_image,
            {
                "no_image",
                "random_patch",
                "gaussian_noise",
                "pixelation",
                "cgpo_flat",
                "cgpo_hierarchical",
                "mask_visual_attention",
                *MODEL_LEVEL_VISUAL_CORRUPTIONS,
            },
            allow_none=True,
        )
        _validate_choice(
            "grounding_consistency_detector", self.grounding_consistency_detector, {"self", "grounding-dino"}
        )
        if not 0.0 < self.noise_t_max <= 1000.0:
            raise ValueError(f"noise_t_max must be in (0, 1000], but got {self.noise_t_max}.")
        _validate_choice("filter_criterion", self.filter_criterion, {"mean_range", "std"})
        _validate_choice(
            "online_filtering_fallback", self.online_filtering_fallback, {"error", "keep_round", "first_round"}
        )
        _validate_choice("corrupt_image_position", self.corrupt_image_position, {"prompt", "response"})
        _validate_choice("entropy_loss_type", self.entropy_loss_type, {"sampled", "full"})
        _validate_choice("visual_sensitivity_reference", self.visual_sensitivity_reference, {"current", "old"})
        # Kept as literals so config parsing stays torch-free; a test asserts they match
        # the constants in visual_sensitivity.py.
        sampled_sensitivity_metrics = {"sampled_low_var_kl", "sampled_boxcox", "sampled_abs_log_ratio"}
        full_vocab_sensitivity_metrics = {
            "full_vocab_jsd",
            "full_vocab_kl",
            "full_vocab_hellinger",
            "full_vocab_abs_entropy_gap",
            "vepo",
        }
        _validate_choice(
            "visual_sensitivity_metric",
            self.visual_sensitivity_metric,
            sampled_sensitivity_metrics | full_vocab_sensitivity_metrics | {"hidden_state_similarity"},
        )
        if self.corrupt_image == "random_patch":
            corruption_kwargs = self.corrupt_image_kwargs or {}
            unknown_kwargs = set(corruption_kwargs) - {"patch_size", "black_prob", "mask_before_resize"}
            if unknown_kwargs:
                raise ValueError(f"random_patch received unsupported corrupt_image_kwargs: {sorted(unknown_kwargs)}.")
            if not isinstance(corruption_kwargs.get("mask_before_resize", False), bool):
                raise ValueError("corrupt_image_kwargs.mask_before_resize must be true or false.")
        if self.corrupt_image == "gaussian_noise":
            corruption_kwargs = self.corrupt_image_kwargs or {}
            unknown_kwargs = set(corruption_kwargs) - {"std"}
            if unknown_kwargs:
                raise ValueError(
                    f"gaussian_noise received unsupported corrupt_image_kwargs: {sorted(unknown_kwargs)}."
                )
            std = corruption_kwargs.get("std", 2.0)
            if isinstance(std, bool) or not isinstance(std, (int, float)) or std <= 0.0:
                raise ValueError(f"corrupt_image_kwargs.std must be a positive number, but got {std!r}.")
        if self.corrupt_image == "pixelation":
            corruption_kwargs = self.corrupt_image_kwargs or {}
            unknown_kwargs = set(corruption_kwargs) - {"ratio"}
            if unknown_kwargs:
                raise ValueError(f"pixelation received unsupported corrupt_image_kwargs: {sorted(unknown_kwargs)}.")
            ratio = corruption_kwargs.get("ratio", 0.1)
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 < float(ratio) <= 1.0:
                raise ValueError(f"corrupt_image_kwargs.ratio must be a number in (0, 1], but got {ratio!r}.")
        if self.corrupt_image in MODEL_LEVEL_VISUAL_CORRUPTIONS:
            if self.corrupt_image_position != "prompt":
                raise ValueError(
                    f"{self.corrupt_image} only supports corrupt_image_position='prompt'. "
                    "The position selects when the corruption is constructed; the intervention query span is "
                    "defined by the visual token boundaries."
                )
            corruption_kwargs = self.corrupt_image_kwargs or {}
            unknown_kwargs = set(corruption_kwargs) - {"saliency_std_multiplier"}
            if unknown_kwargs:
                raise ValueError(
                    f"{self.corrupt_image} received unsupported corrupt_image_kwargs: {sorted(unknown_kwargs)}."
                )
            saliency_std_multiplier = corruption_kwargs.get("saliency_std_multiplier", 2.0)
            if (
                isinstance(saliency_std_multiplier, bool)
                or not isinstance(saliency_std_multiplier, (int, float))
                or saliency_std_multiplier < 0.0
            ):
                raise ValueError(
                    "corrupt_image_kwargs.saliency_std_multiplier must be a non-negative number, "
                    f"but got {saliency_std_multiplier!r}."
                )
            if self.visual_sensitivity_metric in full_vocab_sensitivity_metrics:
                raise ValueError(
                    f"{self.corrupt_image} does not currently support full-vocab visual_sensitivity_metric values."
                )
        if self.visual_sensitivity_boxcox_alpha <= 0.0:
            raise ValueError(
                f"visual_sensitivity_boxcox_alpha must be positive, but got {self.visual_sensitivity_boxcox_alpha}."
            )
        if self.visual_sensitivity_log_metrics is None:
            self.visual_sensitivity_log_metrics = ()
        elif isinstance(self.visual_sensitivity_log_metrics, str):
            self.visual_sensitivity_log_metrics = tuple(
                item.strip() for item in self.visual_sensitivity_log_metrics.split(",") if item.strip()
            )
        else:
            self.visual_sensitivity_log_metrics = tuple(self.visual_sensitivity_log_metrics)
        for log_metric in self.visual_sensitivity_log_metrics:
            if log_metric not in sampled_sensitivity_metrics:
                raise ValueError(
                    "visual_sensitivity_log_metrics only supports sampled metrics "
                    f"{sorted(sampled_sensitivity_metrics)}, but got {log_metric!r}."
                )
        _validate_choice(
            "visual_sensitivity_hidden_metric",
            self.visual_sensitivity_hidden_metric,
            {"cosine", "l1", "l2", "dot"},
        )
        _validate_choice(
            "visual_sensitivity_entropy_gate", self.visual_sensitivity_entropy_gate, {"none", "normal_entropy"}
        )
        _validate_choice("entropy_thr_granularity", self.entropy_thr_granularity, {"batch", "micro_batch", "response"})
        _validate_choice(
            "advantage_scaling_method",
            self.advantage_scaling_method,
            {"vppo", "cgpo", "pgpo", "pepo"},
            allow_none=True,
        )
        _validate_choice(
            "response_advantage_scaling_method",
            self.response_advantage_scaling_method,
            {"vppo", "cgpo"},
            allow_none=True,
        )
        if self.advantage_scaling_method is None and self.response_advantage_scaling_method is not None:
            self.advantage_scaling_method = self.response_advantage_scaling_method
        elif (
            self.advantage_scaling_method is not None
            and self.response_advantage_scaling_method is not None
            and self.advantage_scaling_method != self.response_advantage_scaling_method
        ):
            raise ValueError("`advantage_scaling_method` conflicts with legacy `response_advantage_scaling_method`.")
        _validate_choice("advantage_scaling_schedule", self.advantage_scaling_schedule, {"none", "linear"})
        if self.pepo_gate_alpha < 0.0:
            raise ValueError(f"pepo_gate_alpha must be non-negative, but got {self.pepo_gate_alpha}.")
        if self.pepo_gate_temperature <= 0.0:
            raise ValueError(f"pepo_gate_temperature must be positive, but got {self.pepo_gate_temperature}.")
        if self.advantage_scaling_method == "pepo" and self.visual_sensitivity_metric != "hidden_state_similarity":
            raise ValueError(
                "advantage_scaling_method='pepo' requires visual_sensitivity_metric='hidden_state_similarity'."
            )
        if self.visual_sensitivity_metric == "hidden_state_similarity" and self.advantage_scaling_method in {
            "vppo",
            "cgpo",
        }:
            raise ValueError(
                "visual_sensitivity_metric='hidden_state_similarity' currently supports token-level "
                "advantage scaling methods such as 'pgpo' and 'pepo', not 'vppo' or 'cgpo'."
            )
        _validate_choice(
            "incremental_image_transform",
            self.incremental_image_transform,
            {"vp_diffusion"},
            allow_none=True,
        )
        _validate_choice(
            "perception_thr_granularity", self.perception_thr_granularity, {"batch", "micro_batch", "response"}
        )
        if not 0.0 < self.entropy_top_p <= 1.0:
            raise ValueError(f"entropy_top_p must be in (0, 1], but got {self.entropy_top_p}.")
        uses_batch_entropy_mask = self.top_entropy_quantile < 1.0 and self.entropy_thr_granularity == "batch"
        uses_batch_perception_mask = self.top_perception_quantile < 1.0 and self.perception_thr_granularity == "batch"
        if self.entropy_top_p < 1.0 and not uses_batch_entropy_mask:
            raise ValueError(
                "entropy_top_p only applies to batch-level entropy masks "
                "(top_entropy_quantile < 1 with entropy_thr_granularity=batch)."
            )
        if uses_batch_entropy_mask and self.visual_sensitivity_metric in full_vocab_sensitivity_metrics:
            raise ValueError(
                "entropy_thr_granularity=batch is not supported with full-vocab visual_sensitivity_metric values; "
                "use micro_batch or response."
            )
        if uses_batch_perception_mask:
            if self.visual_sensitivity_metric == "hidden_state_similarity":
                raise ValueError(
                    "perception_thr_granularity=batch needs the perception scores before the update; "
                    "hidden_state_similarity scores only exist during the update (use micro_batch or response)."
                )
            if (
                self.visual_sensitivity_metric not in full_vocab_sensitivity_metrics
                and self.visual_sensitivity_reference != "old"
            ):
                raise ValueError(
                    "perception_thr_granularity=batch scores tokens before the update and therefore requires "
                    "visual_sensitivity_reference=old."
                )
        for field_name, quantile in {
            "top_entropy_quantile": self.top_entropy_quantile,
            "top_perception_quantile": self.top_perception_quantile,
        }.items():
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], but got {quantile}.")
        if not 0.0 <= self.visual_sensitivity_jsd_weight <= 1.0:
            raise ValueError(
                f"visual_sensitivity_jsd_weight must be in [0, 1], but got {self.visual_sensitivity_jsd_weight}."
            )
        if not 0.0 < self.pgpo_token_scaling_threshold < 1.0:
            raise ValueError(
                f"pgpo_token_scaling_threshold must be in (0, 1), but got {self.pgpo_token_scaling_threshold}."
            )
        if self.pgpo_token_scaling_boost < 0.0:
            raise ValueError(
                f"pgpo_token_scaling_boost must be non-negative, but got {self.pgpo_token_scaling_boost}."
            )
        _validate_choice("pgpo_threshold_mode", self.pgpo_threshold_mode, {"absolute", "quantile"})
        if not 0.0 <= self.pgpo_low_weight_floor <= 1.0:
            raise ValueError(f"pgpo_low_weight_floor must be in [0, 1], but got {self.pgpo_low_weight_floor}.")

        uses_hidden_state_visual_sensitivity = self.visual_sensitivity_metric == "hidden_state_similarity"
        requires_decremental_view = (
            self.visual_sensitivity_loss_coef != 0.0
            or self.decremental_entropy_coef != 0.0
            or (
                not uses_hidden_state_visual_sensitivity
                and (self.top_perception_quantile < 1.0 or self.advantage_scaling_method is not None)
            )
        )
        if requires_decremental_view and self.corrupt_image is None:
            raise ValueError("Perception-aware losses require `corrupt_image` to be configured.")

        if self.visual_sensitivity_loss_coef != 0.0 and self.visual_sensitivity_reference != "current":
            raise ValueError(
                "`visual_sensitivity_reference` must be `current` when `visual_sensitivity_loss_coef` is non-zero."
            )

        if self.tor_use_token_weighting and self.top_entropy_quantile >= 1.0 and self.top_perception_quantile >= 1.0:
            raise ValueError(
                "`tor_use_token_weighting` requires `top_entropy_quantile < 1.0` or `top_perception_quantile < 1.0`."
            )

        uses_incremental_dvrp = (
            self.incremental_image_transform is not None
            or self.incremental_image_kwargs is not None
            or self.visual_robustness_loss_coef != 0.0
            or self.incremental_entropy_coef != 0.0
        )
        if uses_incremental_dvrp:
            if self.corrupt_image is None:
                raise ValueError(
                    "Incremental-view (DVRP-style) training requires `corrupt_image` to configure the decremental view."
                )
            if self.incremental_image_transform != "vp_diffusion":
                raise ValueError(
                    "Incremental-view (DVRP-style) training currently requires `incremental_image_transform=vp_diffusion`."
                )

        for field_name, field_value in {
            "noise_t_init": self.noise_t_init,
            "noise_gamma": self.noise_gamma,
            "noise_t_max": self.noise_t_max,
            "grounding_consistency_reward_weight": self.grounding_consistency_reward_weight,
        }.items():
            if field_value <= 0.0:
                raise ValueError(f"{field_name} must be positive, but got {field_value}.")
        if self.grounding_dino_batch_size <= 0:
            raise ValueError(f"grounding_dino_batch_size must be positive, but got {self.grounding_dino_batch_size}.")
        if not self.grounding_dino_device:
            raise ValueError("grounding_dino_device must be non-empty.")

        if self.cgpo_response_scaling_coef < 0.0:
            raise ValueError(
                f"cgpo_response_scaling_coef must be non-negative, but got {self.cgpo_response_scaling_coef}."
            )


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    train_generations_to_log: int = 0
    """number of generations to log per training step, -1 means all generated training samples"""
    log_images: bool = True
    """whether to create and upload image previews in generation tables"""
    generation_log_image_max_size: int = 768
    """maximum longest edge for saved generation image previews"""
    generation_log_caption_max_chars: int = 512
    """maximum caption characters for uploaded generation images"""
    generations_log_file: str = "completions.jsonl"
    """local JSONL filename for generation sample logs"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.rollout.max_dynamic_patch = self.worker.actor.model.max_dynamic_patch
        # the agentic rollout renders multi-turn prompts with the worker's processor, so it must use the same template
        self.worker.actor.model.override_chat_template = self.data.override_chat_template
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self._validate_batch_sizes()
        rollout = self.worker.rollout
        if (
            rollout.agent_observation_min_pixels is not None
            and self.data.max_pixels is not None
            and rollout.agent_observation_min_pixels > self.data.max_pixels
        ):
            raise ValueError(
                f"worker.rollout.agent_observation_min_pixels ({rollout.agent_observation_min_pixels}) cannot exceed "
                f"data.max_pixels ({self.data.max_pixels})."
            )
        if rollout.max_num_batched_tokens < rollout.prompt_length + rollout.response_length:
            raise ValueError(  # checked again when the vLLM rollout starts
                f"worker.rollout.max_num_batched_tokens ({rollout.max_num_batched_tokens}) must be at least "
                f"data.max_prompt_length + data.max_response_length ({rollout.prompt_length + rollout.response_length})."
            )
        if (
            self.algorithm.online_filtering
            and self.algorithm.online_filtering_fallback == "first_round"
            and self.trainer.max_try_make_batch <= 0
        ):
            raise ValueError(
                "algorithm.online_filtering_fallback=first_round requires trainer.max_try_make_batch > 0."
            )
        if (
            self.algorithm.online_filtering
            and self.algorithm.online_filtering_fallback == "first_round"
            and self.data.mini_rollout_batch_size is not None
            and self.data.mini_rollout_batch_size < self.data.rollout_batch_size
        ):
            raise ValueError(
                "algorithm.online_filtering_fallback=first_round trains on the first round unfiltered, which needs a "
                "whole rollout batch: data.mini_rollout_batch_size must be unset or at least data.rollout_batch_size."
            )
        if self.worker.rollout.interaction_mode == "agentic":
            unsupported_interventions = []
            intervention_fields = {
                "algorithm.corrupt_image": self.algorithm.corrupt_image is not None,
                "algorithm.incremental_image_transform": self.algorithm.incremental_image_transform is not None,
                "algorithm.visual_sensitivity_loss_coef": self.algorithm.visual_sensitivity_loss_coef != 0.0,
                "algorithm.decremental_entropy_coef": self.algorithm.decremental_entropy_coef != 0.0,
                "algorithm.visual_robustness_loss_coef": self.algorithm.visual_robustness_loss_coef != 0.0,
                "algorithm.incremental_entropy_coef": self.algorithm.incremental_entropy_coef != 0.0,
                "algorithm.advantage_scaling_method": self.algorithm.advantage_scaling_method is not None,
                "algorithm.response_advantage_scaling_method": (
                    self.algorithm.response_advantage_scaling_method is not None
                ),
                "algorithm.tor_use_token_weighting": self.algorithm.tor_use_token_weighting,
                "algorithm.use_grounding_consistency_reward": self.algorithm.use_grounding_consistency_reward,
            }
            unsupported_interventions.extend(name for name, enabled in intervention_fields.items() if enabled)
            if unsupported_interventions:
                raise ValueError(
                    "DeepEyes agentic training does not support grounded-reasoning "
                    "interventions because their image/box semantics are defined for "
                    "one-shot responses; disable: " + ", ".join(unsupported_interventions)
                )
            if self.data.format_prompt is not None:
                raise ValueError("DeepEyes agentic training cannot use data.format_prompt")
            from ..workers.agent.coordinates import (
                check_chat_template_supports_tools,
                check_prompt_matches_bbox_format,
                resolve_bbox_format,
            )

            # Resolve "auto" once (Qwen2-VL / Qwen2.5-VL -> absolute pixels, others -> 0-1000).
            self.worker.rollout.agent_bbox_format = resolve_bbox_format(
                self.worker.rollout.agent_bbox_format, self.worker.actor.model.model_path
            )
            if self.worker.rollout.agent_prompt_style == "official":
                # the official prompts come with the data and carry the tool schema as text
                if not self.data.system_prompt_key or self.data.system_prompt is not None:
                    raise ValueError(
                        "worker.rollout.agent_prompt_style=official takes the system prompt from the data; "
                        "set data.system_prompt_key (DeepEyes: official_system_prompt) and data.system_prompt=null"
                    )
            else:
                if self.data.system_prompt is None:
                    raise ValueError("DeepEyes agentic training requires data.system_prompt")
                from ..workers.agent.chat import render_deepeyes_system_prompt

                prompt_path = get_abs_path(
                    self.data.system_prompt,
                    prompt="DeepEyes system prompt file",
                )
                if prompt_path is None:
                    raise FileNotFoundError(f"DeepEyes system prompt file was not found: {self.data.system_prompt}")
                with open(prompt_path, encoding="utf-8") as prompt_file:
                    prompt_text = prompt_file.read()
                render_deepeyes_system_prompt(
                    prompt_text,
                    max_tool_calls=self.worker.rollout.agent_max_tool_calls,
                )
                # refuse a system prompt that describes the other coordinate convention
                check_prompt_matches_bbox_format(prompt_text, self.worker.rollout.agent_bbox_format)
                check_chat_template_supports_tools(
                    self.worker.actor.model.model_path, self.data.override_chat_template
                )

    def _validate_batch_sizes(self):
        """Fail early (also under DRY_RUN) on batch sizes the trainer and workers would reject."""
        actor = self.worker.actor
        rollout_n = max(int(self.worker.rollout.n), 1)
        if self.data.rollout_batch_size % actor.global_batch_size != 0:
            raise ValueError(
                f"data.rollout_batch_size ({self.data.rollout_batch_size}) must be divisible by "
                f"worker.actor.global_batch_size ({actor.global_batch_size})."
            )
        if (self.data.rollout_batch_size * rollout_n) % actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "data.rollout_batch_size * worker.rollout.n must be divisible by "
                "worker.actor.micro_batch_size_per_device_for_experience."
            )
        data_parallel_size = self.trainer.n_gpus_per_node * self.trainer.nnodes // max(actor.ulysses_size, 1)
        per_device = actor.global_batch_size * rollout_n // max(data_parallel_size, 1)
        if per_device == 0 or per_device % actor.micro_batch_size_per_device_for_update != 0:
            raise ValueError(
                f"worker.actor.global_batch_size * worker.rollout.n / data-parallel size = {per_device} must be a "
                "positive multiple of worker.actor.micro_batch_size_per_device_for_update "
                f"({actor.micro_batch_size_per_device_for_update}); adjust the batch sizes or N_GPUS_PER_NODE."
            )

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
