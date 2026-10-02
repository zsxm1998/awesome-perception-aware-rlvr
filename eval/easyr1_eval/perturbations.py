# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
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
from __future__ import annotations

import copy
import itertools
import json
import re
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from verl.utils.perturbations.pixel import (
    gaussian_noise_image,
    random_patch_blackening,
    vp_diffusion_noise_fixed_t,
)
from verl.utils.perturbations.seeding import derive_seed


PIXEL_PERTURBATIONS = {"random_patch", "gaussian_noise", "vp_diffusion"}
FEATURE_PERTURBATIONS = {"feature_divergence", "feature_consistency"}


@dataclass(frozen=True)
class PerturbationConfig:
    type: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.type != "none"

    @property
    def is_pixel(self) -> bool:
        return self.type in PIXEL_PERTURBATIONS

    @property
    def is_feature(self) -> bool:
        return self.type in FEATURE_PERTURBATIONS

    def summary(self) -> str:
        if not self.enabled:
            return "clean"
        parts = [self.type]
        for key, value in sorted(self.params.items()):
            if value is None or value == "":
                continue
            parts.append(f"{key}-{value}")
        return slugify("_".join(parts))


def add_perturbation_args(parser) -> None:
    parser.add_argument("--seeds", help="Comma-separated perturbation/evaluation seeds. Overrides --seed per run.")
    parser.add_argument("--perturbation.type", dest="perturbation_type", default="none")
    parser.add_argument("--perturbation.patch_size", dest="perturbation_patch_size")
    parser.add_argument("--perturbation.black_prob", dest="perturbation_black_prob")
    parser.add_argument("--perturbation.std", dest="perturbation_std")
    parser.add_argument(
        "--perturbation.noise_t_init",
        dest="perturbation_noise_t_init",
        help="Fixed VP noise step for eval; training noise_gamma/noise_t_max annealing is not advanced during eval.",
    )
    parser.add_argument(
        "--perturbation.noise_gamma",
        dest="perturbation_noise_gamma",
        help="Accepted for config parity but ignored by eval VP noise; noise_t_init is used as the fixed step.",
    )
    parser.add_argument("--perturbation.noise_t_max", dest="perturbation_noise_t_max")
    parser.add_argument("--perturbation.k", dest="perturbation_k")
    parser.add_argument("--perturbation.ev_threshold", dest="perturbation_ev_threshold")
    parser.add_argument("--perturbation.n_skip", dest="perturbation_n_skip")
    parser.add_argument("--perturbation.auto_rogue", dest="perturbation_auto_rogue")
    parser.add_argument("--perturbation.rogue_corr_threshold", dest="perturbation_rogue_corr_threshold")
    parser.add_argument("--perturbation.rho", dest="perturbation_rho")
    parser.add_argument("--perturbation.token_p", dest="perturbation_token_p")
    parser.add_argument("--perturbation.noise_lambda", dest="perturbation_noise_lambda")
    parser.add_argument("--perturbation.m_tail", dest="perturbation_m_tail")
    parser.add_argument("--perturbation.save_samples", dest="perturbation_save_samples", type=int, default=0)
    parser.add_argument(
        "--perturbation.vllm_force_feature_wrapper",
        dest="perturbation_vllm_force_feature_wrapper",
        action="store_true",
        help="Register the vLLM feature wrapper even for type=none; used for clean equivalence checks.",
    )


def expand_eval_runs(args: Namespace) -> list[Namespace]:
    seeds = _parse_csv(args.seeds, int) if getattr(args, "seeds", None) else [int(args.seed)]
    configs = expand_perturbation_configs(args)
    if len(seeds) == 1 and len(configs) == 1:
        cloned = copy.copy(args)
        cloned.seed = seeds[0]
        cloned.perturbation = configs[0]
        cloned.perturbation_seed = seeds[0]
        cloned.perturbation_run_slug = _run_slug(configs[0], seeds[0])
        return [cloned]

    runs = []
    base_output = Path(args.output_dir)
    for config, seed in itertools.product(configs, seeds):
        cloned = copy.copy(args)
        cloned.seed = seed
        cloned.perturbation = config
        cloned.perturbation_seed = seed
        cloned.perturbation_run_slug = _run_slug(config, seed)
        cloned.output_dir = str(base_output / cloned.perturbation_run_slug)
        runs.append(cloned)
    return runs


def expand_perturbation_configs(args: Namespace) -> list[PerturbationConfig]:
    raw_type = getattr(args, "perturbation_type", "none") or "none"
    types = _parse_csv(raw_type, str)
    configs = []
    for perturb_type in types:
        perturb_type = perturb_type.strip()
        if perturb_type in {"", "none", "clean"}:
            configs.append(PerturbationConfig())
            continue
        if perturb_type in {"cgpo_flat", "cgpo_hierarchical", "no_image", "mask_visual_attention"}:
            raise ValueError(f"{perturb_type!r} is a training auxiliary view, not an eval pixel perturbation.")
        if perturb_type not in PIXEL_PERTURBATIONS | FEATURE_PERTURBATIONS:
            raise ValueError(f"Unknown perturbation.type: {perturb_type!r}")
        param_grid = _param_grid_for_type(args, perturb_type)
        for params in param_grid:
            _validate_params(perturb_type, params)
            configs.append(PerturbationConfig(type=perturb_type, params=params))
    return configs


def apply_pixel_perturbation(
    image: Image.Image,
    *,
    config: PerturbationConfig,
    global_seed: int,
    sample_id: str,
    image_idx: int,
) -> tuple[Image.Image, dict[str, Any]]:
    if not config.is_pixel:
        return image, {}

    seed = derive_seed(global_seed, sample_id, image_idx, "pixel")
    params = dict(config.params)
    diagnostics: dict[str, Any] = {
        "type": config.type,
        "seed": seed,
        "image_idx": image_idx,
        "sample_id": sample_id,
    }
    if config.type == "random_patch":
        kwargs = {"patch_size": int(params.get("patch_size", 14)), "black_prob": float(params.get("black_prob", 0.6))}
        return random_patch_blackening(image, seed=seed, **kwargs), {**diagnostics, **kwargs}
    if config.type == "gaussian_noise":
        kwargs = {"std": float(params.get("std", 0.5))}
        return gaussian_noise_image(image, seed=seed, **kwargs), {**diagnostics, **kwargs}
    if config.type == "vp_diffusion":
        noise_t = float(params.get("noise_t_init", 500.0))
        noise_t_max = float(params.get("noise_t_max", 1000.0))
        perturbed, vp_params = vp_diffusion_noise_fixed_t(
            image,
            noise_t=noise_t,
            noise_t_max=noise_t_max,
            seed=seed,
        )
        return perturbed, {
            **diagnostics,
            "noise_t": vp_params.noise_t,
            "noise_beta": vp_params.noise_beta,
            "noise_t_max": noise_t_max,
            "noise_gamma_ignored_for_eval": params.get("noise_gamma", None),
        }
    raise ValueError(f"Unsupported pixel perturbation: {config.type}")


def write_sample_image(
    image: Image.Image,
    *,
    output_dir: Path,
    config: PerturbationConfig,
    sample_id: str,
    image_idx: int,
    max_samples: int,
    current_count: int,
) -> int:
    if max_samples <= 0 or current_count >= max_samples:
        return current_count
    sample_dir = output_dir / "perturbation_samples" / config.summary()
    sample_dir.mkdir(parents=True, exist_ok=True)
    safe_sample_id = slugify(sample_id)[:80]
    image.save(sample_dir / f"{current_count:04d}_{safe_sample_id}_image{image_idx}.png")
    return current_count + 1


def perturbation_metadata(config: PerturbationConfig, seed: int) -> dict[str, Any]:
    return {
        "perturbation_type": config.type,
        "perturbation_params": json.dumps(config.params, ensure_ascii=False, sort_keys=True),
        "perturbation_seed": seed,
        "perturbation_summary": config.summary(),
    }


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(text).strip())
    text = text.strip("-_.")
    return text or "run"


def _run_slug(config: PerturbationConfig, seed: int) -> str:
    return f"{config.summary()}_seed-{seed}"


def _param_grid_for_type(args: Namespace, perturb_type: str) -> list[dict[str, Any]]:
    if perturb_type == "feature_consistency" and getattr(args, "perturbation_token_p", None) is not None:
        raise ValueError(
            "perturbation.token_p only applies to feature_divergence; feature_consistency perturbs all tokens."
        )
    fields_by_type = {
        "random_patch": {"patch_size": int, "black_prob": float},
        "gaussian_noise": {"std": float},
        "vp_diffusion": {"noise_t_init": float, "noise_gamma": float, "noise_t_max": float},
        "feature_divergence": {
            "k": int,
            "ev_threshold": float,
            "n_skip": int,
            "auto_rogue": _bool,
            "rogue_corr_threshold": float,
            "rho": float,
            "token_p": float,
        },
        "feature_consistency": {
            "k": int,
            "ev_threshold": float,
            "n_skip": int,
            "auto_rogue": _bool,
            "rogue_corr_threshold": float,
            "noise_lambda": float,
            "m_tail": int,
        },
    }[perturb_type]
    keys = []
    values = []
    for name, caster in fields_by_type.items():
        raw = getattr(args, f"perturbation_{name}", None)
        if raw is None:
            continue
        keys.append(name)
        values.append(_parse_csv(raw, caster))
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _validate_params(perturb_type: str, params: dict[str, Any]) -> None:
    if (
        perturb_type in FEATURE_PERTURBATIONS
        and params.get("k") is not None
        and params.get("ev_threshold") is not None
    ):
        raise ValueError("Specify only one of perturbation.k and perturbation.ev_threshold.")
    if perturb_type == "feature_consistency" and "token_p" in params:
        raise ValueError("perturbation.token_p only applies to feature_divergence.")
    if "token_p" in params and not 0.0 <= float(params["token_p"]) <= 1.0:
        raise ValueError("perturbation.token_p must be in [0, 1].")
    if "rho" in params and not 0.0 < float(params["rho"]) <= 1.0:
        raise ValueError("perturbation.rho must be in (0, 1].")
    if "ev_threshold" in params and not 0.0 < float(params["ev_threshold"]) <= 1.0:
        raise ValueError("perturbation.ev_threshold must be in (0, 1].")
    if "m_tail" in params and int(params["m_tail"]) < 0:
        raise ValueError("perturbation.m_tail must be >= 0.")


def _parse_csv(value: Any, caster):
    if isinstance(value, str):
        return [caster(item.strip()) for item in value.split(",") if item.strip()]
    return [caster(value)]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")
