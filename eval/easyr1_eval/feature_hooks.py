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

from dataclasses import dataclass, field
from typing import Any, Callable

from verl.utils.perturbations.feature_views import perturb_visual_features

from .perturbations import PerturbationConfig


@dataclass
class FeaturePerturbationController:
    config: PerturbationConfig
    global_seed: int
    sample_id: str | None = None
    modality: str | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    _cleanup: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def begin_sample(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.modality = None
        self.diagnostics = []

    def end_sample(self) -> list[dict[str, Any]]:
        diagnostics = list(self.diagnostics)
        self.sample_id = None
        self.modality = None
        self.diagnostics = []
        return diagnostics

    def remove(self) -> None:
        for cleanup in reversed(self._cleanup):
            cleanup()
        self._cleanup = []
        self.sample_id = None
        self.modality = None
        self.diagnostics = []


def install_feature_perturbation_hook(
    model, config: PerturbationConfig, global_seed: int
) -> FeaturePerturbationController | None:
    if not config.is_feature:
        return None
    visual = _resolve_visual_module(model)
    if visual is None:
        raise RuntimeError("Feature-space perturbation requires a model.visual module.")
    existing = getattr(visual, "_feature_perturbation_controller", None)
    if existing is not None:
        existing.remove()
    controller = FeaturePerturbationController(config=config, global_seed=global_seed)
    merge_size = _resolve_spatial_merge_size(model, visual)

    def hook(_module, args, kwargs, result):
        return _perturb_visual_result(controller, config, merge_size, args, kwargs, result)

    if hasattr(visual, "register_forward_hook"):
        handle = visual.register_forward_hook(hook, with_kwargs=True)
        controller._cleanup.append(handle.remove)
    else:
        original_visual_forward = visual.forward

        def patched_visual_forward(*args, **kwargs):
            result = original_visual_forward(*args, **kwargs)
            return _perturb_visual_result(controller, config, merge_size, args, kwargs, result)

        visual.forward = patched_visual_forward
        controller._cleanup.append(lambda: setattr(visual, "forward", original_visual_forward))
    setattr(visual, "_feature_perturbation_controller", controller)
    controller._cleanup.append(lambda: _clear_visual_controller(visual, controller))
    _wrap_image_feature_method(model, controller)
    _wrap_video_feature_method(model, controller)
    inner_model = getattr(model, "model", None)
    if inner_model is not None and inner_model is not model:
        _wrap_image_feature_method(inner_model, controller)
        _wrap_video_feature_method(inner_model, controller)
    return controller


def _perturb_visual_result(
    controller: FeaturePerturbationController,
    config: PerturbationConfig,
    merge_size: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result,
):
    if controller.sample_id is None:
        return result
    if controller.modality == "video":
        raise ValueError(
            "Feature-space perturbations currently support image inputs only; video inputs are unsupported."
        )
    image_grid_thw = kwargs.get("grid_thw")
    if image_grid_thw is None and len(args) >= 2:
        image_grid_thw = args[1]
    if image_grid_thw is None:
        raise ValueError("Feature-space perturbation requires grid_thw from the visual tower.")
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Expected model.visual to return (image_embeds, deepstack_image_embeds, ...).")
    main_features, deepstack_features = result[0], result[1]
    main_out, deepstack_out, diagnostics = perturb_visual_features(
        main_features,
        deepstack_features,
        image_grid_thw=image_grid_thw,
        merge_size=merge_size,
        perturbation_type=config.type,
        params=config.params,
        global_seed=controller.global_seed,
        sample_id=controller.sample_id,
    )
    controller.diagnostics.extend(diagnostics)
    return (main_out, deepstack_out, *result[2:])


def _wrap_image_feature_method(target, controller: FeaturePerturbationController) -> None:
    method = getattr(target, "get_image_features", None)
    if method is None:
        return

    def wrapped(*args, **kwargs):
        previous = controller.modality
        controller.modality = "image"
        try:
            return method(*args, **kwargs)
        finally:
            controller.modality = previous

    setattr(target, "get_image_features", wrapped)
    controller._cleanup.append(lambda: setattr(target, "get_image_features", method))


def _wrap_video_feature_method(target, controller: FeaturePerturbationController) -> None:
    method = getattr(target, "get_video_features", None)
    if method is None:
        return

    def wrapped(*args, **kwargs):
        if controller.sample_id is not None:
            raise ValueError(
                "Feature-space perturbations currently support image inputs only; video inputs are unsupported."
            )
        previous = controller.modality
        controller.modality = "video"
        try:
            return method(*args, **kwargs)
        finally:
            controller.modality = previous

    setattr(target, "get_video_features", wrapped)
    controller._cleanup.append(lambda: setattr(target, "get_video_features", method))


def _clear_visual_controller(visual, controller: FeaturePerturbationController) -> None:
    if getattr(visual, "_feature_perturbation_controller", None) is controller:
        delattr(visual, "_feature_perturbation_controller")


def _resolve_visual_module(model):
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        return getattr(inner_model, "visual", None)
    return None


def _resolve_spatial_merge_size(model, visual) -> int:
    for owner in (
        visual,
        getattr(visual, "config", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
    ):
        value = getattr(owner, "spatial_merge_size", None) if owner is not None else None
        if value is not None:
            return int(value)
    return 2
