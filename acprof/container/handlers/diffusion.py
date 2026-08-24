"""Diffusers text-to-image handler."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
)


DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 12345


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _load_eager_attention(pipe: Any, load_options: Optional[Dict[str, Any]]) -> Any:
    """Use Diffusers' explicit eager attention processor for FLOP profiling."""
    options = dict(load_options or {})
    attention_implementation = options.pop("attention_implementation", None)
    if options:
        unsupported = ", ".join(sorted(options))
        raise ValueError(f"unsupported handler load options: {unsupported}")
    if attention_implementation is None:
        return None
    if str(attention_implementation) != "eager":
        raise ValueError(
            "attention_implementation must be 'eager' for compute profiling"
        )

    denoiser = getattr(pipe, "unet", None)
    if denoiser is None or not callable(getattr(denoiser, "set_attn_processor", None)):
        raise ValueError(
            "this Diffusers pipeline cannot expose an eager UNet attention "
            "processor for compute profiling"
        )

    from diffusers.models.attention_processor import AttnProcessor

    denoiser.set_attn_processor(AttnProcessor())
    config = getattr(denoiser, "config", None)
    if config is not None:
        # compute_profile_runner verifies the implementation from the loaded
        # model context before accepting logical FLOP results.
        setattr(config, "_attn_implementation", "eager")
    return denoiser


class DiffusionHandler(BaseHandler):
    """Run text-to-image pipelines without returning image bytes over HTTP."""

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if task_type != "text-to-image":
            raise ValueError(
                f"unsupported diffusion task_type={task_type!r}; "
                "only 'text-to-image' is implemented"
            )
        if backend != "diffusers":
            raise ValueError(
                f"unsupported diffusion backend={backend!r}; expected 'diffusers'"
            )

        import torch
        from diffusers import DiffusionPipeline

        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            model_source,
            **model_revision_kwargs(model_source, model_revision),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        eager_model = _load_eager_attention(pipe, load_options)
        pipe = pipe.to(device)
        set_progress_bar = getattr(pipe, "set_progress_bar_config", None)
        if callable(set_progress_bar):
            set_progress_bar(disable=True)

        model = eager_model or getattr(pipe, "unet", None)
        return {
            "pipeline": pipe,
            "model": model,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        prompt = raw_input.get("prompt")
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list) and prompt and all(
            isinstance(item, str) for item in prompt
        ):
            prompts = list(prompt)
        else:
            raise ValueError("prompt must be a non-empty string or string array")
        if any(not item.strip() for item in prompts):
            raise ValueError("prompt entries must not be empty")

        resolution = _positive_int(raw_input.get("resolution"), "resolution")
        if resolution < 64 or resolution % 8 != 0:
            raise ValueError("resolution must be at least 64 and divisible by 8")

        raw_params = raw_input.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError("params must be an object")
        params = dict(raw_params)
        allowed_params = {
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "negative_prompt",
        }
        unsupported = sorted(set(params) - allowed_params)
        if unsupported:
            raise ValueError(
                "unsupported text-to-image params: " + ", ".join(unsupported)
            )

        steps = _positive_int(
            params.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS),
            "num_inference_steps",
        )
        try:
            guidance_scale = float(
                params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("guidance_scale must be a finite non-negative number") from exc
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be a finite non-negative number")

        raw_seed = params.get("seed", DEFAULT_SEED)
        if isinstance(raw_seed, bool):
            raise ValueError("seed must be an integer")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("seed must be an integer") from exc
        if seed != raw_seed:
            raise ValueError("seed must be an integer")

        negative_prompt = params.get("negative_prompt")
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompts)
            elif isinstance(negative_prompt, list) and all(
                isinstance(item, str) for item in negative_prompt
            ):
                if len(negative_prompt) != len(prompts):
                    raise ValueError(
                        "negative_prompt array length must match prompt array length"
                    )
            else:
                raise ValueError(
                    "negative_prompt must be a string or string array"
                )

        return {
            "prompt": prompts,
            "resolution": resolution,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": negative_prompt,
            "_effective_input_scale": float(resolution),
            "_truncated_by_limit": False,
            "_probe_reason": "diffusion input scale is square output resolution in pixels",
        }

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        import torch

        prompts = processed_input["prompt"]
        device = model_ctx["device"]
        generators = [
            torch.Generator(device=device).manual_seed(processed_input["seed"] + index)
            for index in range(len(prompts))
        ]
        generator: Any = generators[0] if len(generators) == 1 else generators
        kwargs: Dict[str, Any] = {
            "prompt": prompts,
            "height": processed_input["resolution"],
            "width": processed_input["resolution"],
            "num_inference_steps": processed_input["num_inference_steps"],
            "guidance_scale": processed_input["guidance_scale"],
            "generator": generator,
            "output_type": "pil",
            "return_dict": True,
        }
        if processed_input["negative_prompt"] is not None:
            kwargs["negative_prompt"] = processed_input["negative_prompt"]
        return model_ctx["pipeline"](**kwargs)

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        images = getattr(raw_output, "images", None)
        if images is None and isinstance(raw_output, dict):
            images = raw_output.get("images")
        if images is None and isinstance(raw_output, tuple) and raw_output:
            images = raw_output[0]
        if not isinstance(images, (list, tuple)):
            raise ValueError("Diffusers pipeline returned no image collection")

        width = None
        height = None
        if images:
            size = getattr(images[0], "size", None)
            if isinstance(size, tuple) and len(size) == 2:
                width, height = int(size[0]), int(size[1])

        return {
            "task": model_ctx["task_type"],
            "output_type": "image",
            "n_results": len(images),
            "output_length": len(images),
            "image_width": width,
            "image_height": height,
        }

    def get_scale_metadata(
        self,
        model_ctx: Dict[str, Any],
        raw_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "input_scale_type": "resolution_px",
            "resolution_multiple": 8,
            "minimum_resolution_px": 64,
        }


HandlerRegistry.register("diffusion", "diffusers", DiffusionHandler)
