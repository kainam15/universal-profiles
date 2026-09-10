"""Offline Diffusers image and video generation with explicit conditions."""

from __future__ import annotations

import base64
import binascii
import inspect
import io
import math
from typing import Any, Dict, NamedTuple, Optional

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
)


DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 12345
DEFAULT_NUM_FRAMES = 17
STEP_TASKS = {"unconditional-image-generation", "text-to-3d", "image-to-3d"}
NATIVE_TASKS = STEP_TASKS | {"text-to-video", "video-to-video", "image-to-video", "image-to-image"}
VIDEO_TASKS = {"image-text-to-video", "text-to-video", "video-to-video", "image-to-video"}


class _ConditionedOutput(NamedTuple):
    result: Any
    resolution: int
    num_frames: Optional[int]


def _canonical_task(task_type: str) -> str:
    return task_type


def _pipeline_parameters(pipe: Any) -> Dict[str, inspect.Parameter]:
    try:
        parameters = inspect.signature(pipe).parameters
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot inspect the native Diffusers pipeline call signature") from exc
    # **kwargs is insufficient evidence that a condition will be consumed.
    return {
        name: parameter for name, parameter in parameters.items()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
        }
    }


def _require_pipeline_parameters(pipe: Any, required: set[str]) -> Dict[str, inspect.Parameter]:
    parameters = _pipeline_parameters(pipe)
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(
            f"{type(pipe).__name__} does not explicitly support required pipeline "
            f"parameters: {', '.join(missing)}; select a native pipeline matching the requested task"
        )
    return parameters


def _conditioned_pipeline_parameters(pipe: Any, task_type: str) -> Dict[str, inspect.Parameter]:
    _reject_fixed_upscaler(pipe)
    required = {
        "prompt", "image", "num_inference_steps", "guidance_scale", "generator",
        "output_type", "return_dict",
    }
    if task_type == "image-text-to-video":
        required.update({"height", "width", "num_frames"})
    return _require_pipeline_parameters(pipe, required)


def _reject_fixed_upscaler(pipe: Any) -> None:
    name = type(pipe).__name__
    if name in {"StableDiffusionUpscalePipeline", "StableDiffusionLatentUpscalePipeline"}:
        raise ValueError(
            f"{name} is a fixed-factor upscaler; fixed-factor upscaling is not yet "
            "supported by the current output resolution input-scale semantics"
        )


def _spatial_multiple(pipe: Any) -> int:
    spatial = getattr(pipe, "vae_scale_factor_spatial", None)
    if not isinstance(spatial, int) or isinstance(spatial, bool) or spatial <= 0:
        spatial = getattr(pipe, "vae_scale_factor", 8)
    if not isinstance(spatial, int) or isinstance(spatial, bool) or spatial <= 0:
        spatial = 8
    config = getattr(getattr(pipe, "transformer", None), "config", None)
    patch_size = getattr(config, "patch_size", 1)
    if isinstance(patch_size, (tuple, list)) and len(patch_size) >= 2:
        patch_sizes = patch_size[-2:]
    else:
        patch_sizes = [patch_size]
    multiple = math.lcm(8, spatial)
    for patch in patch_sizes:
        if isinstance(patch, int) and not isinstance(patch, bool) and patch > 0:
            multiple = math.lcm(multiple, spatial * patch)
    return multiple


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


def _native_pipeline_parameters(pipe: Any, task: str) -> Dict[str, inspect.Parameter]:
    _reject_fixed_upscaler(pipe)
    required = {"num_inference_steps", "generator", "output_type", "return_dict"}
    if task == "unconditional-image-generation":
        required.add("batch_size")
        _native_image_size(pipe)
    elif task in {"text-to-3d", "image-to-3d"}:
        required.update({"guidance_scale", "prompt" if task == "text-to-3d" else "image"})
        if not callable(getattr(getattr(pipe, "shap_e_renderer", None), "decode_to_mesh", None)):
            raise ValueError("3D generation requires a native ShapE mesh pipeline")
    elif task == "image-to-image":
        required.add("image")
    else:
        if task != "video-to-video":
            required.update({"height", "width"})
        if task == "image-to-video":
            required.update({"image", "num_frames"})
        else:
            required.update({"prompt", "guidance_scale"})
            required.add("video" if task == "video-to-video" else "num_frames")
    return _require_pipeline_parameters(pipe, required)


def _pipeline_for_task(pipe: Any, task: str) -> Any:
    """Reuse native components when a checkpoint advertises its text pipeline."""
    if task not in {"image-to-image", "image-text-to-image", "video-to-video"}:
        return pipe
    parameters = _pipeline_parameters(pipe)
    condition = "video" if task == "video-to-video" else "image"
    if condition in parameters:
        return pipe
    if task != "video-to-video":
        import diffusers

        return diffusers.AutoPipelineForImage2Image.from_pipe(pipe)
    name = type(pipe).__name__
    conversion = {
        "CogVideoXPipeline": "CogVideoXVideoToVideoPipeline",
        "WanPipeline": "WanVideoToVideoPipeline",
        "TextToVideoSDPipeline": "VideoToVideoSDPipeline",
    }
    if name == "WanPipeline" and (
        getattr(pipe, "transformer_2", None) is not None
        or getattr(getattr(pipe, "config", None), "expand_timesteps", False)
    ):
        raise ValueError("video-to-video adaptation cannot preserve Wan transformer_2 or expanded timesteps")
    target = conversion.get(name)
    if target is None:
        raise ValueError(f"{name} has no verified native video-to-video pipeline conversion")
    import diffusers

    return getattr(diffusers, target).from_pipe(pipe)


def _native_image_size(pipe: Any) -> tuple[int, int]:
    size = getattr(getattr(getattr(pipe, "unet", None), "config", None), "sample_size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return _positive_int(size[1], "sample_size"), _positive_int(size[0], "sample_size")
    raise ValueError("unconditional generation requires a fixed native UNet sample_size")


def _decode_image(encoded: Any, name: str, resolution: Optional[int] = None) -> Any:
    from PIL import Image

    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"{name} must contain a non-empty base64 image")
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as source:
            image = source.convert("RGB")
        if resolution is not None and image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    except (ValueError, binascii.Error, OSError) as exc:
        raise ValueError(f"{name} must decode to a valid image") from exc
    return image


def _native_preprocess(model_ctx: Dict[str, Any], raw: Dict[str, Any], task: str) -> Dict[str, Any]:
    pipe = model_ctx["pipeline"]
    parameters = _native_pipeline_parameters(pipe, task)
    optional_prompt = task in {"image-to-video", "image-to-image"}
    has_prompt = task in {"text-to-video", "video-to-video", "text-to-3d"} or (optional_prompt and "prompt" in parameters)
    has_image = task in {"image-to-video", "image-to-3d", "image-to-image"}
    allowed_inputs = {"params", "input_scale" if task in STEP_TASKS else "resolution"}
    if has_prompt:
        allowed_inputs.add("prompt")
    if optional_prompt:
        allowed_inputs.update({"prompt", "prompt_optional"})
        if "prompt_optional" in raw and not isinstance(raw["prompt_optional"], bool):
            raise ValueError("prompt_optional must be a boolean")
        if not has_prompt and "prompt" in raw and raw.get("prompt_optional") is not True:
            raise ValueError("native image-only pipeline does not consume the explicit prompt")
    if has_image:
        allowed_inputs.add("image_base64")
    if task == "video-to-video":
        allowed_inputs.add("frames_base64")
    unsupported = sorted(set(raw) - allowed_inputs)
    if unsupported:
        raise ValueError(f"unsupported {task} inputs: " + ", ".join(unsupported))
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    allowed = {"num_inference_steps", "seed"}
    if task != "unconditional-image-generation" and (task != "image-to-video" or "guidance_scale" in parameters):
        allowed.add("guidance_scale")
    if has_prompt and task not in STEP_TASKS:
        allowed.add("negative_prompt")
    if task in VIDEO_TASKS:
        allowed.add("num_frames")
    if task == "video-to-video":
        allowed.add("strength")
    if task == "image-to-image":
        allowed.update({"strength", "image_guidance_scale"})
    if task == "image-to-video":
        allowed.update({"min_guidance_scale", "max_guidance_scale", "fps", "motion_bucket_id", "noise_aug_strength", "decode_chunk_size"})
    unsupported = sorted(set(params) - allowed)
    if unsupported:
        raise ValueError(f"unsupported {task} params: " + ", ".join(unsupported))
    steps = _positive_int(params.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS), "num_inference_steps")
    train_steps = getattr(getattr(getattr(pipe, "scheduler", None), "config", None), "num_train_timesteps", None)
    if isinstance(train_steps, int) and steps > train_steps:
        raise ValueError("num_inference_steps exceeds scheduler num_train_timesteps")
    seed = params.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    kwargs: Dict[str, Any] = {"num_inference_steps": steps, "output_type": "mesh" if task in {"text-to-3d", "image-to-3d"} else "pil", "return_dict": True}
    if has_prompt:
        prompt = raw.get("prompt")
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], str) or not prompt[0].strip():
            raise ValueError(f"{task} requires one non-empty prompt and batch_size=1")
        kwargs["prompt"] = prompt
    if task == "unconditional-image-generation":
        kwargs["batch_size"] = 1
    if task in STEP_TASKS:
        scale = _positive_int(raw.get("input_scale"), "input_scale")
        if scale != steps:
            raise ValueError("input_scale must equal num_inference_steps")
        resolution = None
    else:
        resolution = _positive_int(raw.get("resolution"), "resolution")
        multiple = _spatial_multiple(pipe)
        if resolution < 64 or resolution % multiple:
            raise ValueError(f"resolution must be at least 64 and divisible by pipeline spatial multiple {multiple}")
        scale = resolution
        kwargs.update({name: resolution for name in ("height", "width") if name in parameters})
    if has_image:
        kwargs["image"] = _decode_image(raw.get("image_base64"), "image_base64", resolution)
    frames = None
    if task in VIDEO_TASKS:
        frames = _positive_int(params.get("num_frames", DEFAULT_NUM_FRAMES), "num_frames")
        temporal = getattr(pipe, "vae_scale_factor_temporal", None)
        if isinstance(temporal, int) and temporal > 1 and (frames - 1) % temporal:
            raise ValueError(f"num_frames must be 1 + a multiple of pipeline temporal factor {temporal}")
        if task == "video-to-video":
            encoded = raw.get("frames_base64")
            if not isinstance(encoded, list) or len(encoded) != frames:
                raise ValueError("frames_base64 must contain exactly num_frames ordered images")
            kwargs["video"] = [_decode_image(item, "frames_base64", resolution) for item in encoded]
            if "num_frames" in parameters:
                kwargs["num_frames"] = frames
        else:
            kwargs["num_frames"] = frames
    for name, value in params.items():
        if name in {"seed", "num_inference_steps", "num_frames"}:
            continue
        _require_pipeline_parameters(pipe, {name})
        if name == "negative_prompt":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
                raise ValueError("negative_prompt must contain one string")
        elif name in {"fps", "motion_bucket_id", "decode_chunk_size"}:
            if name == "motion_bucket_id" and value == 0 and not isinstance(value, bool):
                value = 0
            else:
                value = _positive_int(value, name)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            if name == "strength" and value > 1:
                raise ValueError("strength must be between 0 and 1")
        kwargs[name] = value
    strength_parameter = parameters.get("strength")
    strength = kwargs.get("strength", strength_parameter.default if strength_parameter is not None else None)
    if isinstance(strength, (float, int)) and int(steps * strength) < 1:
        raise ValueError("strength * num_inference_steps must provide at least one denoising step")
    supplied = set(kwargs) | {"generator"}
    missing = [name for name, value in parameters.items() if value.default is inspect.Parameter.empty and name not in supplied]
    if missing:
        raise ValueError("unsupported required pipeline parameters: " + ", ".join(missing))
    return {"native_kwargs": kwargs, "seed": seed, "resolution": resolution, "num_frames": frames,
            "_effective_input_scale": float(scale), "_truncated_by_limit": False,
            "_probe_reason": "diffusion input scale is denoising steps" if task in STEP_TASKS else "diffusion input scale is square output video resolution in pixels"}


class DiffusionHandler(BaseHandler):
    """Return generation metadata without encoding output media over HTTP."""

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = _canonical_task(task_type)
        if task not in {"text-to-image", "image-text-to-image", "image-text-to-video", *NATIVE_TASKS}:
            raise ValueError(f"unsupported diffusion task_type={task_type!r}")
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
        pipe = _pipeline_for_task(pipe, task)
        if task in NATIVE_TASKS:
            _native_pipeline_parameters(pipe, task)
        elif task != "text-to-image":
            _conditioned_pipeline_parameters(pipe, task)
        eager_model = _load_eager_attention(pipe, load_options)
        pipe = pipe.to(device)
        set_progress_bar = getattr(pipe, "set_progress_bar_config", None)
        if callable(set_progress_bar):
            set_progress_bar(disable=True)

        model = eager_model or getattr(pipe, "unet", None) or getattr(pipe, "transformer", None) or getattr(pipe, "prior", None)
        return {
            "pipeline": pipe,
            "model": model,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        task = _canonical_task(model_ctx.get("task_type", "text-to-image"))
        if task in NATIVE_TASKS:
            return _native_preprocess(model_ctx, raw_input, task)
        conditioned = task in {"image-text-to-image", "image-text-to-video"}
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
        if conditioned and len(prompts) != 1:
            raise ValueError(f"{task} requires one prompt and batch_size=1")

        resolution = _positive_int(raw_input.get("resolution"), "resolution")
        if resolution < 64 or resolution % 8 != 0:
            raise ValueError("resolution must be at least 64 and divisible by 8")
        pipe = model_ctx.get("pipeline")
        parameters: Dict[str, inspect.Parameter] = {}
        if conditioned:
            parameters = _conditioned_pipeline_parameters(pipe, task)
            multiple = _spatial_multiple(pipe)
            if resolution % multiple:
                raise ValueError(f"resolution must be divisible by pipeline spatial multiple {multiple}")

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
        if conditioned:
            allowed_params.update({"strength", "image_guidance_scale"})
        if task == "image-text-to-video":
            allowed_params.add("num_frames")
        unsupported = sorted(set(params) - allowed_params)
        if unsupported:
            raise ValueError(
                f"unsupported {task} params: " + ", ".join(unsupported)
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

        processed = {
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
        if conditioned:
            from PIL import Image, UnidentifiedImageError

            encoded_image = raw_input.get("image_base64")
            if not isinstance(encoded_image, str) or not encoded_image:
                raise ValueError("image_base64 must contain one non-empty base64 image")
            try:
                image_bytes = base64.b64decode(encoded_image, validate=True)
                with Image.open(io.BytesIO(image_bytes)) as source:
                    image = source.convert("RGB")
                if image.size != (resolution, resolution):
                    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            except (ValueError, binascii.Error, OSError, UnidentifiedImageError) as exc:
                raise ValueError("image_base64 must decode to a valid image") from exc
            processed["image"] = image
            extra_params = {}
            for name in ("strength", "image_guidance_scale"):
                if name in params:
                    _require_pipeline_parameters(pipe, {name})
                    try:
                        value = float(params[name])
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(f"{name} must be a finite number") from exc
                    if isinstance(params[name], bool) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"{name} must be a finite non-negative number")
                    if name == "strength" and value > 1:
                        raise ValueError("strength must be between 0 and 1")
                    extra_params[name] = value
            strength_parameter = parameters.get("strength")
            strength = extra_params.get(
                "strength", strength_parameter.default if strength_parameter is not None else None
            )
            if isinstance(strength, (int, float)) and int(steps * strength) < 1:
                raise ValueError("strength * num_inference_steps must provide at least one denoising step")
            if negative_prompt is not None:
                _require_pipeline_parameters(pipe, {"negative_prompt"})
            if task == "image-text-to-video":
                frames = _positive_int(params.get("num_frames", DEFAULT_NUM_FRAMES), "num_frames")
                temporal = getattr(pipe, "vae_scale_factor_temporal", None)
                if isinstance(temporal, int) and temporal > 1 and (frames - 1) % temporal:
                    raise ValueError(
                        f"num_frames must be 1 + a multiple of pipeline temporal factor {temporal}"
                    )
                extra_params["num_frames"] = frames
            processed["pipeline_params"] = extra_params
            processed["pipeline_size_params"] = {
                name: resolution for name in ("height", "width") if name in parameters
            }
            # Check mandatory model-specific arguments now, before model execution.
            supplied = {
                "prompt", "image", "height", "width", "num_inference_steps",
                "guidance_scale", "generator", "output_type", "return_dict",
                *extra_params,
            }
            if negative_prompt is not None:
                supplied.add("negative_prompt")
            missing = [
                name for name, parameter in parameters.items()
                if parameter.default is inspect.Parameter.empty and name not in supplied
            ]
            if missing:
                raise ValueError("unsupported required pipeline parameters: " + ", ".join(missing))
        return processed

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        import torch

        if "native_kwargs" in processed_input:
            kwargs = dict(processed_input["native_kwargs"])
            kwargs["generator"] = torch.Generator(device=model_ctx["device"]).manual_seed(processed_input["seed"])
            output = model_ctx["pipeline"](**kwargs)
            if _canonical_task(model_ctx["task_type"]) in VIDEO_TASKS | {"image-to-image"}:
                return _ConditionedOutput(output, processed_input["resolution"], processed_input["num_frames"])
            return output

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
        task = _canonical_task(model_ctx["task_type"])
        if task in {"image-text-to-image", "image-text-to-video"}:
            kwargs["image"] = processed_input["image"]
            kwargs.update(processed_input["pipeline_params"])
            # Img2Img and InstructPix2Pix infer size from the input image.
            # Video pipelines must explicitly accept both output dimensions.
            for dimension in ("height", "width"):
                kwargs.pop(dimension)
            kwargs.update(processed_input["pipeline_size_params"])
        output = model_ctx["pipeline"](**kwargs)
        if task in {"image-text-to-image", "image-text-to-video"}:
            # Carry the requested geometry to postprocess without retaining
            # mutable per-request state or inspecting frames inside profilers.
            return _ConditionedOutput(
                output, processed_input["resolution"],
                processed_input["pipeline_params"].get("num_frames"),
            )
        return output

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        if isinstance(raw_output, _ConditionedOutput):
            metadata = self.postprocess(model_ctx, raw_output.result)
            media = metadata["output_type"]
            actual_size = (metadata[f"{media}_width"], metadata[f"{media}_height"])
            requested = raw_output.resolution
            if metadata["n_results"] != 1 or actual_size != (requested, requested):
                raise ValueError(
                    f"pipeline output differs from requested one {requested}x{requested} {media}: "
                    f"count={metadata['n_results']}, size={actual_size}"
                )
            if media == "video" and metadata["video_frame_count"] != raw_output.num_frames:
                raise ValueError("pipeline video frame count differs from requested num_frames")
            return metadata
        task = _canonical_task(model_ctx["task_type"])
        if task in {"text-to-3d", "image-to-3d"}:
            meshes = getattr(raw_output, "images", None)
            if meshes is None and isinstance(raw_output, dict):
                meshes = raw_output.get("images")
            if not isinstance(meshes, (list, tuple)) or len(meshes) != 1:
                raise ValueError("ShapE must return one decoded mesh")
            vertices, faces = [], []
            for mesh in meshes:
                for attr, target in (("verts", vertices), ("faces", faces)):
                    shape = getattr(getattr(mesh, attr, None), "shape", None)
                    if shape is None or len(shape) != 2 or shape[1] != 3 or shape[0] < 1:
                        raise ValueError("ShapE returned an empty or invalid mesh")
                    target.append(int(shape[0]))
            return {"task": model_ctx["task_type"], "output_type": "mesh", "n_results": len(meshes),
                    "output_length": len(meshes), "mesh_vertex_count": sum(vertices), "mesh_face_count": sum(faces),
                    "mesh_vertex_counts": vertices, "mesh_face_counts": faces}
        if task in VIDEO_TASKS:
            frames = getattr(raw_output, "frames", None)
            if frames is None and isinstance(raw_output, dict):
                frames = raw_output.get("frames")
            if frames is None and isinstance(raw_output, tuple) and raw_output:
                frames = raw_output[0]
            if not isinstance(frames, (list, tuple)) or not frames:
                raise ValueError("Diffusers pipeline returned no decoded video frame collection")
            frame_counts = []
            size = None
            for video in frames:
                if not isinstance(video, (list, tuple)) or not video:
                    raise ValueError("expected video output as non-empty batches of PIL frames")
                frame_counts.append(len(video))
                for frame in video:
                    frame_size = getattr(frame, "size", None)
                    if not isinstance(frame_size, tuple) or len(frame_size) != 2:
                        raise ValueError("expected decoded PIL video frames")
                    if size is not None and frame_size != size:
                        raise ValueError("video frames must have consistent dimensions")
                    size = frame_size
            if len(set(frame_counts)) != 1:
                raise ValueError("video batches must have consistent frame counts")
            width, height = int(size[0]), int(size[1])
            return {
                "task": model_ctx["task_type"],
                "output_type": "video",
                "n_results": len(frames),
                "output_length": sum(frame_counts),
                "video_frame_count": frame_counts[0],
                "video_width": width,
                "video_height": height,
                "output_shape": [len(frames), frame_counts[0], height, width, 3],
            }
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
        if task == "unconditional-image-generation":
            expected = _native_image_size(model_ctx["pipeline"])
            if len(images) != 1 or (width, height) != expected:
                raise ValueError("unconditional output differs from native UNet sample_size")

        return {
            "task": model_ctx["task_type"],
            "output_type": "image",
            "n_results": len(images),
            "output_length": len(images),
            "image_width": width,
            "image_height": height,
            "output_shape": [len(images), height, width, 3],
        }

    def get_scale_metadata(
        self,
        model_ctx: Dict[str, Any],
        raw_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        task = _canonical_task(model_ctx.get("task_type", "text-to-image"))
        if task in STEP_TASKS:
            metadata: Dict[str, Any] = {"input_scale_type": "denoising_steps", "minimum_denoising_steps": 1}
            if task == "unconditional-image-generation":
                width, height = _native_image_size(model_ctx["pipeline"])
                metadata.update({"native_output_width": width, "native_output_height": height})
            else:
                metadata["output_type"] = "mesh"
                metadata["frame_size_applicable"] = False
            return metadata
        metadata = {
            "input_scale_type": "resolution_px",
            "resolution_multiple": _spatial_multiple(model_ctx.get("pipeline")),
            "minimum_resolution_px": 64,
        }
        if task in {"image-to-image", "image-to-video"}:
            metadata["prompt_consumed"] = "prompt" in _pipeline_parameters(model_ctx.get("pipeline"))
        return metadata


HandlerRegistry.register("diffusion", "diffusers", DiffusionHandler)
