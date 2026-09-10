"""Deterministic diffusion workloads at multiple output resolutions."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator


DEFAULT_RESOLUTIONS = [128, 192, 256, 320, 384, 512]
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_FRAMES = 17
BASE_SEED = 12345
PROMPTS = [
    "a studio photograph of a ceramic teapot on a plain wooden table",
    "a small red sailboat on a calm lake under soft daylight",
    "a green bicycle beside a brick wall, realistic photography",
    "a bowl of oranges on a white cloth, natural light",
]
STEP_TASKS = {"unconditional-image-generation", "text-to-3d", "image-to-3d"}
VIDEO_TASKS = {"text-to-video", "image-to-video", "image-text-to-video", "video-to-video"}
IMAGE_TASKS = {"image-text-to-image", "image-text-to-video", "image-to-video", "image-to-3d"}
PROMPT_TASKS = {"text-to-image", "image-text-to-image", "image-text-to-video", "image-to-video", "text-to-video", "video-to-video", "text-to-3d"}


class DiffusionWorkloadGenerator(WorkloadGenerator):
    """Keep prompt and denoising settings fixed while scaling output size."""

    def __init__(
        self,
        model_id: str,
        task_type: str,
        batch_size: int,
        workload_spec_path: Optional[str] = None,
    ):
        super().__init__(model_id, task_type, batch_size)
        self._task = {
            "image-to-image": "image-text-to-image",
        }.get(task_type, task_type)
        if self._task not in {
            *PROMPT_TASKS, *IMAGE_TASKS, *STEP_TASKS,
        }:
            raise ValueError(f"unsupported diffusion task_type={task_type!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self._task != "text-to-image" and batch_size != 1:
            raise ValueError(f"{task_type} requires batch_size=1")

        self._prompts = [PROMPTS[index % len(PROMPTS)] for index in range(batch_size)]
        self._prompt_optional = task_type in {"image-to-image", "image-to-video"}
        self._params = {
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "seed": BASE_SEED,
        }
        if self._task in {"unconditional-image-generation", "image-to-video"}:
            self._params.pop("guidance_scale")
        if self._task in VIDEO_TASKS:
            self._params["num_frames"] = DEFAULT_NUM_FRAMES
        self._scales = [1, 2, 4, 8, 16, 20] if self._task in STEP_TASKS else list(DEFAULT_RESOLUTIONS)
        self._workload_id = f"synthetic-{self._task}-v1"
        self._spec_sha256: Optional[str] = None
        self._image_bytes: Optional[bytes] = None
        self._video_bytes: List[bytes] = []
        self._video_sources: List[str] = []
        self._image_source = "deterministic synthetic geometric RGB image"
        if workload_spec_path is not None:
            self._load_spec(Path(workload_spec_path).expanduser().resolve())
        if self._task in IMAGE_TASKS and self._image_bytes is None:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (256, 256), (220, 230, 240))
            draw = ImageDraw.Draw(image)
            draw.rectangle((24, 100, 120, 232), fill=(30, 110, 180))
            draw.ellipse((140, 24, 224, 108), fill=(235, 140, 35))
            draw.polygon([(140, 232), (188, 124), (236, 232)], fill=(55, 150, 80))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            self._image_bytes = buffer.getvalue()
        if self._task == "video-to-video" and not self._video_bytes:
            from PIL import Image, ImageDraw

            for index in range(self._params["num_frames"]):
                image = Image.new("RGB", (256, 256), (220, 230, 240))
                ImageDraw.Draw(image).rectangle(
                    (16 + index % 17 * 8, 64, 72 + index % 17 * 8, 128),
                    fill=(30, 110, 180),
                )
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                self._video_bytes.append(buffer.getvalue())
            self._video_sources = ["deterministic synthetic moving rectangle"]

    def _load_spec(self, path: Path) -> None:
        content = path.read_bytes()
        spec = json.loads(content)
        if not isinstance(spec, dict):
            raise ValueError("diffusion workload spec must be an object")
        allowed = {
            "schema_version", "workload_id", "image_path", "image_sha256",
            "prompt", "input_scales", "params",
            "video_frames", "frame_paths",
        }
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError("unknown diffusion workload keys: " + ", ".join(unknown))
        if spec.get("schema_version", 1) != 1 or isinstance(spec.get("schema_version"), bool):
            raise ValueError("diffusion workload schema_version must be 1")
        self._spec_sha256 = hashlib.sha256(content).hexdigest()
        if "workload_id" in spec:
            if not isinstance(spec["workload_id"], str) or not spec["workload_id"].strip():
                raise ValueError("workload_id must be a non-empty string")
            self._workload_id = spec["workload_id"]
        if "prompt" in spec:
            if self._task not in PROMPT_TASKS:
                raise ValueError(f"{self._task} does not consume prompt")
            prompts = spec["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts] * self.batch_size
            if (
                not isinstance(prompts, list) or len(prompts) != self.batch_size
                or any(not isinstance(item, str) or not item.strip() for item in prompts)
            ):
                raise ValueError("prompt must contain one non-empty string per batch entry")
            self._prompts = prompts
            self._prompt_optional = False
        if "input_scales" in spec:
            scales = spec["input_scales"]
            if not isinstance(scales, list) or not scales:
                raise ValueError("input_scales must be a non-empty resolution array")
            self._scales = [self._scale(value) for value in scales]
            if len(set(self._scales)) != len(self._scales):
                raise ValueError("input_scales must not contain duplicates")
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        allowed_params = self._allowed_params()
        if self._task in STEP_TASKS and "num_inference_steps" in params:
            raise ValueError("use input_scales to set num_inference_steps for this task")
        unknown = sorted(set(params) - allowed_params)
        if unknown:
            raise ValueError("unsupported diffusion params: " + ", ".join(unknown))
        self._params.update(params)
        for name in ("num_inference_steps", "num_frames", "seed", "fps", "decode_chunk_size", "motion_bucket_id"):
            if name in self._params:
                value = self._params[name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")
                if name not in {"seed", "motion_bucket_id"} and value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
                if name == "motion_bucket_id" and value < 0:
                    raise ValueError("motion_bucket_id must be non-negative")
        for name in ("guidance_scale", "strength", "image_guidance_scale", "min_guidance_scale", "max_guidance_scale", "noise_aug_strength"):
            if name in self._params:
                value = self._params[name]
                if (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                ):
                    raise ValueError(f"{name} must be a finite non-negative number")
                if name == "strength" and value > 1:
                    raise ValueError("strength must be between 0 and 1")
        if "strength" in self._params and int(
            self._params["strength"] * self._params["num_inference_steps"]
        ) < 1:
            raise ValueError("strength * num_inference_steps must provide at least one denoising step")
        negative_prompt = self._params.get("negative_prompt")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            if (
                not isinstance(negative_prompt, list)
                or len(negative_prompt) != self.batch_size
                or any(not isinstance(item, str) for item in negative_prompt)
            ):
                raise ValueError("negative_prompt must contain one string per batch entry")
        if "image_sha256" in spec and "image_path" not in spec:
            raise ValueError("image_sha256 requires image_path")
        if "image_path" in spec:
            if self._task not in IMAGE_TASKS:
                raise ValueError(f"{self._task} does not accept image_path")
            if not isinstance(spec["image_path"], str) or not spec["image_path"].strip():
                raise ValueError("image_path must be a non-empty local path")
            image_path = (path.parent / spec["image_path"]).resolve()
            self._image_bytes = image_path.read_bytes()
            digest = hashlib.sha256(self._image_bytes).hexdigest()
            if "image_sha256" in spec and spec["image_sha256"] != digest:
                raise ValueError("condition image SHA256 does not match image_sha256")
            from PIL import Image

            with Image.open(io.BytesIO(self._image_bytes)) as image:
                image.load()
            self._image_source = str(image_path)
        frame_paths = spec.get("video_frames", spec.get("frame_paths"))
        if "video_frames" in spec and "frame_paths" in spec:
            raise ValueError("use only video_frames or frame_paths")
        if frame_paths is not None:
            if self._task != "video-to-video":
                raise ValueError(f"{self._task} does not accept video_frames")
            if not isinstance(frame_paths, list) or not frame_paths or any(
                not isinstance(item, str) or not item.strip() for item in frame_paths
            ):
                raise ValueError("video_frames must be an ordered non-empty list of image paths")
            if "num_frames" in params and params["num_frames"] != len(frame_paths):
                raise ValueError("num_frames must equal the number of video_frames")
            self._params["num_frames"] = len(frame_paths)
            from PIL import Image

            for item in frame_paths:
                frame_path = (path.parent / item).resolve()
                content = frame_path.read_bytes()
                with Image.open(io.BytesIO(content)) as frame:
                    frame.load()
                self._video_bytes.append(content)
                self._video_sources.append(str(frame_path))

    def _allowed_params(self) -> set[str]:
        allowed = {"num_inference_steps", "seed"}
        if self._task != "unconditional-image-generation":
            allowed.add("guidance_scale")
        if self._task in PROMPT_TASKS and self._task not in STEP_TASKS:
            allowed.add("negative_prompt")
        if self._task in {"image-text-to-image", "image-text-to-video"}:
            allowed.update({"strength", "image_guidance_scale"})
        if self._task == "video-to-video":
            allowed.add("strength")
        if self._task == "image-to-video":
            allowed.update({"min_guidance_scale", "max_guidance_scale", "fps", "motion_bucket_id", "noise_aug_strength", "decode_chunk_size"})
        if self._task in VIDEO_TASKS:
            allowed.add("num_frames")
        return allowed

    def _scale(self, value: float) -> int:
        if self._task not in STEP_TASKS:
            return self._resolution(value)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 1 or value != int(value):
            raise ValueError("denoising_steps input scale must be a positive integer")
        return int(value)

    @staticmethod
    def _resolution(scale_value: float) -> int:
        if isinstance(scale_value, bool):
            raise ValueError("diffusion resolution must be an integer number of pixels")
        try:
            scale = float(scale_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("diffusion input scale must be a resolution in pixels") from exc
        if not math.isfinite(scale) or scale != int(scale):
            raise ValueError("diffusion resolution must be an integer number of pixels")
        resolution = int(scale)
        if resolution < 64 or resolution % 8 != 0:
            raise ValueError("diffusion resolution must be at least 64 and divisible by 8")
        return resolution

    def generate(self, scale_value: float) -> Dict[str, Any]:
        resolution = self._scale(scale_value)
        payload = {
            "params": copy.deepcopy(self._params),
        }
        if self._task in PROMPT_TASKS:
            payload["prompt"] = list(self._prompts)
            if self.task_type in {"image-to-image", "image-to-video"}:
                payload["prompt_optional"] = self._prompt_optional
        if self._task in STEP_TASKS:
            payload["input_scale"] = resolution
            payload["params"]["num_inference_steps"] = resolution
            resolution = 256  # Fixed image conditioning size; mesh has no frame_size.
        else:
            payload["resolution"] = resolution
        if self._image_bytes is not None:
            from PIL import Image

            with Image.open(io.BytesIO(self._image_bytes)) as source:
                image = source.convert("RGB").resize(
                    (resolution, resolution), Image.Resampling.LANCZOS
                )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload["image_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        if self._video_bytes:
            from PIL import Image

            payload["frames_base64"] = []
            for content in self._video_bytes:
                with Image.open(io.BytesIO(content)) as source:
                    frame = source.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                payload["frames_base64"].append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return payload

    def scale_label(self, scale_value: float) -> str:
        if self._task in STEP_TASKS:
            return f"steps{self._scale(scale_value)}"
        return f"res{self._scale(scale_value)}px"

    def effective_input_scale(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> float:
        if payload is None:
            return float(self._scale(scale_value))
        return float(self._scale(payload.get("input_scale" if self._task in STEP_TASKS else "resolution")))

    def max_input_scale(self) -> float:
        return float(max(self._scales))

    def default_input_scales(self) -> List[float]:
        return [float(value) for value in self._scales]

    def plan_metadata(self) -> Dict[str, Any]:
        metadata = {
            "workload_id": self._workload_id,
            "source": "custom diffusion workload spec" if self._spec_sha256 else "deterministic synthetic prompts",
            "input_scale_semantics": "square output image side length in pixels",
            "input_scale_type": "resolution_px",
            "resolution_multiple": 8,
            **copy.deepcopy(self._params),
            "prompts": list(self._prompts),
        }
        if self._task not in PROMPT_TASKS:
            metadata.pop("prompts")
            if not self._spec_sha256:
                metadata["source"] = "deterministic seeded noise" if self._task == "unconditional-image-generation" else self._image_source
        if self._task != "text-to-image":
            metadata["batch_size_limit"] = 1
        if self.task_type in {"image-to-image", "image-to-video"}:
            metadata["prompt_optional"] = self._prompt_optional
            metadata["prompt_consumption"] = "default synthetic prompt is used only if the native pipeline explicitly accepts prompt; explicit workload prompts are required"
        if self._task in STEP_TASKS:
            metadata.update({"input_scale_type": "denoising_steps", "input_scale_semantics": "number of denoising scheduler steps"})
            metadata.pop("resolution_multiple")
            metadata.pop("num_inference_steps")
            metadata["output_geometry"] = "native UNet sample_size" if self._task == "unconditional-image-generation" else "native ShapE mesh decoder grid; frame_size is not used"
        if self._task in VIDEO_TASKS:
            metadata["input_scale_semantics"] = "square output video frame side length in pixels; fixed frame count"
        if self._spec_sha256:
            metadata["workload_spec_sha256"] = self._spec_sha256
        if self._image_bytes is not None:
            metadata.update({
                "condition_source": self._image_source,
                "condition_source_sha256": hashlib.sha256(self._image_bytes).hexdigest(),
                "condition_transform": "RGB; square resize with Pillow LANCZOS; PNG",
                "batch_size_limit": 1,
                "pipeline_specific_params": "native pipeline defaults unless explicitly configured",
            })
        if self._video_bytes:
            metadata.update({
                "condition_sources": list(self._video_sources),
                "condition_source_sha256": [hashlib.sha256(frame).hexdigest() for frame in self._video_bytes],
                "condition_transform": "ordered RGB frames; square resize with Pillow LANCZOS; PNG",
                "condition_frame_count": len(self._video_bytes),
            })
        return metadata

    def input_metadata(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        materialized = self.generate(scale_value) if payload is None else payload
        if self._task in STEP_TASKS:
            steps = self._scale(materialized.get("input_scale"))
            metadata = {"input_scale_type": "denoising_steps", "num_inference_steps": steps, "input_num_samples": 1}
            if "image_base64" in materialized:
                content = base64.b64decode(materialized["image_base64"], validate=True)
                metadata.update({"condition_image_sha256": hashlib.sha256(content).hexdigest(), "condition_image_resolution_px": 256})
            return metadata
        resolution = self._resolution(materialized.get("resolution"))
        prompts = materialized.get("prompt")
        prompt_count = len(prompts) if isinstance(prompts, list) else 1
        metadata = {
            "resolution_px": resolution,
            "input_scale_type": "resolution_px",
            "output_pixel_count_per_image": resolution * resolution,
            "prompt_count": prompt_count,
            "input_num_samples": prompt_count,
        }
        if "image_base64" in materialized:
            image_bytes = base64.b64decode(materialized["image_base64"], validate=True)
            metadata["condition_image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        if self._task in VIDEO_TASKS:
            frames = materialized["params"]["num_frames"]
            metadata["output_pixel_count_per_frame"] = metadata.pop("output_pixel_count_per_image")
            metadata["output_num_frames"] = frames
            metadata["output_pixel_count_per_video"] = resolution * resolution * frames
        if "frames_base64" in materialized:
            metadata["condition_frame_count"] = len(materialized["frames_base64"])
            metadata["condition_frame_sha256"] = [hashlib.sha256(base64.b64decode(frame, validate=True)).hexdigest() for frame in materialized["frames_base64"]]
        if self._task not in PROMPT_TASKS:
            metadata["prompt_count"] = 0
        return metadata


register_generator("diffusion", DiffusionWorkloadGenerator)
