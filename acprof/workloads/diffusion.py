"""Deterministic text-to-image workloads at multiple output resolutions."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator


DEFAULT_RESOLUTIONS = [128, 192, 256, 320, 384, 512]
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
BASE_SEED = 12345
PROMPTS = [
    "a studio photograph of a ceramic teapot on a plain wooden table",
    "a small red sailboat on a calm lake under soft daylight",
    "a green bicycle beside a brick wall, realistic photography",
    "a bowl of oranges on a white cloth, natural light",
]


class DiffusionWorkloadGenerator(WorkloadGenerator):
    """Keep prompt and denoising settings fixed while scaling output size."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        if task_type != "text-to-image":
            raise ValueError(
                f"unsupported diffusion task_type={task_type!r}; "
                "only 'text-to-image' is implemented"
            )
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    @staticmethod
    def _resolution(scale_value: float) -> int:
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
        resolution = self._resolution(scale_value)
        prompts = [PROMPTS[index % len(PROMPTS)] for index in range(self.batch_size)]
        return {
            "prompt": prompts,
            "resolution": resolution,
            "params": {
                "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
                "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                "seed": BASE_SEED,
            },
        }

    def scale_label(self, scale_value: float) -> str:
        return f"res{self._resolution(scale_value)}px"

    def effective_input_scale(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> float:
        if payload is None:
            return float(self._resolution(scale_value))
        return float(self._resolution(payload.get("resolution")))

    def max_input_scale(self) -> float:
        return float(max(DEFAULT_RESOLUTIONS))

    def default_input_scales(self) -> List[float]:
        return [float(value) for value in DEFAULT_RESOLUTIONS]

    def plan_metadata(self) -> Dict[str, Any]:
        return {
            "workload_id": "synthetic-text-to-image-v1",
            "source": "deterministic synthetic prompts",
            "input_scale_semantics": "square output image side length in pixels",
            "resolution_multiple": 8,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "seed": BASE_SEED,
            "prompts": [
                PROMPTS[index % len(PROMPTS)] for index in range(self.batch_size)
            ],
        }

    def input_metadata(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        materialized = self.generate(scale_value) if payload is None else payload
        resolution = self._resolution(materialized.get("resolution"))
        prompts = materialized.get("prompt")
        prompt_count = len(prompts) if isinstance(prompts, list) else 1
        return {
            "resolution_px": resolution,
            "output_pixel_count_per_image": resolution * resolution,
            "prompt_count": prompt_count,
            "input_num_samples": prompt_count,
        }


register_generator("diffusion", DiffusionWorkloadGenerator)
