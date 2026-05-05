"""CV workload generator - synthetic images at various resolutions."""

from __future__ import annotations

import base64
import io
import random
from typing import Any, Dict

from acprof.workloads import WorkloadGenerator, register_generator

BASE_SEED = 12345
BASE_RESOLUTION = 224  # Default base resolution for most vision models


class CVWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic RGB images at scaled resolutions."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)
        self._rng = random.Random(BASE_SEED)
        self._base_res = BASE_RESOLUTION

    def _generate_image_base64(self, width: int, height: int) -> str:
        from PIL import Image

        # Generate a synthetic image with deterministic random pixels
        img = Image.new("RGB", (width, height))
        pixels = []
        rng = random.Random(BASE_SEED + width * 1000 + height)
        for _ in range(width * height):
            pixels.append((
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            ))
        img.putdata(pixels)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate(self, scale_value: float) -> Dict[str, Any]:
        resolution = max(1, int(self._base_res * scale_value))
        image_b64 = self._generate_image_base64(resolution, resolution)
        return {
            "image_base64": image_b64,
            "params": {},
        }

    def scale_label(self, scale_value: float) -> str:
        return f"res{scale_value}"


register_generator("cv", CVWorkloadGenerator)
