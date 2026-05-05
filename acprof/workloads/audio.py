"""Audio workload generator - synthetic sine wave audio at various durations."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from acprof.workloads import WorkloadGenerator, register_generator

BASE_SEED = 12345
SAMPLE_RATE = 16000  # 16kHz standard for speech models


class AudioWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic audio signals at target durations."""

    def __init__(self, model_id: str, task_type: str, batch_size: int):
        super().__init__(model_id, task_type, batch_size)

    def _generate_audio(self, duration_s: float) -> List[float]:
        n_samples = int(SAMPLE_RATE * duration_s)
        freq = 440.0  # A4 note
        samples = []
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            # Mix of sine waves for a more realistic signal
            val = (
                0.5 * math.sin(2 * math.pi * freq * t) +
                0.3 * math.sin(2 * math.pi * freq * 2 * t) +
                0.2 * math.sin(2 * math.pi * freq * 0.5 * t)
            )
            samples.append(val * 0.5)  # Scale to [-0.5, 0.5]
        return samples

    def generate(self, scale_value: float) -> Dict[str, Any]:
        duration_s = float(scale_value)
        audio_samples = self._generate_audio(duration_s)
        return {
            "audio_samples": audio_samples,
            "sample_rate": SAMPLE_RATE,
            "params": {},
        }

    def scale_label(self, scale_value: float) -> str:
        return f"dur{scale_value}s"


register_generator("audio", AudioWorkloadGenerator)
