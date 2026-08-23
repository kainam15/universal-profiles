"""Manifest-backed audio workload generation.

Audio inputs are derived from one checked-in, real-speech WAV asset.  Every
scale is a prefix of the same waveform so duration is the only changing input
dimension.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import wave

from acprof.workloads import WorkloadGenerator, register_generator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
ASR_TASK_TYPE = "automatic-speech-recognition"
SHORT_FORM_MAX_DURATION_S = 30.0
DEFAULT_WORKLOAD_ID = "librispeech-clean-test-en-transcribe-short-v1"
DEFAULT_WORKLOAD_SPEC = (
    PROJECT_ROOT
    / "assets"
    / "audio"
    / "librispeech-clean-test-en-30s"
    / "source.json"
)


def _portable_workload_spec_path(path: Path) -> str:
    """Use a stable project-relative path for checked-in workload specs."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Explicit custom specs may live outside the project. Preserve their
        # resolved location because there is no truthful repository-relative
        # representation for them.
        return str(path)


_TOP_LEVEL_KEYS = {
    "schema_version",
    "workload_id",
    "task_family",
    "pipeline_tag",
    "asset",
    "input_scale",
    "inference",
    "provenance",
}
_ASSET_KEYS = {
    "path",
    "sha256",
    "format",
    "pcm_subtype",
    "sample_rate",
    "channels",
    "num_samples",
    "duration_s",
}
_INPUT_SCALE_KEYS = {"type", "values", "construction"}
_INFERENCE_KEYS = {
    "mode",
    "asr_task",
    "language",
    "return_timestamps",
    "pipeline_kwargs",
}
_BUILTIN_PROVENANCE_REQUIRED_KEYS = {
    "dataset_id",
    "dataset_revision",
    "config",
    "split",
    "dataset_url",
    "license",
    "license_url",
    "sources",
    "transform",
}


def _require_object(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str] | set[str],
    location: str,
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"invalid {location} ({'; '.join(details)})")


def _require_keys(
    value: Mapping[str, Any],
    required: Sequence[str] | set[str],
    location: str,
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"invalid {location} (missing keys: {', '.join(missing)})")


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _require_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _require_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_nonempty_string(value, location).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{location} must be a 64-character hexadecimal SHA256")
    return digest


def _validate_provenance(spec: Mapping[str, Any], asset: Mapping[str, Any]) -> None:
    """Validate generic provenance, plus stronger invariants for the built-in asset."""
    provenance = _require_object(spec["provenance"], "provenance")
    _require_keys(provenance, {"license", "sources", "transform"}, "provenance")
    _require_nonempty_string(provenance["license"], "provenance.license")

    sources = provenance["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("provenance.sources must be a non-empty array")
    normalized_sources = []
    for index, source_value in enumerate(sources):
        source = _require_object(source_value, f"provenance.sources[{index}]")
        if not source:
            raise ValueError(f"provenance.sources[{index}] must not be empty")
        normalized_sources.append(source)
    transform = _require_object(provenance["transform"], "provenance.transform")
    if not transform:
        raise ValueError("provenance.transform must not be empty")

    # The bundled LibriSpeech workload has a pinned, auditable construction.
    # Custom workloads retain arbitrary provenance while the derived WAV itself
    # is still strictly verified below by SHA256, PCM format and duration.
    if spec["workload_id"] != DEFAULT_WORKLOAD_ID:
        return

    _require_keys(
        provenance,
        _BUILTIN_PROVENANCE_REQUIRED_KEYS,
        "built-in provenance",
    )
    for key in (
        "dataset_id",
        "dataset_revision",
        "config",
        "split",
        "dataset_url",
        "license_url",
    ):
        _require_nonempty_string(provenance[key], f"provenance.{key}")
    for index, source in enumerate(normalized_sources):
        _require_keys(
            source,
            {"id", "filename", "sha256", "text"},
            f"provenance.sources[{index}]",
        )
        _require_nonempty_string(source["id"], f"provenance.sources[{index}].id")
        _require_nonempty_string(
            source["filename"],
            f"provenance.sources[{index}].filename",
        )
        _require_sha256(source["sha256"], f"provenance.sources[{index}].sha256")
        if not isinstance(source["text"], str):
            raise ValueError(f"provenance.sources[{index}].text must be a string")
    _require_keys(
        transform,
        {
            "operation",
            "source_order",
            "crop_start_sample",
            "output_num_samples",
            "output_duration_s",
            "gain_applied",
            "normalized",
            "resampled",
        },
        "provenance.transform",
    )
    if transform["operation"] != "concatenate_then_prefix_crop":
        raise ValueError(
            "built-in provenance.transform.operation must be "
            "'concatenate_then_prefix_crop'"
        )
    source_order = transform["source_order"]
    if not isinstance(source_order, list) or not source_order:
        raise ValueError("provenance.transform.source_order must be a non-empty array")
    if source_order != [source["id"] for source in normalized_sources]:
        raise ValueError("provenance.transform.source_order must match provenance.sources")
    if _require_int(
        transform["crop_start_sample"],
        "provenance.transform.crop_start_sample",
    ) != 0:
        raise ValueError(
            "built-in provenance.transform.crop_start_sample must be 0"
        )
    if _require_int(
        transform["output_num_samples"],
        "provenance.transform.output_num_samples",
    ) != asset["num_samples"]:
        raise ValueError("provenance.transform.output_num_samples must match asset.num_samples")
    transform_duration = _require_number(
        transform["output_duration_s"],
        "provenance.transform.output_duration_s",
    )
    if not math.isclose(transform_duration, float(asset["duration_s"]), abs_tol=1e-9):
        raise ValueError("provenance.transform.output_duration_s must match asset.duration_s")
    for field_name in ("gain_applied", "normalized", "resampled"):
        if transform[field_name] is not False:
            raise ValueError(
                f"built-in provenance.transform.{field_name} must be false"
            )


def _load_workload_spec(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"audio workload spec does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read audio workload spec {path}: {exc}") from exc

    spec = _require_object(raw, "workload spec")
    _require_exact_keys(spec, _TOP_LEVEL_KEYS, "workload spec")
    if _require_int(spec["schema_version"], "schema_version") != 1:
        raise ValueError(
            f"unsupported audio workload schema_version={spec['schema_version']!r}; expected 1"
        )
    _require_nonempty_string(spec["workload_id"], "workload_id")
    if spec["task_family"] != "audio":
        raise ValueError("task_family must be 'audio'")
    _require_nonempty_string(spec["pipeline_tag"], "pipeline_tag")

    asset = _require_object(spec["asset"], "asset")
    _require_exact_keys(asset, _ASSET_KEYS, "asset")
    _require_nonempty_string(asset["path"], "asset.path")
    _require_sha256(asset["sha256"], "asset.sha256")
    if asset["format"] != "wav":
        raise ValueError("asset.format must be 'wav'")
    if asset["pcm_subtype"] != "PCM_16":
        raise ValueError("asset.pcm_subtype must be 'PCM_16'")
    if _require_int(asset["sample_rate"], "asset.sample_rate") != SAMPLE_RATE:
        raise ValueError(f"asset.sample_rate must be {SAMPLE_RATE}")
    if _require_int(asset["channels"], "asset.channels") != CHANNELS:
        raise ValueError("asset.channels must be 1")
    if _require_int(asset["num_samples"], "asset.num_samples") <= 0:
        raise ValueError("asset.num_samples must be positive")
    if _require_number(asset["duration_s"], "asset.duration_s") <= 0:
        raise ValueError("asset.duration_s must be positive")

    input_scale = _require_object(spec["input_scale"], "input_scale")
    _require_exact_keys(input_scale, _INPUT_SCALE_KEYS, "input_scale")
    if input_scale["type"] != "duration_s":
        raise ValueError("input_scale.type must be 'duration_s'")
    if input_scale["construction"] != "prefix":
        raise ValueError("input_scale.construction must be 'prefix'")
    values = input_scale["values"]
    if not isinstance(values, list) or not values:
        raise ValueError("input_scale.values must be a non-empty array")
    normalized_values = [
        _require_number(value, f"input_scale.values[{index}]")
        for index, value in enumerate(values)
    ]
    if any(value <= 0 for value in normalized_values):
        raise ValueError("input_scale.values must all be positive")
    if any(right <= left for left, right in zip(normalized_values, normalized_values[1:])):
        raise ValueError("input_scale.values must be strictly increasing")

    inference = _require_object(spec["inference"], "inference")
    _require_exact_keys(inference, _INFERENCE_KEYS, "inference")
    if inference["mode"] != "short_form":
        raise ValueError("schema v1 only supports inference.mode='short_form'")
    if inference["asr_task"] != "transcribe":
        raise ValueError("schema v1 only supports inference.asr_task='transcribe'")
    _require_nonempty_string(inference["language"], "inference.language")
    if not isinstance(inference["return_timestamps"], bool):
        raise ValueError("inference.return_timestamps must be a boolean")
    if inference["return_timestamps"] is not False:
        raise ValueError("schema v1 requires inference.return_timestamps=false")
    pipeline_kwargs = _require_object(
        inference["pipeline_kwargs"],
        "inference.pipeline_kwargs",
    )
    for field_name in ("chunk_length_s", "stride_length_s"):
        if pipeline_kwargs.get(field_name) is not None:
            raise ValueError(
                f"inference.pipeline_kwargs.{field_name} must be null or omitted "
                "for short_form; create a separate long-form workload"
            )

    _validate_provenance(spec, asset)

    return spec


def _read_pcm16_wav(wav_bytes: bytes, location: str) -> tuple[bytes, int, int]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            num_samples = wav_file.getnframes()
            compression = wav_file.getcomptype()
            pcm_bytes = wav_file.readframes(num_samples)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"{location} is not a valid WAV file: {exc}") from exc

    if compression != "NONE":
        raise ValueError(f"{location} must be uncompressed PCM WAV")
    if sample_width != SAMPLE_WIDTH_BYTES:
        raise ValueError(f"{location} must use 16-bit PCM samples")
    if channels != CHANNELS:
        raise ValueError(f"{location} must be mono")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{location} must use a {SAMPLE_RATE} Hz sample rate")
    expected_bytes = num_samples * CHANNELS * SAMPLE_WIDTH_BYTES
    if len(pcm_bytes) != expected_bytes:
        raise ValueError(
            f"{location} contains {len(pcm_bytes)} PCM bytes; expected {expected_bytes}"
        )
    return pcm_bytes, num_samples, sample_rate


def _encode_pcm16_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


class AudioWorkloadGenerator(WorkloadGenerator):
    """Generate deterministic real-speech audio prefixes from a workload spec."""

    def __init__(
        self,
        model_id: str,
        task_type: str,
        batch_size: int,
        workload_spec_path: Optional[str] = None,
    ):
        super().__init__(model_id, task_type, batch_size)
        if batch_size != 1:
            raise ValueError(
                "audio workloads require batch_size=1; batching repeated audio "
                "would not represent a larger real input"
            )
        if workload_spec_path is None:
            if task_type != ASR_TASK_TYPE:
                raise ValueError(
                    f"audio task '{task_type}' requires an explicit workload spec; "
                    "the built-in workload is only for automatic speech recognition"
                )
            spec_path = DEFAULT_WORKLOAD_SPEC
        else:
            spec_path = Path(workload_spec_path).expanduser()
        self.workload_spec_path = spec_path.resolve()
        self._spec = _load_workload_spec(self.workload_spec_path)
        if self._spec["pipeline_tag"] != task_type:
            raise ValueError(
                f"workload pipeline_tag={self._spec['pipeline_tag']!r} does not match "
                f"requested task_type={task_type!r}"
            )

        asset_path_value = Path(self._spec["asset"]["path"])
        if asset_path_value.is_absolute():
            raise ValueError("asset.path must be relative to the workload spec")
        spec_directory = self.workload_spec_path.parent.resolve()
        self.asset_path = (spec_directory / asset_path_value).resolve()
        try:
            self.asset_path.relative_to(spec_directory)
        except ValueError as exc:
            raise ValueError("asset.path must stay within the workload spec directory") from exc
        try:
            asset_bytes = self.asset_path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"audio asset does not exist: {self.asset_path}") from exc
        except OSError as exc:
            raise ValueError(f"cannot read audio asset {self.asset_path}: {exc}") from exc

        asset = self._spec["asset"]
        actual_sha256 = hashlib.sha256(asset_bytes).hexdigest()
        expected_sha256 = str(asset["sha256"]).lower()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"audio asset SHA256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        pcm_bytes, num_samples, sample_rate = _read_pcm16_wav(
            asset_bytes,
            str(self.asset_path),
        )
        if num_samples != asset["num_samples"]:
            raise ValueError(
                f"audio asset has {num_samples} samples; spec declares "
                f"{asset['num_samples']}"
            )
        actual_duration_s = num_samples / sample_rate
        if not math.isclose(
            actual_duration_s,
            float(asset["duration_s"]),
            abs_tol=0.5 / sample_rate,
        ):
            raise ValueError(
                f"audio asset duration is {actual_duration_s}s; spec declares "
                f"{asset['duration_s']}s"
            )
        self._asset_sha256 = actual_sha256
        self._asset_pcm_bytes = pcm_bytes
        self._asset_num_samples = num_samples
        self._sample_rate = sample_rate
        self._asset_duration_s = actual_duration_s
        self._max_short_form_duration_s = SHORT_FORM_MAX_DURATION_S
        self._default_scales = [
            float(value) for value in self._spec["input_scale"]["values"]
        ]
        for value in self._default_scales:
            self._validate_scale(value)

    def _validate_scale(self, scale_value: float) -> tuple[float, int]:
        if isinstance(scale_value, bool):
            raise ValueError("audio duration must be a finite positive number")
        try:
            duration_s = float(scale_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("audio duration must be a finite positive number") from exc
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("audio duration must be a finite positive number")
        max_duration_s = self.max_input_scale()
        if max_duration_s is not None and duration_s > max_duration_s:
            raise ValueError(
                f"audio duration {duration_s:g}s exceeds the short-form/asset limit "
                f"of {max_duration_s:g}s; create a separate long-form workload "
                "instead"
            )
        num_samples = int(round(duration_s * self._sample_rate))
        if num_samples <= 0:
            raise ValueError("audio duration is shorter than one sample")
        max_samples = min(
            self._asset_num_samples,
            int(round(self._max_short_form_duration_s * self._sample_rate)),
        )
        if num_samples > max_samples:
            raise ValueError(
                f"audio duration resolves to {num_samples} samples, exceeding "
                f"the limit of {max_samples} samples"
            )
        return duration_s, num_samples

    def generate(self, scale_value: float) -> Dict[str, Any]:
        _, num_samples = self._validate_scale(scale_value)
        pcm_byte_count = num_samples * CHANNELS * SAMPLE_WIDTH_BYTES
        prefix_pcm = self._asset_pcm_bytes[:pcm_byte_count]
        wav_bytes = _encode_pcm16_wav(prefix_pcm, self._sample_rate)
        inference = self._spec["inference"]
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "audio_format": "wav",
            "sample_rate": self._sample_rate,
            "params": {
                "mode": inference["mode"],
                "asr_task": inference["asr_task"],
                "language": inference["language"],
                "return_timestamps": inference["return_timestamps"],
                "pipeline_kwargs": copy.deepcopy(inference["pipeline_kwargs"]),
            },
        }

    def scale_label(self, scale_value: float) -> str:
        return f"dur{float(scale_value):g}s"

    def effective_input_scale(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        if payload is not None and "audio_samples" in payload:
            sample_rate = payload.get("sample_rate", SAMPLE_RATE)
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
                raise ValueError("legacy audio payload has an invalid sample_rate")
            if float(sample_rate) <= 0:
                raise ValueError("legacy audio payload sample_rate must be positive")
            samples = payload["audio_samples"]
            if not isinstance(samples, list):
                raise ValueError("legacy audio payload audio_samples must be an array")
            return len(samples) / float(sample_rate)
        metadata = self.input_metadata(scale_value, payload)
        return float(metadata["duration_s"])

    def max_input_scale(self) -> Optional[float]:
        if not hasattr(self, "_asset_duration_s"):
            return SHORT_FORM_MAX_DURATION_S
        return min(self._asset_duration_s, self._max_short_form_duration_s)

    def default_input_scales(self) -> Optional[List[float]]:
        return list(self._default_scales)

    def plan_metadata(self) -> Dict[str, Any]:
        metadata = copy.deepcopy(self._spec)
        metadata["workload_spec_path"] = _portable_workload_spec_path(
            self.workload_spec_path
        )
        metadata["workload_constraints"] = {
            "sample_rate": self._sample_rate,
            "channels": CHANNELS,
            "pcm_subtype": "PCM_16",
            "max_short_form_duration_s": self._max_short_form_duration_s,
        }
        return metadata

    def input_metadata(
        self,
        scale_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        materialized = self.generate(scale_value) if payload is None else payload
        if not isinstance(materialized, dict):
            raise ValueError("audio payload must be a JSON object")
        if materialized.get("audio_format") != "wav":
            raise ValueError("audio payload audio_format must be 'wav'")
        if materialized.get("sample_rate") != self._sample_rate:
            raise ValueError(
                f"audio payload sample_rate must be {self._sample_rate}"
            )
        encoded = materialized.get("audio_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("audio payload audio_base64 must be a non-empty string")
        try:
            wav_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio payload contains invalid base64") from exc
        _, num_samples, sample_rate = _read_pcm16_wav(wav_bytes, "audio payload")
        return {
            "duration_s": num_samples / sample_rate,
            "input_num_samples": num_samples,
            "audio_wav_bytes": len(wav_bytes),
            "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
        }


register_generator("audio", AudioWorkloadGenerator)
