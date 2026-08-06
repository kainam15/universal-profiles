"""Audio task handler - speech recognition, audio classification, etc."""

from __future__ import annotations

import base64
import binascii
import io
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)


_ASR_TASK_TYPES = {
    "automatic-speech-recognition",
    "asr",
    "speech-recognition",
}
_WHISPER_SHORT_FORM_SECONDS = 30.0


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _optional_positive_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _optional_positive_int(value: Any) -> Optional[int]:
    parsed = _optional_positive_number(value)
    if parsed is None:
        return None
    return int(parsed)


class AudioHandler(BaseHandler):

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch
        from transformers import pipeline as hf_pipeline

        device_map = device if device == "cpu" else "auto"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        pipe = hf_pipeline(
            task=task_type,
            model=model_source,
            **model_revision_kwargs(model_source, model_revision),
            **transformers_pipeline_load_kwargs(load_options),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
            "audio_metadata": self._extract_audio_metadata(pipe),
        }

    @staticmethod
    def _extract_audio_metadata(pipe: Any) -> Dict[str, Any]:
        feature_extractor = getattr(pipe, "feature_extractor", None)
        model = getattr(pipe, "model", None)
        config = getattr(model, "config", None)

        sampling_rate = _optional_positive_int(
            getattr(feature_extractor, "sampling_rate", None)
        )
        chunk_length_s = _optional_positive_number(
            getattr(feature_extractor, "chunk_length", None)
        )
        model_input_num_samples = _optional_positive_int(
            getattr(feature_extractor, "n_samples", None)
        )
        model_input_frames = _optional_positive_int(
            getattr(feature_extractor, "nb_max_frames", None)
        )
        hop_length = _optional_positive_int(
            getattr(feature_extractor, "hop_length", None)
        )

        if model_input_num_samples is None and sampling_rate and chunk_length_s:
            model_input_num_samples = int(round(sampling_rate * chunk_length_s))
        if chunk_length_s is None and sampling_rate and model_input_num_samples:
            chunk_length_s = model_input_num_samples / sampling_rate
        if model_input_frames is None and model_input_num_samples and hop_length:
            model_input_frames = model_input_num_samples // hop_length

        model_type = str(getattr(config, "model_type", "") or "").lower()
        if model_type == "whisper" and chunk_length_s is None:
            chunk_length_s = _WHISPER_SHORT_FORM_SECONDS
        frontend_feature_bins = _optional_positive_int(
            getattr(config, "num_mel_bins", None)
        ) or _optional_positive_int(getattr(feature_extractor, "feature_size", None))
        short_form_fixed_padding = bool(
            model_type == "whisper"
            and model_input_num_samples is not None
            and model_input_frames is not None
        )

        return {
            "sampling_rate": sampling_rate,
            "max_short_form_duration_s": chunk_length_s,
            "model_input_num_samples": model_input_num_samples,
            "model_input_frames": model_input_frames,
            "short_form_fixed_padding": short_form_fixed_padding,
            "fixed_frontend_num_samples": (
                model_input_num_samples if short_form_fixed_padding else None
            ),
            "fixed_frontend_num_frames": (
                model_input_frames if short_form_fixed_padding else None
            ),
            "frontend_feature_bins": frontend_feature_bins,
            "encoder_positions": _optional_positive_int(
                getattr(config, "max_source_positions", None)
            ),
            "decoder_output_token_limit": _optional_positive_int(
                getattr(config, "max_target_positions", None)
            ),
            "model_type": model_type or None,
        }

    def _audio_metadata(self, model_ctx: Dict[str, Any]) -> Dict[str, Any]:
        metadata = model_ctx.get("audio_metadata")
        if isinstance(metadata, dict):
            return metadata
        return self._extract_audio_metadata(model_ctx.get("pipeline"))

    @staticmethod
    def _decode_wav(audio_base64: Any) -> Tuple[np.ndarray, int]:
        if not isinstance(audio_base64, str) or not audio_base64.strip():
            raise ValueError("audio_base64 must be a non-empty Base64 string")
        try:
            wav_bytes = base64.b64decode(audio_base64.strip(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio_base64 is not valid Base64") from exc
        if not wav_bytes:
            raise ValueError("audio_base64 decodes to an empty payload")

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                compression = wav_file.getcomptype()
                pcm_bytes = wav_file.readframes(frame_count)
        except (EOFError, wave.Error) as exc:
            raise ValueError("audio_base64 must contain a valid PCM WAV file") from exc

        if channels != 1:
            raise ValueError(f"WAV must be mono, got {channels} channels")
        if sample_width != 2:
            raise ValueError(
                f"WAV must use signed 16-bit PCM samples, got {sample_width * 8}-bit"
            )
        if compression != "NONE":
            raise ValueError("WAV must use uncompressed PCM encoding")
        if frame_count <= 0 or not pcm_bytes:
            raise ValueError("WAV must contain at least one audio sample")
        if len(pcm_bytes) != frame_count * sample_width:
            raise ValueError("WAV PCM payload is truncated")

        pcm = np.frombuffer(pcm_bytes, dtype="<i2")
        audio_array = pcm.astype(np.float32) / 32768.0
        return audio_array, int(sample_rate)

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        has_base64 = "audio_base64" in raw_input
        has_legacy_samples = "audio_samples" in raw_input
        if has_base64 and has_legacy_samples:
            raise ValueError("provide exactly one of audio_base64 or audio_samples")
        if not has_base64 and not has_legacy_samples:
            raise ValueError("missing audio input: expected audio_base64 or audio_samples")

        metadata = self._audio_metadata(model_ctx)
        required_sample_rate = _optional_positive_int(metadata.get("sampling_rate"))

        if has_base64:
            audio_format = raw_input.get("audio_format")
            if audio_format != "wav":
                raise ValueError("audio_format must be 'wav' for audio_base64 input")
            if "sample_rate" not in raw_input:
                raise ValueError("sample_rate is required for audio_base64 input")
            declared_sample_rate = _positive_int(
                raw_input.get("sample_rate"), "sample_rate"
            )
            audio_array, wav_sample_rate = self._decode_wav(
                raw_input.get("audio_base64")
            )
            if declared_sample_rate != wav_sample_rate:
                raise ValueError(
                    "sample_rate does not match the WAV header "
                    f"({declared_sample_rate} != {wav_sample_rate})"
                )
            sample_rate = wav_sample_rate
        else:
            sample_rate = _positive_int(
                raw_input.get("sample_rate", required_sample_rate or 16000),
                "sample_rate",
            )
            try:
                audio_array = np.asarray(raw_input.get("audio_samples"), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("audio_samples must be a one-dimensional numeric array") from exc
            if audio_array.ndim != 1:
                raise ValueError("audio_samples must be a one-dimensional numeric array")
            if audio_array.size == 0:
                raise ValueError("audio_samples must contain at least one sample")
            if not np.all(np.isfinite(audio_array)):
                raise ValueError("audio_samples must contain only finite values")

        if required_sample_rate and sample_rate != required_sample_rate:
            raise ValueError(
                "audio sample rate does not match the model feature extractor "
                f"({sample_rate} != {required_sample_rate})"
            )

        input_num_samples = int(audio_array.size)
        duration_s = input_num_samples / sample_rate
        max_short_form_duration_s = _optional_positive_number(
            metadata.get("max_short_form_duration_s")
        )
        exceeds_short_form_limit = bool(
            max_short_form_duration_s is not None
            and duration_s > max_short_form_duration_s
        )
        if exceeds_short_form_limit:
            reason = (
                f"audio duration {duration_s:.6g}s exceeds the model short-form limit "
                f"of {max_short_form_duration_s:.6g}s; use a separate long-form workload"
            )
        elif max_short_form_duration_s is not None:
            reason = (
                f"audio duration {duration_s:.6g}s is within the model short-form "
                f"limit of {max_short_form_duration_s:.6g}s"
            )
        else:
            reason = "model does not expose a short-form duration limit"

        return {
            "audio": audio_array,
            "sample_rate": sample_rate,
            "params": raw_input.get("params", {}),
            "_effective_input_scale": duration_s,
            "_input_num_samples": input_num_samples,
            "_duration_s": duration_s,
            "_truncated_by_limit": exceeds_short_form_limit,
            "_probe_reason": reason,
        }

    def get_scale_metadata(
        self,
        model_ctx: Dict[str, Any],
        raw_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        del raw_input
        metadata = self._audio_metadata(model_ctx)
        max_duration = _optional_positive_number(
            metadata.get("max_short_form_duration_s")
        )
        model_type = metadata.get("model_type")
        decoder_limit = metadata.get("decoder_output_token_limit")

        if model_type == "whisper":
            fixed_samples = metadata.get("fixed_frontend_num_samples")
            fixed_frames = metadata.get("fixed_frontend_num_frames")
            fixed_padding = metadata.get("short_form_fixed_padding") is True
            fixed_frontend = ""
            if fixed_padding and fixed_samples and fixed_frames:
                fixed_frontend = (
                    " The feature extractor pads every accepted short-form "
                    f"waveform to a fixed {fixed_samples}-sample / "
                    f"{fixed_frames}-frame frontend input before encoding."
                )
            reason = (
                "Whisper short-form audio uses a fixed receptive field of up to "
                f"{max_duration or _WHISPER_SHORT_FORM_SECONDS:g} seconds. "
                "decoder_output_token_limit is an output-token limit, not an "
                f"audio input-length limit.{fixed_frontend}"
            )
        else:
            reason = "audio input scale is duration in seconds"
            if decoder_limit is not None:
                reason += (
                    "; decoder_output_token_limit describes generated output, "
                    "not audio input duration"
                )

        return {
            "input_scale_type": "duration_s",
            "required_sampling_rate": metadata.get("sampling_rate"),
            "max_short_form_duration_s": max_duration,
            "max_effective_input_scale": max_duration,
            "model_input_num_samples": metadata.get("model_input_num_samples"),
            "model_input_frames": metadata.get("model_input_frames"),
            "short_form_fixed_padding": metadata.get("short_form_fixed_padding"),
            "fixed_frontend_num_samples": metadata.get(
                "fixed_frontend_num_samples"
            ),
            "fixed_frontend_num_frames": metadata.get(
                "fixed_frontend_num_frames"
            ),
            "frontend_feature_bins": metadata.get("frontend_feature_bins"),
            "encoder_positions": metadata.get("encoder_positions"),
            "decoder_output_token_limit": decoder_limit,
            "model_type": model_type,
            "reason": reason,
        }

    @staticmethod
    def _is_whisper(model_ctx: Dict[str, Any]) -> bool:
        metadata = model_ctx.get("audio_metadata")
        if isinstance(metadata, dict) and metadata.get("model_type") == "whisper":
            return True
        pipe = model_ctx.get("pipeline")
        config = getattr(getattr(pipe, "model", None), "config", None)
        return str(getattr(config, "model_type", "") or "").lower() == "whisper"

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        pipe = model_ctx["pipeline"]
        task_type = model_ctx["task_type"]
        audio = processed_input["audio"]
        sample_rate = processed_input["sample_rate"]
        params = processed_input.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")

        mode = params.get("mode", "short_form")
        if mode != "short_form":
            raise ValueError(
                "this audio handler only supports mode='short_form'; "
                "use a separate long-form workload"
            )

        asr_task = params.get("asr_task", "transcribe")
        if asr_task != "transcribe":
            raise ValueError(
                "this workload only supports asr_task='transcribe'; "
                "translation must be profiled as a separate workload"
            )

        language = params.get("language", "en")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")
        language = language.strip()

        return_timestamps = params.get("return_timestamps", False)
        if not isinstance(return_timestamps, bool):
            raise ValueError("return_timestamps must be a boolean")
        if return_timestamps:
            raise ValueError("short-form profiling requires return_timestamps=false")

        pipeline_kwargs = params.get("pipeline_kwargs", {})
        if not isinstance(pipeline_kwargs, dict):
            raise ValueError("pipeline_kwargs must be an object")
        call_kwargs = dict(pipeline_kwargs)
        for field_name in ("chunk_length_s", "stride_length_s"):
            if call_kwargs.get(field_name) is not None:
                raise ValueError(
                    f"pipeline_kwargs.{field_name} is a chunked long-form "
                    "setting; use a separate long-form workload"
                )

        is_whisper = self._is_whisper(model_ctx)
        if is_whisper:
            metadata = self._audio_metadata(model_ctx)
            max_duration = _optional_positive_number(
                metadata.get("max_short_form_duration_s")
            ) or _WHISPER_SHORT_FORM_SECONDS
            duration_s = float(
                processed_input.get("_duration_s", len(audio) / sample_rate)
            )
            if duration_s > max_duration:
                raise ValueError(
                    f"Whisper short-form input is limited to {max_duration:g}s; "
                    f"received {duration_s:.6g}s. Use a separate long-form workload."
                )

        audio_input = {"raw": audio, "sampling_rate": sample_rate}
        if task_type in _ASR_TASK_TYPES and is_whisper:
            generate_kwargs = call_kwargs.pop("generate_kwargs", {})
            if not isinstance(generate_kwargs, dict):
                raise ValueError("pipeline_kwargs.generate_kwargs must be an object")
            generate_kwargs = dict(generate_kwargs)
            for key, semantic_value in {
                "task": asr_task,
                "language": language,
            }.items():
                if key in generate_kwargs and generate_kwargs[key] != semantic_value:
                    raise ValueError(
                        f"pipeline_kwargs.generate_kwargs.{key} conflicts with "
                        f"the semantic {key} parameter"
                    )
                generate_kwargs[key] = semantic_value

            if (
                "return_timestamps" in call_kwargs
                and call_kwargs["return_timestamps"] is not False
            ):
                raise ValueError(
                    "pipeline_kwargs.return_timestamps conflicts with "
                    "return_timestamps=false"
                )
            call_kwargs["return_timestamps"] = False
            call_kwargs["generate_kwargs"] = generate_kwargs

        # Non-Whisper ASR and classification pipelines do not necessarily accept
        # Whisper's language/task keywords, so only explicit pipeline_kwargs pass.
        return pipe(audio_input, **call_kwargs)

    @staticmethod
    def _output_token_count(pipe: Any, text: str) -> Optional[int]:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            if callable(getattr(tokenizer, "encode", None)):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            elif callable(tokenizer):
                encoded = tokenizer(text, add_special_tokens=False)
                token_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            else:
                return None
            if token_ids is None:
                return None
            array = np.asarray(token_ids)
            return int(array.size)
        except (TypeError, ValueError, RuntimeError):
            return None

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        task_type = model_ctx["task_type"]

        if task_type in _ASR_TASK_TYPES:
            text = raw_output.get("text", "") if isinstance(raw_output, dict) else ""
            if not isinstance(text, str):
                text = str(text)
            return {
                "task": task_type,
                "output_type": "transcription",
                "text": text,
                "output_length": len(text),
                "output_token_count": self._output_token_count(
                    model_ctx.get("pipeline"), text
                ),
            }
        if isinstance(raw_output, list):
            return {
                "task": task_type,
                "output_type": "classification",
                "n_results": len(raw_output),
            }
        if isinstance(raw_output, dict):
            return {
                "task": task_type,
                "output_type": "classification",
                "n_results": 1,
            }
        return {
            "task": task_type,
            "output_type": "unknown",
        }


HandlerRegistry.register("audio", "transformers_pipeline", AudioHandler)
HandlerRegistry.register("audio", "transformers_model", AudioHandler)
