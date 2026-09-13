"""Audio task handler - speech recognition, audio classification, etc."""

from __future__ import annotations

import base64
import binascii
import io
import math
from pathlib import Path
from types import SimpleNamespace
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
_TEXT_AUDIO_TASK_TYPES = {"text-to-speech", "text-to-audio"}
_CODEC_MODEL_TYPES = {"encodec": "EncodecModel", "dac": "DacModel"}
_TEXT_AUDIO_PROFILE_TOKEN_CAP = 512


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
        if task_type == "voice-activity-detection":
            return self._load_vad(model_source, device, model_revision, load_options)
        if task_type == "audio-to-audio":
            return self._load_codec(model_source, device, model_revision, load_options)
        if task_type not in _ASR_TASK_TYPES | _TEXT_AUDIO_TASK_TYPES | {"audio-classification"}:
            raise ValueError(f"unsupported audio task: {task_type}")
        import torch
        from transformers import pipeline as hf_pipeline

        if task_type in _TEXT_AUDIO_TASK_TYPES:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_source, **model_revision_kwargs(model_source, model_revision)
            )
            if config.model_type in {"speecht5", "fastspeech2_conformer"}:
                raise ValueError(
                    "text-to-audio/speech currently requires a self-contained waveform model "
                    "(for example VITS, Bark or MusicGen); spectrogram models require an "
                    "additional vocoder/speaker asset contract and are not supported"
                )
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
    def _load_vad(
        model_source: str, device: str, model_revision: str,
        load_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if device != "cpu":
            raise ValueError("Silero TorchScript voice activity detection supports CPU execution only")
        if load_options:
            raise ValueError("Silero TorchScript does not support attention load options")
        root = Path(model_source)
        if not root.is_dir():
            raise ValueError("Silero VAD requires a baked local snapshot containing silero_vad.jit")
        candidates = sorted(root.rglob("silero_vad.jit"))
        if len(candidates) != 1:
            raise ValueError(
                "Silero VAD snapshot must contain exactly one silero_vad.jit "
                f"(found {len(candidates)}); ONNX and other TorchScript architectures are unsupported"
            )
        import torch

        model = torch.jit.load(str(candidates[0]), map_location="cpu").eval()
        if not callable(getattr(model, "reset_states", None)):
            raise ValueError("silero_vad.jit must expose the Silero reset_states interface")
        return {
            "model": model, "task_type": "voice-activity-detection", "device": device,
            "model_revision": model_revision or "main", "load_options": {},
            "audio_metadata": {
                "sampling_rate": 16000, "model_type": "silero_vad", "short_form_fixed_padding": False,
            },
            "vad_model_file": str(candidates[0].relative_to(root)),
        }

    @staticmethod
    def _load_codec(
        model_source: str, device: str, model_revision: str,
        load_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        import transformers
        import torch

        revision = model_revision_kwargs(model_source, model_revision)
        config = transformers.AutoConfig.from_pretrained(model_source, **revision)
        model_class = _CODEC_MODEL_TYPES.get(config.model_type)
        if model_class is None:
            raise ValueError(
                "audio-to-audio supports Transformers Encodec/DAC waveform reconstruction; "
                f"model_type={config.model_type!r} needs a separate enhancement/separation adapter"
            )
        if getattr(config, "audio_channels", 1) != 1:
            raise ValueError("audio-to-audio currently requires a mono Encodec/DAC checkpoint")
        options = transformers_pipeline_load_kwargs(load_options).get("model_kwargs", {})
        model = getattr(transformers, model_class).from_pretrained(
            model_source, **revision, **options, torch_dtype=torch.float32
        ).to(device).eval()
        processor = transformers.AutoProcessor.from_pretrained(model_source, **revision)
        feature_extractor = getattr(processor, "feature_extractor", processor)
        metadata = AudioHandler._extract_audio_metadata(
            SimpleNamespace(model=model, feature_extractor=feature_extractor)
        )
        if metadata["sampling_rate"] is None:
            metadata["sampling_rate"] = _optional_positive_int(getattr(config, "sampling_rate", None))
        if metadata["sampling_rate"] is None:
            raise ValueError("audio codec must expose its input sampling rate")
        # Encodec chunk_length is an internal processing block, not an input limit.
        metadata["max_short_form_duration_s"] = None
        return {
            "model": model, "processor": processor, "task_type": "audio-to-audio",
            "device": device, "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}), "audio_metadata": metadata,
        }

    @staticmethod
    def _text_tokenizer(model_ctx: Dict[str, Any]) -> Any:
        pipe = model_ctx.get("pipeline")
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(getattr(pipe, "processor", None), "tokenizer", None)
        if not callable(getattr(tokenizer, "encode", None)):
            raise ValueError("text-to-audio/speech requires a tokenizer exposing encode for input measurement")
        return tokenizer

    @classmethod
    def _text_input_limit(cls, model_ctx: Dict[str, Any]) -> Optional[int]:
        tokenizer = cls._text_tokenizer(model_ctx)
        pipe = model_ctx.get("pipeline")
        config = getattr(getattr(pipe, "model", None), "config", None)
        if getattr(config, "model_type", None) == "bark":
            # TextToAudioPipeline feeds Bark a fixed semantic input window with
            # add_special_tokens=False, independent of the base BERT tokenizer.
            semantic = getattr(getattr(pipe, "generation_config", None), "semantic_config", {})
            semantic_limit = (
                semantic.get("max_input_semantic_length") if isinstance(semantic, dict)
                else getattr(semantic, "max_input_semantic_length", None)
            )
            return _optional_positive_int(semantic_limit) or 256
        limit = _optional_positive_int(getattr(tokenizer, "model_max_length", None))
        if limit is None or limit >= 1_000_000:
            return None
        special = getattr(tokenizer, "num_special_tokens_to_add", None)
        special_count = int(special(pair=False)) if callable(special) else 0
        return max(1, limit - special_count)

    def _preprocess_text(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        text = raw_input.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text-to-audio/speech input requires non-empty text")
        tokenizer = self._text_tokenizer(model_ctx)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        limit = self._text_input_limit(model_ctx) or _TEXT_AUDIO_PROFILE_TOKEN_CAP
        truncated = limit is not None and len(token_ids) > limit
        if truncated:
            text = tokenizer.decode(token_ids[:limit], skip_special_tokens=True)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        return {
            "text": text, "params": raw_input.get("params", {}),
            "_effective_input_scale": len(token_ids), "_truncated_by_limit": truncated,
            "_probe_reason": "text input token count excludes special tokens; output audio duration is independent",
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
        task_type = model_ctx["task_type"]
        if task_type in _TEXT_AUDIO_TASK_TYPES:
            return self._preprocess_text(model_ctx, raw_input)
        if "audio_samples" in raw_input or "audio_base64" not in raw_input:
            raise ValueError("audio input requires audio_base64 WAV; audio_samples is no longer supported")

        metadata = self._audio_metadata(model_ctx)
        required_sample_rate = _optional_positive_int(metadata.get("sampling_rate"))

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

        input_num_samples = int(audio_array.size)
        duration_s = input_num_samples / sample_rate
        source_sample_rate = sample_rate
        if required_sample_rate and sample_rate != required_sample_rate:
            if task_type in _ASR_TASK_TYPES:
                # Preserve the established ASR input contract and provenance.
                raise ValueError(
                    "audio sample rate does not match the model feature extractor "
                    f"({sample_rate} != {required_sample_rate})"
                )
            from scipy.signal import resample_poly

            divisor = math.gcd(sample_rate, required_sample_rate)
            audio_array = resample_poly(
                audio_array, required_sample_rate // divisor, sample_rate // divisor
            ).astype(np.float32)
            sample_rate = required_sample_rate
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

        processed = {
            "audio": audio_array,
            "sample_rate": sample_rate,
            "params": raw_input.get("params", {}),
            "_effective_input_scale": duration_s,
            "_input_num_samples": input_num_samples,
            "_duration_s": duration_s,
            "_truncated_by_limit": exceeds_short_form_limit,
            "_probe_reason": reason,
        }
        if task_type not in _ASR_TASK_TYPES:
            processed["_source_sample_rate"] = source_sample_rate
            processed["_model_input_num_samples"] = int(audio_array.size)
            if task_type == "audio-to-audio":
                processor_kwargs = {}
                if metadata.get("model_type") == "dac":
                    # DAC's model accepts input_values only. Its feature extractor's
                    # batched padding path emits a padding_mask and an extra axis.
                    # One waveform needs neither; the model handles codec framing.
                    processor_kwargs["padding"] = False
                inputs = model_ctx["processor"](
                    raw_audio=audio_array, sampling_rate=sample_rate, return_tensors="pt",
                    **processor_kwargs,
                )
                processed["model_inputs"] = {
                    key: value.to(model_ctx["device"]) for key, value in inputs.items()
                }
            elif task_type == "voice-activity-detection":
                import torch

                frame_samples = 512
                padding = (-len(audio_array)) % frame_samples
                framed = np.pad(audio_array, (0, padding)).reshape(-1, frame_samples)
                processed["frames"] = torch.from_numpy(framed)
        return processed

    def get_scale_metadata(
        self,
        model_ctx: Dict[str, Any],
        raw_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        if model_ctx["task_type"] in _TEXT_AUDIO_TASK_TYPES:
            model_limit = self._text_input_limit(model_ctx)
            return {
                "input_scale_type": "seq_length",
                "max_effective_input_scale": model_limit or _TEXT_AUDIO_PROFILE_TOKEN_CAP,
                "model_max_input_tokens": model_limit,
                "input_limit_source": "tokenizer_or_generation_config" if model_limit else "profiling_text_token_cap",
                "reason": (
                    "text input token count excludes special tokens; generated audio samples are output"
                    if model_limit else
                    "tokenizer exposes no finite limit; profiling uses an explicit 512-token workload cap, not a model limit"
                ),
            }
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

        result = {
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
        if model_ctx["task_type"] not in _ASR_TASK_TYPES:
            result["source_sampling_rate"] = raw_input.get("sample_rate", 16000)
            result["resampling_policy"] = "scipy.signal.resample_poly_if_required_in_preprocess"
        return result

    @staticmethod
    def _is_whisper(model_ctx: Dict[str, Any]) -> bool:
        metadata = model_ctx.get("audio_metadata")
        if isinstance(metadata, dict) and metadata.get("model_type") == "whisper":
            return True
        pipe = model_ctx.get("pipeline")
        config = getattr(getattr(pipe, "model", None), "config", None)
        return str(getattr(config, "model_type", "") or "").lower() == "whisper"

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        task_type = model_ctx["task_type"]
        params = processed_input.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        pipeline_kwargs = params.get("pipeline_kwargs", {})
        if not isinstance(pipeline_kwargs, dict):
            raise ValueError("pipeline_kwargs must be an object")
        if task_type in _TEXT_AUDIO_TASK_TYPES:
            if "preprocess_params" in pipeline_kwargs:
                raise ValueError("preprocess_params cannot override tokenization after input scale measurement")
            return model_ctx["pipeline"](processed_input["text"], **pipeline_kwargs)
        if task_type == "audio-to-audio":
            import torch

            if set(pipeline_kwargs) - {"bandwidth", "n_quantizers"}:
                raise ValueError("audio codec only accepts bandwidth (Encodec) or n_quantizers (DAC)")
            with torch.inference_mode():
                output = model_ctx["model"](**processed_input["model_inputs"], **pipeline_kwargs)
            return {"audio": output.audio_values, "sampling_rate": model_ctx["audio_metadata"]["sampling_rate"]}
        if task_type == "voice-activity-detection":
            import torch

            if set(pipeline_kwargs) - {"threshold"}:
                raise ValueError("Silero VAD only accepts pipeline_kwargs.threshold")
            threshold = pipeline_kwargs.get("threshold", 0.5)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
                raise ValueError("VAD threshold must be a number strictly between zero and one")
            model = model_ctx["model"]
            # The recurrent state belongs to one request, including profiler replays.
            model.reset_states()
            with torch.inference_mode():
                probabilities = [model(frame, 16000) for frame in processed_input["frames"]]
            return {
                "probabilities": probabilities, "frame_samples": 512,
                "sample_rate": 16000, "input_num_samples": len(processed_input["audio"]),
                "threshold": float(threshold),
            }

        pipe = model_ctx["pipeline"]
        audio = processed_input["audio"]
        sample_rate = processed_input["sample_rate"]

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

        if task_type in _TEXT_AUDIO_TASK_TYPES | {"audio-to-audio"}:
            if not isinstance(raw_output, dict) or "audio" not in raw_output:
                raise ValueError("audio generation must return an audio waveform")
            waveform = raw_output["audio"]
            if callable(getattr(waveform, "detach", None)):
                waveform = waveform.detach().float().cpu().numpy()
            waveform = np.asarray(waveform)
            if waveform.ndim not in (1, 2, 3) or not waveform.size or not np.isfinite(waveform).all():
                raise ValueError("generated audio must be a non-empty finite waveform")
            sample_rate = _positive_int(raw_output.get("sampling_rate"), "output sampling_rate")
            return {
                "task": task_type, "output_type": "audio", "n_results": 1,
                "audio_shape": list(waveform.shape), "audio_num_samples": int(waveform.shape[-1]),
                "audio_sample_rate": sample_rate, "audio_duration_s": waveform.shape[-1] / sample_rate,
            }
        if task_type == "voice-activity-detection":
            probabilities = [float(value) for value in raw_output["probabilities"]]
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
                raise ValueError("VAD model must return finite speech probabilities in [0, 1]")
            frame_size = raw_output["frame_samples"]
            sample_count = raw_output["input_num_samples"]
            sample_rate = raw_output["sample_rate"]
            segments = []
            start = None
            for index, probability in enumerate(probabilities + [0.0]):
                position = min(index * frame_size, sample_count)
                if probability >= raw_output["threshold"] and start is None:
                    start = position
                elif probability < raw_output["threshold"] and start is not None:
                    segments.append({"start": start / sample_rate, "end": position / sample_rate})
                    start = None
            return {
                "task": task_type, "output_type": "voice_activity", "n_results": len(segments),
                "segments": segments, "timestamp_unit": "seconds",
                "speech_duration_s": sum(segment["end"] - segment["start"] for segment in segments),
                "threshold": raw_output["threshold"], "frame_duration_s": frame_size / sample_rate,
                "segmentation_policy": "adjacent_frames_above_threshold",
            }
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
HandlerRegistry.register("audio", "torchscript", AudioHandler)
