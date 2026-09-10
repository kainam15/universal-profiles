"""Built-in Transformers multimodal tasks with explicit measurement phases.

API reference: huggingface/transformers v4.57.6 (Apache-2.0), processors,
QA pipelines, ColPali/ColQwen2 retrieval and Qwen2.5-Omni generation.
Media decoding and device transfer happen in preprocess; predict contains
model operations (including retrieval MaxSim); decoding is postprocessing.
"""

from __future__ import annotations

import base64
import binascii
import io
import inspect
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np

from acprof.container.handlers import (
    BaseHandler,
    HandlerRegistry,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)
from acprof.container.handlers.audio import AudioHandler, _positive_int


_TASKS = {
    "audio-text-to-text", "image-text-to-text", "visual-question-answering",
    "document-question-answering", "video-text-to-text",
    "visual-document-retrieval", "any-to-any",
}
_QA_TASKS = {"visual-question-answering", "document-question-answering"}
_OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _to_device(inputs: Any, model: Any) -> Any:
    """Move only tensor values, preserving integer token IDs and masks."""
    if isinstance(inputs, Mapping):
        return {key: _to_device(value, model) for key, value in inputs.items()}
    if isinstance(inputs, list):
        return [_to_device(value, model) for value in inputs]
    if callable(getattr(inputs, "to", None)):
        kwargs = {"device": model.device}
        if inputs.is_floating_point():
            kwargs["dtype"] = model.dtype
        return inputs.to(**kwargs)
    return inputs


def _as_numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


class MultimodalHandler(BaseHandler):
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
        import transformers

        if task_type not in _TASKS:
            raise ValueError(f"unsupported multimodal task: {task_type}")
        if backend not in {"transformers_model", "transformers_pipeline"}:
            raise ValueError(f"unsupported multimodal backend: {backend}")
        attention_options = transformers_pipeline_load_kwargs(load_options)
        if task_type == "any-to-any" and attention_options:
            raise RuntimeError(
                "any-to-any eager FLOP profiling is unsupported: Qwen2.5-Omni "
                "token2wav requires SDPA and cannot verify all-eager attention; "
                "normal inference and vendor profilers remain available"
            )
        source_kwargs = {
            **model_revision_kwargs(model_source, model_revision),
            "trust_remote_code": False,
        }
        dtype = torch.float32 if device == "cpu" else torch.float16
        ctx = {
            "task_type": task_type, "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }
        if task_type in _QA_TASKS:
            pipe = transformers.pipeline(
                task=task_type, model=model_source, **source_kwargs,
                device_map="cpu" if device == "cpu" else "auto",
                torch_dtype=dtype, **attention_options,
            )
            # Supplied word boxes must bypass OCR, including processor OCR.
            if task_type == "document-question-answering":
                for component in (getattr(pipe, "image_processor", None), getattr(pipe, "feature_extractor", None)):
                    if component is not None and hasattr(component, "apply_ocr"):
                        component.apply_ocr = False
            return {**ctx, "pipeline": pipe, "model": pipe.model, "mode": "qa"}

        config = transformers.AutoConfig.from_pretrained(model_source, **source_kwargs)
        model_type = str(config.model_type)
        mode = "generate"
        if task_type == "any-to-any":
            if model_type != "qwen2_5_omni":
                raise ValueError(
                    f"unsupported any-to-any architecture {model_type!r}; "
                    "expected qwen2_5_omni for text+audio output"
                )
            class_name, mode = "Qwen2_5OmniForConditionalGeneration", "omni"
        elif task_type == "audio-text-to-text":
            class_name = {
                "qwen2_audio": "Qwen2AudioForConditionalGeneration",
                "qwen2_5_omni": "Qwen2_5OmniThinkerForConditionalGeneration",
            }.get(model_type)
            if class_name is None:
                raise ValueError(
                    f"unsupported audio-text-to-text architecture {model_type!r}; "
                    "expected qwen2_audio or qwen2_5_omni"
                )
        elif task_type == "visual-document-retrieval":
            class_name = {"colpali": "ColPaliForRetrieval", "colqwen2": "ColQwen2ForRetrieval"}.get(model_type)
            if class_name is None:
                raise ValueError(
                    f"unsupported visual-document-retrieval architecture {model_type!r}; "
                    "expected colpali or colqwen2"
                )
            mode = "retrieval"
        else:
            class_name = "AutoModelForImageTextToText"
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise RuntimeError(f"{class_name} is unavailable; rebuild the multimodal image with transformers==4.57.6")
        processor = transformers.AutoProcessor.from_pretrained(model_source, **source_kwargs)
        if task_type == "video-text-to-text" and "videos" not in inspect.signature(processor.__call__).parameters:
            raise ValueError(f"architecture {model_type!r} has no native video processor")
        model = model_class.from_pretrained(
            model_source, **source_kwargs,
            device_map="cpu" if device == "cpu" else "auto", torch_dtype=dtype,
            **attention_options.get("model_kwargs", {}),
            **({"enable_audio_output": True} if mode == "omni" else {}),
        )
        model.eval()
        if mode == "omni":
            # Otherwise upstream generate() performs this conversion in predict.
            model.token2wav.float()
        return {**ctx, "model": model, "processor": processor, "model_type": model_type, "mode": mode}

    @staticmethod
    def _image(encoded: Any) -> Any:
        from PIL import Image, UnidentifiedImageError

        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_base64 must be a non-empty Base64 string")
        try:
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("image_base64 must contain a valid Base64 image") from exc

    @staticmethod
    def _params(task: str, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("params must be an object")
        allowed = {"max_new_tokens", "do_sample"}
        if task in _QA_TASKS:
            allowed |= {"top_k"}
        if task == "document-question-answering":
            allowed |= {"max_answer_len", "max_seq_len", "doc_stride"}
        if task == "any-to-any":
            allowed |= {"talker_max_new_tokens", "speaker", "return_audio", "seed"}
        if task == "visual-document-retrieval":
            allowed = set()
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unsupported params for {task}: {', '.join(sorted(unknown))}")
        params = dict(value)
        for name in ("max_new_tokens", "talker_max_new_tokens", "top_k", "max_answer_len", "max_seq_len", "doc_stride"):
            if name in params:
                params[name] = _positive_int(params[name], name)
        if "do_sample" in params and params["do_sample"] is not False:
            raise ValueError("do_sample must be false for reproducible multimodal workloads")
        if "speaker" in params and params["speaker"] not in {"Chelsie", "Ethan"}:
            raise ValueError("speaker must be Chelsie or Ethan")
        if "return_audio" in params and params["return_audio"] is not True:
            raise ValueError("any-to-any requires return_audio=true for joint text+audio output")
        if "seed" in params and (
            isinstance(params["seed"], bool)
            or not isinstance(params["seed"], int)
            or not 0 <= params["seed"] < 2**63
        ):
            raise ValueError("seed must be an integer in [0, 2**63)")
        if task != "visual-document-retrieval":
            params.setdefault("max_new_tokens", 64)
            params.setdefault("do_sample", False)
        return params

    @staticmethod
    def _word_boxes(sample: Dict[str, Any]) -> list[Any]:
        words, boxes = sample.get("words"), sample.get("boxes")
        if not isinstance(words, list) or not words or not all(isinstance(word, str) and word for word in words):
            raise ValueError("document words and boxes must be non-empty parallel lists")
        if not isinstance(boxes, list) or len(boxes) != len(words):
            raise ValueError("document boxes must match words")
        for box in boxes:
            if (
                not isinstance(box, list) or len(box) != 4
                or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 1000 for x in box)
                or box[0] > box[2] or box[1] > box[3]
            ):
                raise ValueError("document boxes must be ordered [x0,y0,x1,y1] integers normalized to 0..1000")
        return list(zip(words, boxes))

    def preprocess(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Any:
        task = model_ctx["task_type"]
        unknown = set(raw_input) - {"samples", "params", "input_scale", "input_scale_type"}
        if unknown:
            raise ValueError(f"unsupported request fields: {', '.join(sorted(unknown))}")
        samples = raw_input.get("samples")
        if not isinstance(samples, list) or len(samples) != 1 or not isinstance(samples[0], dict):
            raise ValueError("multimodal requests require batch_size=1 (exactly one sample)")
        sample = samples[0]
        allowed = {"text", "image_base64", "audio_base64", "sampling_rate", "video_frames_base64", "fps"}
        if task == "document-question-answering":
            allowed |= {"words", "boxes"}
        unknown = set(sample) - allowed
        if unknown:
            raise ValueError(f"unsupported sample fields: {', '.join(sorted(unknown))}")
        if "sampling_rate" in sample and "audio_base64" not in sample:
            raise ValueError("sampling_rate requires audio_base64")
        if "fps" in sample and "video_frames_base64" not in sample:
            raise ValueError("fps requires video_frames_base64")
        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("sample text must be a non-empty string")
        params = self._params(task, raw_input.get("params", {}))
        image_tasks = {
            "image-text-to-text", "visual-question-answering",
            "document-question-answering", "visual-document-retrieval",
        }
        if task in image_tasks and "image_base64" not in sample:
            raise ValueError(f"{task} requires image_base64")
        if task == "audio-text-to-text" and "audio_base64" not in sample:
            raise ValueError("audio-text-to-text requires audio_base64")
        if task == "video-text-to-text" and "video_frames_base64" not in sample:
            raise ValueError("video-text-to-text requires video_frames_base64")
        if task != "any-to-any":
            for key, valid_tasks in (
                ("image_base64", image_tasks),
                ("audio_base64", {"audio-text-to-text"}),
                ("video_frames_base64", {"video-text-to-text"}),
            ):
                if key in sample and task not in valid_tasks:
                    raise ValueError(f"{task} does not accept {key}")
        image = self._image(sample["image_base64"]) if "image_base64" in sample else None
        media_scales = {}
        if image is not None:
            if image.width != image.height:
                raise ValueError("resolution_px requires a square image")
            media_scales["resolution_px"] = float(image.width)
        if task in _QA_TASKS:
            explicit_generation_params = {"max_new_tokens", "do_sample"}.intersection(
                raw_input.get("params", {})
            )
            if not model_ctx["pipeline"].model.can_generate() and explicit_generation_params:
                raise ValueError("generation params are inapplicable to a classification/extractive QA model")
            return {**self._preprocess_qa(model_ctx, sample, image, params), **self._scale_metadata(raw_input, media_scales)}
        processor, model = model_ctx["processor"], model_ctx["model"]
        if task == "visual-document-retrieval":
            return {
                "query_inputs": _to_device(processor.process_queries(text=[text], return_tensors="pt"), model),
                "document_inputs": _to_device(processor.process_images(images=[image], return_tensors="pt"), model),
                **self._scale_metadata(raw_input, media_scales),
            }
        content, media_kwargs = [], {}
        if image is not None:
            content.append({"type": "image"})
            media_kwargs["images"] = [image]
        if "audio_base64" in sample:
            audio, header_rate = AudioHandler._decode_wav(sample["audio_base64"])
            rate = _positive_int(sample.get("sampling_rate"), "sampling_rate")
            if rate != header_rate:
                raise ValueError("sampling_rate does not match the WAV header")
            required_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
            if isinstance(required_rate, int) and rate != required_rate:
                raise ValueError(
                    f"sampling_rate {rate} does not match processor rate {required_rate}; "
                    "resample before submitting"
                )
            max_samples = getattr(getattr(processor, "feature_extractor", None), "n_samples", None)
            if isinstance(max_samples, int) and audio.size > max_samples:
                raise ValueError(f"audio exceeds processor limit ({max_samples / rate:g}s); truncation is not allowed")
            media_scales["duration_s"] = audio.size / rate
            content.append({"type": "audio"})
            media_kwargs.update(audio=[audio], sampling_rate=rate)
        if "video_frames_base64" in sample:
            frames = sample["video_frames_base64"]
            if not isinstance(frames, list) or not frames:
                raise ValueError("video_frames_base64 must be a non-empty frame list")
            fps = sample.get("fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be a finite positive number")
            decoded = [np.asarray(self._image(frame)) for frame in frames]
            if len({frame.shape for frame in decoded}) != 1:
                raise ValueError("video frames must have identical dimensions")
            content.append({"type": "video"})
            media_kwargs.update(videos=[np.stack(decoded)], fps=float(fps))
            media_scales["frame_count"] = float(len(frames))
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        if model_ctx.get("model_type") == "qwen2_5_omni":
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": _OMNI_SYSTEM_PROMPT}]})
        if not getattr(processor, "chat_template", None):
            raise ValueError("this multimodal processor has no chat_template; a model-specific prompt adapter is required")
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], return_tensors="pt", padding=True, add_special_tokens=False, **media_kwargs)
        inputs = _to_device(inputs, model)
        if "input_ids" not in inputs:
            raise ValueError("multimodal processor did not return input_ids")
        for modality, present, keys in (
            ("image", image is not None, {"pixel_values", "image_pixel_values", "images"}),
            ("audio", "audio_base64" in sample, {"input_features", "audio_values", "audio_features"}),
            ("video", "video_frames_base64" in sample, {"pixel_values_videos", "video_pixel_values", "video_values", "pixel_values"}),
        ):
            if present and not keys.intersection(inputs):
                raise ValueError(f"processor discarded the requested {modality} modality")
        return {
            "inputs": inputs, "prompt_length": int(inputs["input_ids"].shape[-1]),
            "params": params, **self._scale_metadata(raw_input, media_scales),
        }

    @staticmethod
    def _scale_metadata(raw_input: Dict[str, Any], media_scales: Dict[str, float]) -> Dict[str, Any]:
        scale_type = raw_input.get("input_scale_type")
        if scale_type is None and len(media_scales) == 1:
            scale_type = next(iter(media_scales))
        if scale_type not in media_scales:
            raise ValueError("input_scale_type must select one of the actual input media dimensions")
        effective = media_scales[scale_type]
        declared = raw_input.get("input_scale", effective)
        if (
            isinstance(declared, bool) or not isinstance(declared, (int, float))
            or not math.isfinite(declared)
            or not math.isclose(declared, effective, rel_tol=0, abs_tol=1e-6)
        ):
            raise ValueError(f"input_scale does not match decoded {scale_type}: {declared!r} != {effective}")
        return {
            "_effective_input_scale": effective, "_truncated_by_limit": False,
            "_probe_reason": "raw_media_scale_verified; processor may resize or pad internally",
        }

    def _preprocess_qa(
        self, model_ctx: Dict[str, Any], sample: Dict[str, Any],
        image: Any, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        pipe = model_ctx["pipeline"]
        inputs = {"image": image, "question": sample["text"]}
        if model_ctx["task_type"] == "document-question-answering":
            # Donut consumes pixels only; requiring the same reproducible word
            # boxes also keeps extractive LayoutLM runs independent of OCR.
            inputs["word_boxes"] = self._word_boxes(sample)
        pre_params = {key: params[key] for key in ("max_seq_len", "doc_stride") if key in params}
        post_params = {key: params[key] for key in ("top_k", "max_answer_len") if key in params}
        forward_params = {}
        if pipe.model.can_generate():
            forward_params = {key: params[key] for key in ("max_new_tokens", "do_sample")}
        chunks = pipe.preprocess(inputs, **pre_params)
        if isinstance(chunks, Mapping):
            chunks = [chunks]
        else:
            chunks = list(chunks)
        if not chunks:
            raise ValueError("QA preprocessing returned no chunks")
        chunks = [pipe._ensure_tensor_on_device(dict(chunk), device=pipe.device) for chunk in chunks]
        return {"chunks": chunks, "forward_params": forward_params, "post_params": post_params}

    def predict(self, model_ctx: Dict[str, Any], processed_input: Any) -> Any:
        import torch

        mode = model_ctx["mode"]
        with torch.inference_mode():
            if mode == "qa":
                # Document QA pops metadata from its argument: fresh dicts are
                # essential when repeat-in-window and profilers reuse inputs.
                outputs = [
                    model_ctx["pipeline"]._forward(dict(chunk), **processed_input["forward_params"])
                    for chunk in processed_input["chunks"]
                ]
                return {"qa_outputs": outputs, "post_params": processed_input["post_params"]}
            model = model_ctx["model"]
            if mode == "retrieval":
                query = model(**processed_input["query_inputs"]).embeddings
                document = model(**processed_input["document_inputs"]).embeddings
                scores = model_ctx["processor"].score_retrieval(query, document, output_device=model.device)
                return {"scores": scores}
            params = dict(processed_input["params"])
            if mode == "omni":
                seed = params.get("seed", 12345)
                params = {
                    "thinker_max_new_tokens": params["max_new_tokens"], "thinker_do_sample": False,
                    "talker_max_new_tokens": params.get("talker_max_new_tokens", 256), "talker_do_sample": False,
                    "speaker": params.get("speaker", "Chelsie"), "return_audio": True,
                }
                # Token2Wav samples diffusion noise even with do_sample=False.
                # Preserve outside RNG state while repeating identical requests.
                device = str(model_ctx.get("device", getattr(model, "device", "cpu")))
                devices = list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
                with torch.random.fork_rng(devices=devices):
                    # torch.manual_seed also changes CUDA generators on CPU
                    # runs. Seed only the generators protected by fork_rng.
                    torch.random.default_generator.manual_seed(seed)
                    for device_index in devices:
                        torch.cuda.default_generators[device_index].manual_seed(seed)
                    generated = model.generate(**processed_input["inputs"], **params)
            else:
                generated = model.generate(**processed_input["inputs"], **params)
            return {"generated": generated, "prompt_length": processed_input["prompt_length"]}

    @staticmethod
    def _text_summary(tokenizer: Any, texts: list[str]) -> Dict[str, Any]:
        token_count = None
        if tokenizer is not None:
            try:
                token_count = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass
        return {"n_results": len(texts), "output_length": sum(map(len, texts)), "output_token_count": token_count}

    def postprocess(self, model_ctx: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        task, mode = model_ctx["task_type"], model_ctx["mode"]
        result = {"task": task}
        if mode == "qa":
            import torch

            pipe = model_ctx["pipeline"]
            outputs = pipe._ensure_tensor_on_device(raw_output["qa_outputs"], device=torch.device("cpu"))
            answers = pipe.postprocess(
                outputs if task == "document-question-answering" else outputs[0],
                **raw_output["post_params"],
            )
            if not isinstance(answers, list) or any(
                not isinstance(answer, dict)
                or not isinstance(answer.get("answer"), (str, type(None)))
                for answer in answers
            ):
                raise ValueError("QA output must contain answer records")
            texts = [answer.get("answer") or "" for answer in answers]
            return {
                **result, "output_type": "answers", "answers": answers,
                **self._text_summary(getattr(pipe, "tokenizer", None), texts),
            }
        if mode == "retrieval":
            scores = _as_numpy(raw_output["scores"])
            if scores.shape != (1, 1) or not np.isfinite(scores).all():
                raise ValueError("retrieval scores must be a finite 1x1 matrix")
            return {
                **result, "output_type": "retrieval_scores", "scores": scores.tolist(),
                "n_queries": 1, "n_documents": 1, "n_results": 1,
                "retrieval_scope": "query_and_document_encoding_plus_scoring",
            }
        generated = raw_output["generated"]
        audio_summary = {}
        if mode == "omni":
            if not isinstance(generated, tuple) or len(generated) != 2:
                raise ValueError("any-to-any requires joint text and audio output")
            generated, audio = generated
            waveform = _as_numpy(audio)
            if waveform.size == 0 or waveform.ndim not in (1, 2) or not np.isfinite(waveform).all():
                raise ValueError("any-to-any returned invalid or empty audio")
            audio_summary = {
                "audio_num_samples": int(waveform.size), "audio_sample_rate": 24000,
                "audio_duration_s": waveform.size / 24000,
            }
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        if not getattr(model_ctx["model"].config, "is_encoder_decoder", False):
            generated = generated[:, raw_output["prompt_length"]:]
        processor = model_ctx["processor"]
        texts = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("generated output must decode to text strings")
        return {
            **result, "output_type": "text_audio" if mode == "omni" else "text", "texts": texts,
            **self._text_summary(getattr(processor, "tokenizer", None), texts), **audio_summary,
        }

    def get_scale_metadata(self, model_ctx: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
        processor = model_ctx.get("processor")
        rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        return {"sampling_rate": rate} if isinstance(rate, int) and rate > 0 else {}


HandlerRegistry.register("multimodal", "transformers_model", MultimodalHandler)
HandlerRegistry.register("multimodal", "transformers_pipeline", MultimodalHandler)
