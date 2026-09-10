"""Materialized multimodal inputs with one explicit scaling dimension.

Assets are read once on the host. Inference and profiler containers consume the
same inline payload, without network access, video decoding, or asset discovery.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator


TASK_MODALITY = {
    "audio-text-to-text": "audio",
    "image-text-to-text": "image",
    "visual-question-answering": "image",
    "document-question-answering": "image",
    "video-text-to-text": "video",
    "visual-document-retrieval": "image",
    "any-to-any": "audio",
}
SCALE_TYPES = {"image": "resolution_px", "audio": "duration_s", "video": "frame_count"}
DEFAULT_SCALES = {"image": [224, 336, 448], "audio": [1, 2, 5, 10], "video": [2, 4, 8]}
DEFAULT_TEXT = {
    "image-text-to-text": "Describe the image briefly.",
    "visual-question-answering": "What color is the square?",
    "document-question-answering": "What is the total?",
    "visual-document-retrieval": "An invoice with a total of 42 dollars.",
    "audio-text-to-text": "Summarize what the speaker says.",
    "video-text-to-text": "Describe how the square moves.",
    "any-to-any": "Describe the provided content briefly.",
}
SPEC_KEYS = {
    "schema_version", "workload_id", "task", "text", "image_path", "audio_path",
    "video_frames", "fps", "words", "boxes", "params", "input_scales",
    "modalities", "scale_modality", "image_resolution", "audio_duration_s",
    "video_num_frames", "provenance",
}


def _positive_number(value: Any, name: str, *, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (integer and result != int(result)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    return result


def _png(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _scene(*, document: bool = False, frame: int = 0) -> tuple[Any, list, list]:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    words, boxes = [], []
    if document:
        font = ImageFont.load_default(size=26)
        for text, x, y in (("INVOICE", 32, 40), ("Total", 32, 140), ("42", 170, 140), ("USD", 230, 140)):
            draw.text((x, y), text, fill="black", font=font)
            box = draw.textbbox((x, y), text, font=font)
            words.append(text)
            boxes.append([round(value * 1000 / 448) for value in box])
    else:
        x = 40 + (frame * 16) % 240
        draw.rectangle((x, 110, x + 100, 210), fill="red")
        draw.ellipse((290, 280, 390, 380), fill="blue")
    return image, words, boxes


class MultimodalWorkloadGenerator(WorkloadGenerator):
    def __init__(self, model_id: str, task_type: str, batch_size: int,
                 workload_spec_path: Optional[str] = None):
        super().__init__(model_id, task_type, batch_size)
        if task_type not in TASK_MODALITY:
            raise ValueError(f"unsupported multimodal task: {task_type}")
        if isinstance(batch_size, bool) or batch_size != 1:
            raise ValueError("multimodal workloads require batch_size=1")
        self._assets: List[Dict[str, Any]] = []
        self._spec: Dict[str, Any] = {}
        self._base = Path.cwd()
        self._manifest_sha256 = None
        if workload_spec_path:
            path = Path(workload_spec_path).resolve()
            content = path.read_bytes()
            self._manifest_sha256 = hashlib.sha256(content).hexdigest()
            self._base = path.parent
            self._spec = json.loads(content)
            if not isinstance(self._spec, dict):
                raise ValueError("multimodal workload manifest must be an object")
            unknown = set(self._spec) - SPEC_KEYS
            if unknown:
                raise ValueError(f"unknown multimodal workload keys: {', '.join(sorted(unknown))}")
            if self._spec.get("schema_version") != 1:
                raise ValueError("multimodal workload schema_version must be 1")
            if self._spec.get("task", task_type) != task_type:
                raise ValueError("multimodal workload task does not match detected task")
        spec = self._spec
        self._modalities = spec.get("modalities", [TASK_MODALITY[task_type]])
        if (not isinstance(self._modalities, list) or not self._modalities
                or any(item not in SCALE_TYPES for item in self._modalities)
                or len(set(self._modalities)) != len(self._modalities)):
            raise ValueError("modalities must be a nonempty unique list of image/audio/video")
        if task_type != "any-to-any" and self._modalities != [TASK_MODALITY[task_type]]:
            raise ValueError(f"{task_type} requires modality {TASK_MODALITY[task_type]}")
        modality_fields = {
            "image": {"image_path", "words", "boxes"},
            "audio": {"audio_path", "audio_duration_s"},
            "video": {"video_frames", "fps", "video_num_frames"},
        }
        for modality, keys in modality_fields.items():
            unused = set(spec) & keys if modality not in self._modalities else set()
            if unused:
                raise ValueError(f"fields require modality {modality}: {', '.join(sorted(unused))}")
        if "image_resolution" in spec and not {"image", "video"} & set(self._modalities):
            raise ValueError("image_resolution requires image or video modality")
        if {"words", "boxes"} & set(spec) and task_type != "document-question-answering":
            raise ValueError("words and boxes are only supported for document-question-answering")
        default_modality = TASK_MODALITY[task_type]
        if default_modality not in self._modalities:
            default_modality = self._modalities[0]
        self._scale_modality = spec.get("scale_modality", default_modality)
        if self._scale_modality not in self._modalities:
            raise ValueError("scale_modality must be present in modalities")
        active_fixed_field = {"image": "image_resolution", "audio": "audio_duration_s", "video": "video_num_frames"}[self._scale_modality]
        if active_fixed_field in spec and not (self._scale_modality == "image" and "video" in self._modalities):
            raise ValueError(f"{active_fixed_field} is not fixed on the active scale axis; use input_scales")
        self.input_scale_type = SCALE_TYPES[self._scale_modality]
        self._text = spec.get("text", DEFAULT_TEXT[task_type])
        if not isinstance(self._text, str) or not self._text.strip():
            raise ValueError("text must be a nonempty prompt or query")
        defaults = {} if task_type in {"visual-question-answering", "document-question-answering", "visual-document-retrieval"} else {"max_new_tokens": 64, "do_sample": False}
        if task_type == "any-to-any":
            defaults.update(return_audio=True, talker_max_new_tokens=256, speaker="Chelsie", seed=12345)
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        self._params = {**defaults, **params}
        self._fps = _positive_number(spec.get("fps", 2), "fps")
        self._image_resolution = int(_positive_number(spec.get("image_resolution", 224), "image_resolution", integer=True))
        self._audio_duration = _positive_number(spec.get("audio_duration_s", 2), "audio_duration_s")
        self._video_count = int(_positive_number(spec.get("video_num_frames", 4), "video_num_frames", integer=True))
        self._image = None
        self._frames = None
        self._words, self._boxes = [], []
        if "image" in self._modalities:
            if "image_path" in spec:
                self._image = self._read_image(spec["image_path"])
                if task_type == "document-question-answering" and not {"words", "boxes"} <= set(spec):
                    raise ValueError("custom document images require precomputed words and boxes")
            else:
                self._image, self._words, self._boxes = _scene(document=task_type in {"document-question-answering", "visual-document-retrieval"})
            self._words = spec.get("words", self._words)
            self._boxes = spec.get("boxes", self._boxes)
            if (not isinstance(self._words, list) or not isinstance(self._boxes, list)
                    or len(self._words) != len(self._boxes)
                    or any(not isinstance(word, str) or not word.strip() for word in self._words)):
                raise ValueError("words and boxes must be lists of equal lengths with nonempty words")
            for box in self._boxes:
                if (not isinstance(box, list) or len(box) != 4
                        or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 1000 for v in box)
                        or box[0] > box[2] or box[1] > box[3]):
                    raise ValueError("boxes must be ordered integer [x0,y0,x1,y1] coordinates in 0..1000")
        if "audio" in self._modalities:
            if "audio_path" in spec:
                data = self._read_asset(spec["audio_path"], "audio")
            else:
                from acprof.workloads.audio import AudioWorkloadGenerator
                # Reuse the existing asset/provenance validation, including SHA256.
                audio_source = AudioWorkloadGenerator(model_id, "automatic-speech-recognition", 1)
                data = self._read_asset(str(audio_source.asset_path), "audio")
                self._assets[-1]["provenance"] = audio_source.plan_metadata()["provenance"]
            with wave.open(io.BytesIO(data), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                    raise ValueError("audio_path must be mono PCM16 WAV")
                self._sample_rate = source.getframerate()
                self._pcm = source.readframes(source.getnframes())
        if "video" in self._modalities and "video_frames" in spec:
            paths = spec["video_frames"]
            if not isinstance(paths, list) or not paths:
                raise ValueError("video_frames must be a nonempty ordered list of local image paths")
            self._frames = [self._read_image(path) for path in paths]
            if len({frame.size for frame in self._frames}) != 1:
                raise ValueError("all video frames must have the same dimensions")
        scales = spec.get("input_scales", DEFAULT_SCALES[self._scale_modality])
        if not isinstance(scales, list) or not scales:
            raise ValueError("input_scales must be a nonempty list")
        self._scales = sorted(set(self._scale(value) for value in scales))

    def _read_asset(self, value: Any, modality: str) -> bytes:
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError("asset path must name a local file")
        path = (self._base / value).resolve()
        data = path.read_bytes()
        self._assets.append({"path": value, "modality": modality, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return data

    def _read_image(self, path: str) -> Any:
        from PIL import Image
        with Image.open(io.BytesIO(self._read_asset(path, "image"))) as image:
            return image.convert("RGB")

    def _scale(self, value: Any) -> float:
        return _positive_number(value, self.input_scale_type, integer=self._scale_modality != "audio")

    def generate(self, scale_value: float) -> Dict[str, Any]:
        from PIL import Image
        scale = self._scale(scale_value)
        sample = {"text": self._text}
        if "image" in self._modalities:
            resolution = int(scale) if self._scale_modality == "image" else self._image_resolution
            sample["image_base64"] = _png(self._image.resize((resolution, resolution), Image.Resampling.BICUBIC))
            if self.task_type == "document-question-answering":
                sample.update(words=copy.deepcopy(self._words), boxes=copy.deepcopy(self._boxes))
        if "audio" in self._modalities:
            duration = scale if self._scale_modality == "audio" else self._audio_duration
            count = round(duration * self._sample_rate)
            if count <= 0 or count * 2 > len(self._pcm):
                raise ValueError("requested audio duration exceeds the available waveform duration")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self._sample_rate)
                output.writeframes(self._pcm[:count * 2])
            sample.update(audio_base64=base64.b64encode(buffer.getvalue()).decode("ascii"), sampling_rate=self._sample_rate)
            if self._scale_modality == "audio":
                scale = count / self._sample_rate
        if "video" in self._modalities:
            count = int(scale) if self._scale_modality == "video" else self._video_count
            if self._frames is not None and count > len(self._frames):
                raise ValueError("requested frame_count exceeds available video_frames")
            source_frames = self._frames[:count] if self._frames is not None else [_scene(frame=i)[0] for i in range(count)]
            frames = [frame.resize((self._image_resolution, self._image_resolution), Image.Resampling.BICUBIC) for frame in source_frames]
            sample.update(video_frames_base64=[_png(frame) for frame in frames], fps=self._fps)
        return {"samples": [sample], "params": copy.deepcopy(self._params), "input_scale": scale, "input_scale_type": self.input_scale_type}

    def scale_label(self, scale_value: float) -> str:
        return f"{self.input_scale_type}{self._scale(scale_value):g}"

    def effective_input_scale(self, scale_value: float, payload: Optional[Dict[str, Any]] = None) -> float:
        return self._scale((payload or self.generate(scale_value))["input_scale"])

    def default_input_scales(self) -> List[float]:
        return list(self._scales)

    def max_input_scale(self) -> float:
        return max(self._scales)

    def plan_metadata(self) -> Dict[str, Any]:
        fixed_media = {}
        if "image" in self._modalities and self._scale_modality != "image" or "video" in self._modalities:
            fixed_media["image_resolution"] = self._image_resolution
        if "audio" in self._modalities and self._scale_modality != "audio":
            fixed_media["audio_duration_s"] = self._audio_duration
        if "video" in self._modalities:
            fixed_media["fps"] = self._fps
            if self._scale_modality != "video":
                fixed_media["video_num_frames"] = self._video_count
        return {
            "workload_id": self._spec.get("workload_id", f"multimodal-{self.task_type}-v1"),
            "input_scale_type": self.input_scale_type,
            "scale_modality": self._scale_modality,
            "modalities": list(self._modalities),
            "text": self._text,
            "params": copy.deepcopy(self._params),
            "assets": copy.deepcopy(self._assets),
            "manifest_sha256": self._manifest_sha256,
            "provenance": copy.deepcopy(self._spec.get("provenance", {})),
            "construction": "fixed image resized square / audio prefix / ordered video frame prefix resized square",
            "fixed_media": fixed_media,
            "synthetic_visual_version": 1,
            "input_num_samples_semantics": "one multimodal request example, not audio samples or video frames",
            "processor_scale_caveat": "raw media dimensions; model processor may resize, tile, pad or limit context",
        }

    def input_metadata(self, scale_value: float, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        materialized = payload or self.generate(scale_value)
        sample = materialized["samples"][0]
        result = {"input_num_samples": 1, "input_scale_type": self.input_scale_type, "payload_sha256": hashlib.sha256(json.dumps(materialized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        if "image_base64" in sample:
            from PIL import Image
            with Image.open(io.BytesIO(base64.b64decode(sample["image_base64"]))) as image:
                result.update(image_width=image.width, image_height=image.height)
        if "audio_base64" in sample:
            with wave.open(io.BytesIO(base64.b64decode(sample["audio_base64"]))) as audio:
                result.update(audio_num_samples=audio.getnframes(), sampling_rate=audio.getframerate(), audio_duration_s=audio.getnframes() / audio.getframerate())
        if "video_frames_base64" in sample:
            result.update(video_num_frames=len(sample["video_frames_base64"]), fps=sample["fps"])
        return result


register_generator("multimodal", MultimodalWorkloadGenerator)
