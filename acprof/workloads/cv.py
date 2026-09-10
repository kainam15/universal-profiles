"""CV workload generator - synthetic images at various resolutions."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from acprof.workloads import WorkloadGenerator, register_generator

BASE_SEED = 12345
BASE_RESOLUTION = 224  # Default base resolution for most vision models
CV_TASKS = {
    "depth-estimation", "image-classification", "object-detection", "image-segmentation",
    "image-to-text", "zero-shot-image-classification", "image-feature-extraction",
    "zero-shot-object-detection", "mask-generation", "keypoint-detection", "video-classification",
}
ZERO_SHOT_TASKS = {"zero-shot-image-classification", "zero-shot-object-detection"}


class CVWorkloadGenerator(WorkloadGenerator):
    """Generate synthetic RGB images at scaled resolutions."""

    def __init__(self, model_id: str, task_type: str, batch_size: int,
                 workload_spec_path: Optional[str] = None):
        super().__init__(model_id, task_type, batch_size)
        if task_type not in CV_TASKS:
            raise ValueError(f"unsupported CV task_type={task_type!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size != 1:
            raise ValueError(f"{task_type} requires batch_size=1")
        self._base_res = BASE_RESOLUTION
        self._params: Dict[str, Any] = {}
        self._labels = ["cat", "dog", "car", "person"]
        self._boxes = [[0.0, 0.0, 1.0, 1.0]]
        self._num_frames = 16
        self._assets: List[bytes] = []
        self._sources: List[str] = []
        self._scales: Optional[List[float]] = None
        self._workload_id = f"synthetic-{task_type}-v1"
        self._spec_sha256: Optional[str] = None
        if workload_spec_path is not None:
            self._load_spec(Path(workload_spec_path).expanduser().resolve())

    def _load_spec(self, path: Path) -> None:
        content = path.read_bytes()
        spec = json.loads(content)
        if not isinstance(spec, dict):
            raise ValueError("CV workload spec must be an object")
        allowed = {"schema_version", "workload_id", "image_path", "image_sha256", "video_frames",
                   "num_frames", "candidate_labels", "boxes", "params", "input_scales"}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError("unknown CV workload keys: " + ", ".join(unknown))
        if spec.get("schema_version", 1) != 1 or isinstance(spec.get("schema_version"), bool):
            raise ValueError("CV workload schema_version must be 1")
        self._spec_sha256 = hashlib.sha256(content).hexdigest()
        if "workload_id" in spec:
            if not isinstance(spec["workload_id"], str) or not spec["workload_id"].strip():
                raise ValueError("workload_id must be a non-empty string")
            self._workload_id = spec["workload_id"]
        if "input_scales" in spec:
            scales = spec["input_scales"]
            if not isinstance(scales, list) or not scales:
                raise ValueError("input_scales must be a non-empty array")
            for scale in scales:
                self._resolution(scale)
            self._scales = [float(scale) for scale in scales]
            if len(set(self._scales)) != len(self._scales):
                raise ValueError("input_scales must not contain duplicates")
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        reserved = set(params) & {"batch_size", "num_workers", "candidate_labels", "num_frames", "boxes"}
        if reserved:
            raise ValueError("unsupported CV params: " + ", ".join(sorted(reserved)))
        self._params = copy.deepcopy(params)
        if "candidate_labels" in spec:
            labels = spec["candidate_labels"]
            if self.task_type not in ZERO_SHOT_TASKS:
                raise ValueError("candidate_labels requires a zero-shot CV task")
            if not isinstance(labels, list) or not labels or any(
                not isinstance(label, str) or not label.strip() for label in labels
            ):
                raise ValueError("candidate_labels must be a non-empty list of non-empty strings")
            self._labels = list(labels)
        if "boxes" in spec:
            if self.task_type != "keypoint-detection":
                raise ValueError("boxes requires keypoint-detection")
            boxes = spec["boxes"]
            if not isinstance(boxes, list) or not boxes:
                raise ValueError("boxes must be a non-empty array of normalized xywh boxes")
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in box
                ):
                    raise ValueError("boxes must contain normalized xywh boxes")
                x, y, width, height = box
                if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
                    raise ValueError("normalized boxes must fit within the image")
            self._boxes = copy.deepcopy(boxes)
        if self.task_type == "video-classification":
            if "image_path" in spec or "image_sha256" in spec:
                raise ValueError("video-classification accepts video_frames instead of image_path")
            if "num_frames" in spec:
                count = spec["num_frames"]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError("num_frames must be a positive integer")
                self._num_frames = count
            if "video_frames" in spec:
                paths = spec["video_frames"]
                if not isinstance(paths, list) or not paths:
                    raise ValueError("video_frames must be a non-empty ordered array of local image paths")
                if "num_frames" in spec and len(paths) != self._num_frames:
                    raise ValueError("num_frames must match the number of video_frames")
                self._num_frames = len(paths)
                for item in paths:
                    self._read_asset(path, item)
        else:
            if "video_frames" in spec or "num_frames" in spec:
                raise ValueError("video_frames and num_frames require video-classification")
            if "image_sha256" in spec and "image_path" not in spec:
                raise ValueError("image_sha256 requires image_path")
            if "image_path" in spec:
                self._read_asset(path, spec["image_path"])
                if "image_sha256" in spec and spec["image_sha256"] != hashlib.sha256(self._assets[0]).hexdigest():
                    raise ValueError("image SHA256 does not match image_sha256")

    def _read_asset(self, spec_path: Path, value: Any) -> None:
        from PIL import Image

        if not isinstance(value, str) or not value.strip():
            raise ValueError("image path must be a non-empty local path")
        asset_path = (spec_path.parent / value).resolve()
        content = asset_path.read_bytes()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
        self._assets.append(content)
        self._sources.append(str(asset_path))

    @staticmethod
    def _resolution(scale_value: float) -> int:
        if isinstance(scale_value, bool) or not isinstance(scale_value, (int, float)):
            raise ValueError("CV input scale must be a finite positive resolution multiplier")
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise ValueError("CV input scale must be a finite positive resolution multiplier")
        return max(1, int(BASE_RESOLUTION * scale_value))

    def _generate_image_base64(self, width: int, height: int, frame_index: int = 0) -> str:
        from PIL import Image

        # Generate a synthetic image with deterministic random pixels
        img = Image.new("RGB", (width, height))
        pixels = []
        rng = random.Random(BASE_SEED + width * 1000 + height + frame_index * 1000003)
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
        from PIL import Image

        resolution = self._resolution(scale_value)
        count = self._num_frames if self.task_type == "video-classification" else 1
        images = []
        for index in range(count):
            if self._assets:
                with Image.open(io.BytesIO(self._assets[index])) as source:
                    image = source.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            else:
                images.append(self._generate_image_base64(resolution, resolution, index))
        payload = {"params": copy.deepcopy(self._params), "input_scale": float(scale_value)}
        if self.task_type == "video-classification":
            payload["frames_base64"] = images
        else:
            payload["image_base64"] = images[0]
        if self.task_type in ZERO_SHOT_TASKS:
            payload["candidate_labels"] = list(self._labels)
        if self.task_type == "keypoint-detection":
            payload["boxes"] = [[value * resolution for value in box] for box in self._boxes]
        return payload

    def scale_label(self, scale_value: float) -> str:
        return f"res{scale_value}"

    def default_input_scales(self) -> Optional[List[float]]:
        return list(self._scales) if self._scales is not None else None

    def plan_metadata(self) -> Dict[str, Any]:
        metadata = {
            "workload_id": self._workload_id,
            "workload_spec_sha256": self._spec_sha256,
            "input_scale_type": "resolution_scale",
            "input_scale_semantics": "square image side = max(1, int(224 * scale)); processor may resize",
            "base_resolution": BASE_RESOLUTION,
            "seed": BASE_SEED,
            "source_paths": list(self._sources),
            "source_sha256": [hashlib.sha256(asset).hexdigest() for asset in self._assets],
            "params": copy.deepcopy(self._params),
        }
        if self.task_type in ZERO_SHOT_TASKS:
            metadata["candidate_labels"] = list(self._labels)
        if self.task_type == "video-classification":
            metadata.update(num_frames=self._num_frames, frame_sampling="all provided frames, in order")
        if self.task_type == "keypoint-detection":
            metadata.update(boxes=copy.deepcopy(self._boxes), boxes_format="normalized xywh; used by VitPose")
        return metadata

    def input_metadata(self, scale_value: float, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self.generate(scale_value) if payload is None else payload
        resolution = self._resolution(scale_value)
        metadata = {"image_width": resolution, "image_height": resolution, "input_num_samples": 1}
        frames = payload.get("frames_base64")
        if frames is not None:
            metadata.update(num_frames=len(frames), frame_sha256=[
                hashlib.sha256(base64.b64decode(frame)).hexdigest() for frame in frames
            ])
        else:
            metadata["image_sha256"] = hashlib.sha256(base64.b64decode(payload["image_base64"])).hexdigest()
        return metadata


register_generator("cv", CVWorkloadGenerator)
