"""按离线加载器的文件约定规划下载；只依赖标准库，不加载模型。

选择顺序参考 Transformers 4.57.6 modeling_utils.py 和 Diffusers 0.39.0
pipeline_utils.py（Apache-2.0）。不调用会自行联网/转换权重的私有接口。
未覆盖的结构保留完整快照；已选择的分片缺失属于错误，不能静默换权重。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Callable


PLAN_VERSION = 1
PLAN_FILENAME = "model_download_plan.json"
DIFFUSION_PIPELINES = {
    "StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline", "StableDiffusionXLPipeline",
    "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline",
    "DDPMPipeline", "DDIMPipeline",
}
_WEIGHTS = re.compile(
    r"(?:model|pytorch_model|diffusion_pytorch_model)"
    r"(?:\.(?:fp16|fp32|bf16|non_ema))?(?:-\d+-of-\d+)?"
    r"\.(?:safetensors|bin)(?:\.index\.json)?$"
)


class ModelFilesError(ValueError):
    """模型文件声明不完整或不合法；重试下载不能修复这个错误。"""


def safe_path(name: str) -> str:
    if not isinstance(name, str):
        raise ModelFilesError(f"invalid model file path: {name!r}")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ModelFilesError(f"invalid model file path: {name!r}")
    return name


def seal_plan(plan: dict) -> dict:
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_VERSION:
        raise ModelFilesError("unsupported model download plan schema")
    expected = plan.get("plan_sha256")
    if not expected or seal_plan(dict(plan))["plan_sha256"] != expected:
        raise ModelFilesError("model download plan hash mismatch")
    files = plan.get("files")
    if not isinstance(files, list) or not files or any(not isinstance(item, dict) for item in files):
        raise ModelFilesError("model download plan has no valid file list")
    paths = [safe_path(item.get("path")) for item in files]
    if len(paths) != len(set(paths)):
        raise ModelFilesError("model download plan contains duplicate file paths")
    if plan.get("verification") == "sha256":
        if any(not isinstance(item.get("size"), int) or item["size"] < 0 or
               not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) for item in files):
            raise ModelFilesError("verified model download plan has incomplete file hashes")


def _checkpoint(names: set[str], read_json: Callable, prefix: str, *, diffusion: bool = False) -> dict | None:
    stems = [("diffusion_pytorch_model", "safetensors"), ("diffusion_pytorch_model", "bin")] if diffusion else [
        ("model", "safetensors"), ("pytorch_model", "bin"),
    ]
    for stem, extension in stems:
        base = prefix + stem + "." + extension
        if base in names:
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None, "files": [base]}
        index = base + ".index.json"
        if index in names:
            data = read_json(index)
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ModelFilesError(f"invalid or empty weight_map: {index}")
            if not all(isinstance(value, str) for value in weight_map.values()):
                raise ModelFilesError(f"invalid shard filenames: {index}")
            shards = {prefix + safe_path(value) for value in weight_map.values()}
            missing = shards - names
            if missing:
                raise ModelFilesError(f"missing checkpoint shards for {index}: {', '.join(sorted(missing))}")
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None,
                    "files": [index, *sorted(shards)]}
    return None


def _alternate_weights(name: str, prefix: str) -> bool:
    if not name.startswith(prefix) or "/" in name[len(prefix):]:
        return False
    basename = name[len(prefix):]
    return bool(_WEIGHTS.fullmatch(basename)) or basename in {"flax_model.msgpack", "tf_model.h5"}


def plan_download(
    *, model_id: str, revision: str, family: str, backend: str,
    files: dict[str, dict], read_json: Callable[[str], Any], policy: str = "auto",
    adapter: str = "family-default", native_model_types: set[str] | None = None,
    library_versions: dict[str, str] | None = None,
) -> dict:
    if policy not in {"auto", "full"}:
        raise ModelFilesError("model download policy must be auto or full")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelFilesError("model download plan requires a fixed commit SHA")
    names = {safe_path(name) for name in files}
    if not names:
        raise ModelFilesError("model repository has no files")
    selected = set(names)
    weights: list[dict] = []
    reason = "explicit_full" if policy == "full" else "unsupported_layout"
    effective = "full"

    def transformer(prefix: str) -> tuple[dict | None, str]:
        config_path = prefix + "config.json"
        if config_path not in names:
            return None, "missing_transformers_config"
        config = read_json(config_path)
        if not isinstance(config, dict):
            raise ModelFilesError(f"invalid configuration: {config_path}")
        if config.get("auto_map") or config.get("quantization_config") or "adapter_config.json" in names:
            return None, "custom_or_quantized_model"
        model_type = config.get("model_type")
        if not isinstance(model_type, str) or model_type not in (native_model_types or set()):
            return None, "unregistered_model_type"
        checkpoint = _checkpoint(names, read_json, prefix)
        return checkpoint, "native_transformers" if checkpoint else "nonstandard_checkpoint"

    if policy == "auto" and adapter != "family-default":
        reason = "custom_adapter"
    elif policy == "auto" and backend in {"transformers_pipeline", "transformers_model", "sentence_transformers", "chronos"}:
        if "modules.json" in names:
            modules = read_json("modules.json")
            if not isinstance(modules, list) or not modules:
                raise ModelFilesError("invalid sentence-transformers modules.json")
            prefixes = []
            for module in modules:
                if not isinstance(module, dict) or not str(module.get("type", "")).startswith("sentence_transformers.models."):
                    reason = "custom_sentence_transformers_module"
                    break
                if module["type"] == "sentence_transformers.models.Transformer":
                    path = module.get("path", "")
                    prefixes.append(safe_path(path) + "/" if path else "")
            else:
                candidates = [transformer(prefix) for prefix in prefixes]
                if candidates and all(item[0] for item in candidates):
                    weights = [item[0] for item in candidates]
                    reason = "sentence_transformers_modules"
                else:
                    reason = "unsupported_sentence_transformers_layout"
        else:
            checkpoint, reason = transformer("")
            if checkpoint:
                weights = [checkpoint]
        if weights:
            for checkpoint in weights:
                prefix = "" if checkpoint["component"] == "." else checkpoint["component"] + "/"
                selected -= {name for name in names if _alternate_weights(name, prefix)} - set(checkpoint["files"])
            effective = "selected"
    elif policy == "auto" and backend == "diffusers" and "model_index.json" in names:
        config = read_json("model_index.json")
        if not isinstance(config, dict):
            raise ModelFilesError("invalid model_index.json")
        pipeline = config.get("_class_name")
        if isinstance(pipeline, str) and pipeline in DIFFUSION_PIPELINES:
            reason = "native_diffusers_components"
            for component, spec in config.items():
                if component.startswith("_") or not isinstance(spec, (list, dict)):
                    continue
                if not isinstance(spec, list) or len(spec) != 2:
                    reason = "custom_diffusers_component"
                    break
                library, class_name = spec
                if library is None and class_name is None:
                    continue
                if (not isinstance(library, str) or not isinstance(class_name, str) or
                        library not in {"diffusers", "transformers", "stable_diffusion"}):
                    reason = "custom_diffusers_component"
                    break
                prefix = safe_path(component) + "/"
                component_names = {name for name in names if name.startswith(prefix)}
                if not component_names:
                    raise ModelFilesError(f"missing pipeline component: {component}")
                has_weights = any(_alternate_weights(name, prefix) for name in component_names)
                if not has_weights:
                    if not any(marker in str(class_name) for marker in ("Scheduler", "Tokenizer", "Processor", "FeatureExtractor")):
                        reason = "nonstandard_diffusers_component"
                        break
                    continue
                checkpoint = _checkpoint(names, read_json, prefix, diffusion=library == "diffusers")
                if checkpoint is None:
                    reason = "nonstandard_diffusers_checkpoint"
                    break
                weights.append(checkpoint)
            else:
                if weights:
                    for checkpoint in weights:
                        prefix = checkpoint["component"] + "/"
                        selected -= {name for name in names if _alternate_weights(name, prefix)} - set(checkpoint["files"])
                    selected -= {name for name in names if "/" not in name and name.endswith((".ckpt", ".safetensors"))}
                    effective = "selected"
        else:
            reason = "unregistered_diffusers_pipeline"
    elif policy == "auto" and backend in {"torchscript", "skops"}:
        manifest = read_json("acprof_model.json") if "acprof_model.json" in names else None
        artifact = None
        if manifest is not None:
            if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("format") != backend:
                raise ModelFilesError("invalid acprof_model.json format/schema")
            artifact = safe_path(manifest.get("model_file", ""))
            if artifact not in names:
                raise ModelFilesError(f"missing structured model file: {artifact}")
        elif backend == "skops":
            candidates = [name for name in names if "/" not in name and name.endswith(".skops")]
            if len(candidates) == 1:
                artifact = candidates[0]
        if artifact:
            selected = {artifact} | ({"acprof_model.json"} & names) | {
                name for name in names if PurePosixPath(name).name.upper().startswith(("LICENSE", "NOTICE", "README"))
            }
            weights = [{"component": ".", "format": backend, "variant": None, "files": [artifact]}]
            effective, reason = "selected", "structured_artifact_contract"

    if effective == "full":
        selected, weights = names, []
    def entries(chosen: set[str]) -> list[dict]:
        return [{"path": name, **files[name]} for name in sorted(chosen)]

    selected_entries = entries(selected)
    sizes = [item.get("size") for item in selected_entries]
    return seal_plan({
        "schema_version": PLAN_VERSION, "model_id": model_id, "model_revision": revision,
        "task_family": family, "backend": backend, "adapter": adapter,
        "requested_policy": policy, "effective_policy": effective, "reason": reason,
        "library_versions": library_versions or {}, "weights": weights,
        "files": selected_entries, "excluded_files": entries(names - selected),
        "selected_bytes": sum(sizes) if all(isinstance(size, int) for size in sizes) else None,
    })
