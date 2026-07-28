"""One-shot in-container inference runner for vendor compute profilers."""
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.metadata
import json
import math
import os
import time
from typing import Any, Dict, Optional

# Compute profiling must use the same offline model artifact as normal inference.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from acprof.container.handlers import HandlerRegistry, resolve_model_source


def _find_payload(payload_file: str, input_scale: float) -> Dict[str, Any]:
    with open(payload_file, "r", encoding="utf-8") as f:
        plan = json.load(f)

    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(f"invalid compute payload file: {payload_file}")

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            scale = float(entry.get("input_scale"))
        except (TypeError, ValueError):
            continue
        if math.isclose(scale, float(input_scale), rel_tol=0.0, abs_tol=1e-6):
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError(f"compute payload entry has no payload for scale={input_scale}")
            return payload

    raise RuntimeError(f"compute payload not found for scale={input_scale}")


class _ITTControl:
    def __init__(self) -> None:
        self._lib: Optional[Any] = None
        self._resume: Optional[Any] = None
        self._pause: Optional[Any] = None
        candidates = self._library_candidates()
        for name in candidates:
            if name is not None and not os.path.exists(name) and os.sep in name:
                continue
            try:
                lib = ctypes.CDLL(name) if name else ctypes.CDLL(None)
                resume = getattr(lib, "__itt_resume")
                pause = getattr(lib, "__itt_pause")
                self._lib = lib
                self._resume = resume
                self._pause = pause
                return
            except Exception:
                self._lib = None
                self._resume = None
                self._pause = None

    @staticmethod
    def _library_candidates() -> list[Optional[str]]:
        raw_candidates = [
            os.getenv("ADVISOR_ITT_LIB") or "",
            os.getenv("INTEL_LIBITTNOTIFY64") or "",
            os.getenv("INTEL_JIT_PROFILER64") or "",
            "/opt/intel/oneapi/advisor/latest/lib64/runtime/libittnotify_collector.so",
            *glob.glob("/opt/intel/oneapi/advisor/*/lib64/runtime/libittnotify_collector.so"),
            "/opt/intel/oneapi/advisor/latest/lib64/runtime/libittnotify.so",
            *glob.glob("/opt/intel/oneapi/advisor/*/lib64/runtime/libittnotify.so"),
            None,
            "libittnotify.so",
        ]
        candidates: list[Optional[str]] = []
        seen = set()
        for name in raw_candidates:
            if name == "":
                continue
            if name in seen:
                continue
            candidates.append(name)
            seen.add(name)
        return candidates

    def resume(self) -> None:
        if self._resume is None:
            return
        try:
            self._resume()
        except Exception:
            pass

    def pause(self) -> None:
        if self._pause is None:
            return
        try:
            self._pause()
        except Exception:
            pass


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sum_profiled_flops(prof: Any) -> float:
    total = 0.0
    for event in prof.key_averages():
        try:
            flops = float(getattr(event, "flops", 0) or 0)
        except (TypeError, ValueError):
            flops = 0.0
        if flops > 0:
            total += flops
    return total


def _run_torch_flop_profile(handler: Any, model_ctx: Any, processed: Any, repeat: int) -> float:
    # PyTorch derives FLOPs from operator shapes, so CPU activity is enough for
    # both CPU and CUDA tensors and avoids CUPTI/performance-counter privileges.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(repeat):
            handler.predict(model_ctx, processed)
            _cuda_synchronize()
    return _sum_profiled_flops(prof)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return "unknown"


def _gpu_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "gpu_compute_capability": "",
        "gpu_sm_count": None,
    }
    if not torch.cuda.is_available():
        return metadata
    try:
        capability = torch.cuda.get_device_capability()
        metadata["gpu_compute_capability"] = (
            f"{int(capability[0])}.{int(capability[1])}"
        )
    except Exception:
        pass
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        metadata["gpu_sm_count"] = int(properties.multi_processor_count)
    except Exception:
        pass
    return metadata


def _attention_implementation(model_ctx: Any) -> str:
    candidates = []
    if isinstance(model_ctx, dict):
        for key in ("model", "pipeline"):
            candidate = model_ctx.get(key)
            if candidate is not None:
                candidates.append(candidate)
    else:
        candidates.append(model_ctx)

    expanded = list(candidates)
    for candidate in candidates:
        nested_model = getattr(candidate, "model", None)
        if nested_model is not None:
            expanded.append(nested_model)

    for candidate in expanded:
        config = getattr(candidate, "config", None)
        if config is None:
            continue
        for attribute in (
            "_attn_implementation",
            "_attn_implementation_internal",
            "attn_implementation",
        ):
            try:
                value = getattr(config, attribute, None)
            except Exception:
                value = None
            if value:
                return str(value)
    return ""


def _verify_eager_attention(model_ctx: Any) -> str:
    implementation = _attention_implementation(model_ctx)
    if implementation != "eager":
        detail = implementation or "unavailable"
        raise RuntimeError(
            "torch_profiler_eager_attention_verification_failed:"
            f"expected=eager,actual={detail}"
        )
    return implementation


def _load_options_for_profile_mode(
    profile_mode: str,
) -> Optional[Dict[str, str]]:
    if profile_mode in {
        "torch_cpu",
        "torch_gpu",
        "torch_eager_cpu",
        "torch_eager_gpu",
    }:
        return {"attention_implementation": "eager"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--input-scale", required=True, type=float)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument(
        "--profile-mode",
        required=True,
        choices=(
            "cpu",
            "gpu",
            "torch_cpu",
            "torch_gpu",
            "torch_eager_cpu",
            "torch_eager_gpu",
        ),
    )
    args = parser.parse_args()

    model_id = os.getenv("MODEL_ID", "")
    model_revision = os.getenv("MODEL_REVISION", "main")
    model_source = resolve_model_source(model_id, os.getenv("MODEL_LOCAL_PATH"))
    task_family = os.getenv("TASK_FAMILY", "nlp")
    task_type = os.getenv("TASK_TYPE", os.getenv("PIPELINE_TAG", "text-generation"))
    runtime_backend = os.getenv("RUNTIME_BACKEND", "transformers_pipeline")
    use_gpu = int(os.getenv("USE_GPU", "0"))
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "0") or "0")
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    handler = HandlerRegistry.get(task_family, runtime_backend)
    load_options = _load_options_for_profile_mode(args.profile_mode)
    torch_eager_mode = load_options is not None
    t_load = time.perf_counter()
    handler_load_kwargs: Dict[str, Any] = {}
    if load_options is not None:
        handler_load_kwargs["load_options"] = load_options
    model_ctx = handler.load(
        model_source,
        task_type,
        runtime_backend,
        device,
        model_revision,
        **handler_load_kwargs,
    )
    attention_implementation = (
        _verify_eager_attention(model_ctx)
        if torch_eager_mode
        else ""
    )
    payload = _find_payload(args.payload_file, args.input_scale)
    processed = handler.preprocess(model_ctx, payload)

    with torch.inference_mode():
        handler.predict(model_ctx, processed)
        _cuda_synchronize()

        repeat = max(1, int(args.repeat))
        if torch_eager_mode:
            total_flops = _run_torch_flop_profile(handler, model_ctx, processed, repeat)
            _cuda_synchronize()
            elapsed_s = time.perf_counter() - t_load
            print(json.dumps({
                "status": "ok",
                "model_id": model_id,
                "model_revision": model_revision,
                "device": device,
                "input_scale": args.input_scale,
                "repeat": repeat,
                "elapsed_s": elapsed_s,
                "total_flops": total_flops,
                "model_logical_mflop_per_request_torch_profiler_eager": (
                    total_flops / 1_000_000.0
                ) / float(repeat),
                "profile_tool": "torch_profiler_eager",
                "attention_implementation": attention_implementation,
                "attention_implementation_verified": True,
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "transformers_version": _package_version("transformers"),
                **_gpu_metadata(),
            }))
            return

        profile_window_t0 = time.perf_counter()
        if args.profile_mode == "gpu":
            _cuda_synchronize()
            torch.cuda.nvtx.range_push("acprof_compute")
            try:
                for _ in range(repeat):
                    handler.predict(model_ctx, processed)
                _cuda_synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            itt = _ITTControl()
            itt.resume()
            try:
                for _ in range(repeat):
                    handler.predict(model_ctx, processed)
            finally:
                itt.pause()
        profile_window_wall_time_ms = (
            time.perf_counter() - profile_window_t0
        ) * 1000.0

    elapsed_s = time.perf_counter() - t_load
    print(json.dumps({
        "status": "ok",
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "input_scale": args.input_scale,
        "repeat": max(1, int(args.repeat)),
        "elapsed_s": elapsed_s,
        "profile_window_wall_time_ms": profile_window_wall_time_ms,
        "profile_window_wall_time_ms_per_request": (
            profile_window_wall_time_ms / float(repeat)
        ),
        **_gpu_metadata(),
    }))


if __name__ == "__main__":
    main()
