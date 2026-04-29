"""One-shot in-container inference runner for vendor compute profilers."""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import time
from typing import Any, Dict, Optional

import torch

from handlers import HandlerRegistry


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
        candidates = [
            None,
            os.getenv("ADVISOR_ITT_LIB") or "",
            "/opt/intel/oneapi/advisor/latest/lib64/runtime/libittnotify.so",
            *glob.glob("/opt/intel/oneapi/advisor/*/lib64/runtime/libittnotify.so"),
            "libittnotify.so",
        ]
        for name in candidates:
            if name and not os.path.exists(name) and os.sep in name:
                continue
            try:
                self._lib = ctypes.CDLL(name) if name else ctypes.CDLL(None)
                getattr(self._lib, "__itt_resume")
                getattr(self._lib, "__itt_pause")
                return
            except Exception:
                self._lib = None

    def resume(self) -> None:
        if self._lib is None:
            return
        try:
            self._lib.__itt_resume()
        except Exception:
            pass

    def pause(self) -> None:
        if self._lib is None:
            return
        try:
            self._lib.__itt_pause()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--input-scale", required=True, type=float)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument(
        "--profile-mode",
        required=True,
        choices=("cpu", "gpu", "torch_cpu", "torch_gpu"),
    )
    args = parser.parse_args()

    model_id = os.getenv("MODEL_ID", "")
    task_family = os.getenv("TASK_FAMILY", "nlp")
    task_type = os.getenv("TASK_TYPE", os.getenv("PIPELINE_TAG", "text-generation"))
    runtime_backend = os.getenv("RUNTIME_BACKEND", "transformers_pipeline")
    use_gpu = int(os.getenv("USE_GPU", "0"))
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    torch_threads = int(os.getenv("TORCH_NUM_THREADS", "0") or "0")
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    handler = HandlerRegistry.get(task_family, runtime_backend)
    t_load = time.perf_counter()
    model_ctx = handler.load(model_id, task_type, runtime_backend, device)
    payload = _find_payload(args.payload_file, args.input_scale)
    processed = handler.preprocess(model_ctx, payload)

    with torch.inference_mode():
        handler.predict(model_ctx, processed)
        _cuda_synchronize()

        repeat = max(1, int(args.repeat))
        if args.profile_mode in {"torch_cpu", "torch_gpu"}:
            total_flops = _run_torch_flop_profile(handler, model_ctx, processed, repeat)
            _cuda_synchronize()
            elapsed_s = time.perf_counter() - t_load
            print(json.dumps({
                "status": "ok",
                "model_id": model_id,
                "device": device,
                "input_scale": args.input_scale,
                "repeat": repeat,
                "elapsed_s": elapsed_s,
                "total_flops": total_flops,
                "model_mflop_per_request": (total_flops / 1_000_000.0) / float(repeat),
                "profile_tool": "torch_profiler",
            }))
            return

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

    elapsed_s = time.perf_counter() - t_load
    print(json.dumps({
        "status": "ok",
        "model_id": model_id,
        "device": device,
        "input_scale": args.input_scale,
        "repeat": max(1, int(args.repeat)),
        "elapsed_s": elapsed_s,
    }))


if __name__ == "__main__":
    main()
