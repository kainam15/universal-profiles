"""独立容器中的完整接口验证，不产生性能测量行。"""

from __future__ import annotations

import json
import os
import sys
import traceback


RESULT_PREFIX = "ACPROF_RUNTIME_VALIDATION="


def validate(payload: dict) -> dict:
    import torch
    from acprof.container.handlers import HandlerRegistry, resolve_model_source

    use_gpu = os.getenv("USE_GPU", "0") == "1"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("请求 GPU 验证，但容器内 CUDA 不可用")
    device = "cuda" if use_gpu else "cpu"
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
    handler = HandlerRegistry.get(os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"])
    context = handler.load(
        resolve_model_source(os.environ["MODEL_ID"]), os.environ["TASK_TYPE"],
        os.environ["RUNTIME_BACKEND"], device, os.environ["MODEL_REVISION"],
    )
    processed = handler.preprocess(context, payload)
    with torch.inference_mode():
        output = handler.predict(context, processed)
    response = handler.postprocess(context, output)
    if not isinstance(response, dict) or not response:
        raise ValueError("postprocess 必须返回非空 JSON object")
    # The same JSON transport contract as Flask /predict, without an HTTP server.
    json.dumps(response, allow_nan=False)
    if response.get("error"):
        raise ValueError(f"模型响应包含 error: {response['error']}")
    return {
        "status": "ok", "device": device,
        "dtype": str(getattr(context.get("model"), "dtype", "unknown")),
        "attention_implementation": context.get("attention_implementation", "model_default"),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "adapter": type(handler).__name__, "response": response,
        "effective_input_scale": processed.get("_effective_input_scale") if isinstance(processed, dict) else None,
    }


def main() -> int:
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            result = validate(json.load(stream))
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
