"""独立容器中的完整接口验证，不产生性能测量行。"""

from __future__ import annotations

import json
import os
import sys
import traceback


RESULT_PREFIX = "ACPROF_RUNTIME_VALIDATION="


def validate(payload: dict) -> dict:
    from acprof.container.handlers import HandlerRegistry, load_handler, resolve_model_source
    from acprof.container.execution import configured_execution
    from acprof.extensions import get_extension

    use_gpu = os.getenv("USE_GPU", "0") == "1"
    execution, device = configured_execution(
        os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"], use_gpu=use_gpu,
        threads=max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))),
        adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
    )
    handler = HandlerRegistry.get(os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"])
    context = load_handler(handler,
        resolve_model_source(os.environ["MODEL_ID"]), os.environ["TASK_TYPE"],
        os.environ["RUNTIME_BACKEND"], device, os.environ["MODEL_REVISION"],
    )
    context["_validation_entrypoint"] = get_extension(
        os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"],
        adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
        task=os.environ["TASK_TYPE"],
    ).validation_entrypoint
    processed = handler.preprocess(context, payload)
    with execution.inference_context():
        output = handler.predict(context, processed)
    response = handler.postprocess(context, output)
    validation = handler.validate_output(context, payload, processed, output, response)
    for layer in ('protocol', 'task'):
        if not isinstance(validation, dict) or not isinstance(validation.get(layer), dict) or validation[layer].get('status') != 'verified':
            raise ValueError(f'{layer} validation must be verified before profiling: {validation!r}')
    return {
        "status": "ok", "device": device,
        "dtype": str(getattr(context.get("model"), "dtype", context.get("dtype", "unknown"))),
        "attention_implementation": context.get("attention_implementation", "model_default"),
        **execution.metadata(), "validation": validation,
        "workload_contract": validation["workload_contract"],
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
