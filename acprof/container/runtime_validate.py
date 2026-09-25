"""独立容器中的完整接口验证，不产生性能测量行。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any


RESULT_PREFIX = "ACPROF_RUNTIME_VALIDATION="


def validate(payload: dict, *, stages: list[dict] | None = None) -> dict:
    from acprof.container.handlers import HandlerRegistry, load_handler, resolve_model_source
    from acprof.container.execution import complete_prediction, configured_execution
    from acprof.runtime_settings import runtime_threads
    from acprof.extensions import get_extension

    stages = [] if stages is None else stages

    def check(stage, operation):
        try:
            result = operation()
        except Exception as exc:
            stages.append({"stage": stage, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            # Preserve typed errors, especially TimeoutError used by callers.
            raise
        stages.append({"stage": stage, "status": "verified"})
        return result

    use_gpu = os.getenv("USE_GPU", "0") == "1"
    execution, device = check("execution", lambda: configured_execution(
        os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"], use_gpu=use_gpu,
        threads=runtime_threads(default=1),
        adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
    ))

    def load():
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
        return handler, context

    handler, context = check("load", load)
    processed = check("preprocess", lambda: handler.preprocess(context, payload))
    with execution.inference_context():
        output = check("predict", lambda: handler.predict(context, processed))
        output = check("completion", lambda: complete_prediction(execution, context, output))
    response = check("postprocess", lambda: handler.postprocess(context, output))

    def validate_output():
        validation = handler.validate_output(context, payload, processed, output, response)
        for layer in ('protocol', 'task'):
            if not isinstance(validation, dict) or not isinstance(validation.get(layer), dict) or validation[layer].get('status') != 'verified':
                raise ValueError(f'{layer} validation must be verified before profiling: {validation!r}')
        return validation

    validation = check("validate_output", validate_output)
    runtime_metadata = check("metadata", execution.metadata)
    return {
        "status": "ok", "mode": "full", "device": device, "stages": stages,
        "dtype": str(getattr(context.get("model"), "dtype", context.get("dtype", "unknown"))),
        "attention_implementation": context.get("attention_implementation", "model_default"),
        **runtime_metadata, "validation": validation,
        "runtime_parameters": context.get("runtime_parameters", runtime_metadata.get("runtime_parameters", {})),
        "artifact": context.get("artifact", {}),
        "model_spec": context.get("model_spec", context.get("manifest", {})),
        "workload_contract": validation["workload_contract"],
        "adapter": type(handler).__name__, "response": response,
        "effective_input_scale": processed.get("_effective_input_scale") if isinstance(processed, dict) else None,
    }


def main() -> int:
    stages = []
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            payload = json.load(stream)
        if os.getenv("ACPROF_CONTRACT_PROBE_MODE", "full") == "basic":
            from acprof.container.model_probe import validate_basic
            result = validate_basic(payload)
        else:
            result = validate(payload, stages=stages)
    except Exception as exc:
        traceback.print_exc()
        result: dict[str, Any] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result.update(failed_stage=stages[-1]["stage"] if stages and stages[-1]["status"] == "error" else "input_or_execution_context",
                      stages=stages)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
