"""Preparation-only inspection and isolated probes shared by CLI and TUI."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from acprof.artifacts import atomic_write_json
from acprof.model_contract import write_model_resolution


class ProbePreparationError(RuntimeError):
    """Expose an environment failure separately from model execution failures."""
    stage = "environment"


def explain_resolution(task_info, *, explain: bool = False) -> str:
    resolution = task_info.model_resolution
    contract = resolution.get("contract", {})
    lines = [f"Model: {task_info.model_id}", f"Revision: {task_info.model_revision}",
             f"Task: {task_info.pipeline_tag}", f"Backend: {task_info.runtime_backend}",
             f"Status: {contract.get('status', resolution.get('status', 'unknown'))}",
             f"Runtime: {resolution.get('runtime_validation', {}).get('status', 'not_run')}"]
    if resolution.get("semantics"):
        lines.append(f"Semantics: {resolution['semantics']['status']} ({resolution.get('benchmark_kind', 'unknown')})")
    if explain and resolution.get("provenance"):
        provenance = resolution["provenance"]
        lines.append(f"Decision: {provenance['decision']} / {provenance['identity_sha256']}")
        for item in provenance["observations"]:
            kind = f" [{item['kind']}]" if item.get("kind") else ""
            lines.append(f"  <- {item['source_id']}:{item['field']} = {item.get('value', item.get('task'))}{kind}")
            if item.get("reason"):
                lines.append("     " + item["reason"])
    names = contract.get("fields", {}) if explain else contract.get("unresolved_fields", [])
    for name in names:
        field = contract["fields"][name]
        lines.append(f"{name} [{field['state']}]: {json.dumps(field['value'], ensure_ascii=False)}")
        if field.get("reason"):
            lines.append("  " + field["reason"])
        if explain:
            lines.extend("  <- " + source for source in field.get("sources", []))
    if not contract:
        lines.extend(resolution.get("conflicts", []) + resolution.get("missing", []))
    validation = resolution.get("runtime_validation", contract.get("runtime_validation"))
    if isinstance(validation, dict):
        for device, item in validation.get("devices", {}).items():
            if item.get("error"):
                lines.append(f"{device}: {item['error']}")
    return "\n".join(lines)


def probe_model_contract(task_info, output_dir: str | Path, *, mode: str, cpus: int = 2,
                         memory_gb: int = 4, gpu: bool = False, timeout_seconds: float = 300,
                         reuse_existing: bool = False) -> dict:
    from acprof.host.docker_runtime import prepare_image
    from acprof.host.preflight import require_native_linux_host, require_native_docker
    from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
    from acprof.host.runtime_validation import validate_runtime
    from acprof.host.task_support import require_task_support
    from acprof.installation import resource_root
    from acprof.host.input_plan import _get_task_generator, resolve_input_scales

    if (mode not in {"basic", "full"} or mode == "basic" and gpu or type(cpus) is not int or cpus <= 0
            or type(memory_gb) is not int or memory_gb <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("invalid contract probe mode/resources/timeout")
    if mode == "basic" and not task_info.model_resolution.get("contract"):
        raise ValueError("basic signature probe requires a Pipeline contract; use --probe full for native models")
    require_task_support(task_info)
    require_native_linux_host()
    require_native_docker()
    root = Path(output_dir)
    with MeasurementLock(), ResultDirectoryLock(root):
        if (root / "runtime_validation.json").exists():
            raise ValueError("probe output already contains validation; choose a new output directory")
        try:
            image = prepare_image(task_info, str(resource_root()), reuse_existing=reuse_existing)
        except (RuntimeError, ValueError, OSError) as exc:
            raise ProbePreparationError(str(exc)) from exc
        write_model_resolution(task_info, root)
        payload, scale = {}, 0
        if mode == "full":
            generator = _get_task_generator(task_info, 1)
            scales = generator.default_input_scales()
            if not scales and not task_info.model_resolution.get("contract"):
                scales = resolve_input_scales(task_info.task_family)
            if not scales:
                raise ValueError("contract probe requires declared default workload scales")
            scale = min(scales)
            payload = generator.generate(scale)
            # Probe a minimal deterministic response, independent of measurement settings.
            if task_info.model_resolution.get("contract") and task_info.task_family == "multimodal":
                payload.setdefault("params", {}).update(max_new_tokens=1, do_sample=False)
        plan = root / "contract_probe_input.json"
        atomic_write_json(plan, {"schema_version": 2, "scope": "contract_probe_only",
                                "entries": [{"input_scale": scale, "payload": payload}]})
        return validate_runtime(task_info=task_info, image_info=image, planned=SimpleNamespace(plan_file=str(plan)),
                                cpu_list=[cpus], mem_list=[memory_gb], gpu_list=["on" if gpu else "off"],
                                output_dir=str(root), timeout_seconds=timeout_seconds, mode=mode)
