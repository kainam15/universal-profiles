"""Profiler applicability, collection, and plan reuse."""
from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from acprof.host.compute_profile_plan import (
    NCU_ERROR_FIELD,
    NCU_KERNEL_COUNT_FIELD,
    NCU_KERNEL_TIME_FIELD,
    NCU_PROFILE_KEY,
    NCU_SCALAR_MFLOP_FIELD,
    NCU_TENSOR_MFLOP_FIELD,
    NCU_TENSOR_SHARE_FIELD,
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_PROFILE_KEY,
    find_compute_profile_entry,
)
from acprof.host.execution_profile_plan import find_execution_profile_entry
from acprof.host.posthoc.context import (
    COMPUTE_PLAN_METRIC_FIELDS,
    PROJECT_DIR,
    PosthocError,
    ResultContext,
    SUPPORTED_TOOLS,
    TOOL_ERROR_FIELD,
    TOOL_GPU_MODES,
    TOOL_METRIC_FIELDS,
    _finite_float,
    _read_plan,
)


def parse_tools(value: str | Iterable[str]) -> Tuple[str, ...]:
    raw_values = [value] if isinstance(value, str) else list(value)
    tools: List[str] = []
    for raw in raw_values:
        for token in str(raw).replace(";", ",").split(","):
            tool = token.strip().lower()
            if not tool:
                continue
            if tool not in SUPPORTED_TOOLS:
                raise PosthocError(
                    f"unsupported tool {tool!r}; choose from {', '.join(SUPPORTED_TOOLS)}"
                )
            if tool not in tools:
                tools.append(tool)
    if not tools:
        raise PosthocError("at least one profiler tool is required")
    return tuple(tool for tool in SUPPORTED_TOOLS if tool in tools)


def applicable_tools(
    context: ResultContext,
    tools: Iterable[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    modes = {case[2] for case in context.resource_cases}
    applicable: List[str] = []
    skipped: List[str] = []
    for tool in tools:
        if modes.intersection(TOOL_GPU_MODES[tool]):
            applicable.append(tool)
        else:
            skipped.append(tool)
    return tuple(applicable), tuple(skipped)


def compute_plan_covers_ncu(
    plan: Mapping[str, Any],
    scales: Iterable[float],
) -> bool:
    if not isinstance(plan, Mapping):
        return False
    for scale in scales:
        profile = find_compute_profile_entry(dict(plan), "on", scale)
        required_fields = (
            NCU_TOTAL_MFLOP_FIELD,
            NCU_TENSOR_MFLOP_FIELD,
            NCU_SCALAR_MFLOP_FIELD,
            NCU_TENSOR_SHARE_FIELD,
            NCU_KERNEL_COUNT_FIELD,
            NCU_KERNEL_TIME_FIELD,
        )
        if not all(
            math.isfinite(_finite_float(profile.get(field)))
            for field in required_fields
        ):
            return False
        if str(profile.get(NCU_ERROR_FIELD) or "").strip():
            return False
    return True


def compute_plan_covers_tool(
    plan: Mapping[str, Any],
    context: ResultContext,
    tool: str,
) -> bool:
    if tool not in {"torch", "ncu"} or not isinstance(plan, Mapping):
        return False
    found_mode = False
    for mode in TOOL_GPU_MODES[tool]:
        scales = context.scales_by_mode.get(mode, [])
        if not scales:
            continue
        found_mode = True
        for scale in scales:
            profile = find_compute_profile_entry(dict(plan), mode, scale)
            if not all(
                math.isfinite(_finite_float(profile.get(field)))
                for field in COMPUTE_PLAN_METRIC_FIELDS[tool]
            ):
                return False
            if str(profile.get(TOOL_ERROR_FIELD[tool]) or "").strip():
                return False
    return found_mode


def execution_plan_covers_tool(
    plan: Mapping[str, Any],
    context: ResultContext,
    tool: str,
) -> bool:
    if tool not in {"massif", "nsys"} or not isinstance(plan, Mapping):
        return False
    mode = TOOL_GPU_MODES[tool][0]
    cases = context.cases_for_mode(mode)
    scales = context.scales_by_mode.get(mode, [])
    if not cases or not scales:
        return False
    for cpu, mem, gpu_mode in cases:
        for scale in scales:
            profile = find_execution_profile_entry(
                dict(plan), cpu, mem, gpu_mode, scale
            )
            if not all(
                math.isfinite(_finite_float(profile.get(field)))
                for field in TOOL_METRIC_FIELDS[tool]
            ):
                return False
            if str(profile.get(TOOL_ERROR_FIELD[tool]) or "").strip():
                return False
    return True


def _plan_matches_context(
    plan: Mapping[str, Any],
    context: ResultContext,
) -> bool:
    model_id = str(plan.get("model_id") or "").strip()
    if model_id and model_id != context.task_info.model_id:
        return False
    revision = str(plan.get("model_revision") or "").strip()
    expected_revision = str(context.task_info.model_revision or "main").strip()
    if revision and revision != expected_revision:
        return False
    for key, expected in (
        ("task_family", context.task_info.task_family),
        ("pipeline_tag", context.task_info.pipeline_tag),
        ("runtime_backend", context.task_info.runtime_backend),
    ):
        value = str(plan.get(key) or "").strip()
        if value and value != expected:
            return False
    return True


def _compute_plan_candidates(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> List[Path]:
    return [
        context.result_dir / "compute_profile_plan.json",
        workspace / "compute_profile_plan.json",
        workspace / tool / "compute_profile_plan.json",
    ]


def _execution_plan_candidates(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> List[Path]:
    return [
        context.result_dir / "execution_profile_plan.json",
        workspace / "execution_profile_plan.json",
        workspace / tool / "execution_profile_plan.for_backfill.json",
        workspace / tool / "execution_profile_plan.json",
    ]


def _find_reusable_compute_plan(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> Optional[Dict[str, Any]]:
    for path in _compute_plan_candidates(context, workspace, tool):
        plan = _read_plan(path)
        if (
            plan is not None
            and _plan_matches_context(plan, context)
            and compute_plan_covers_tool(plan, context, tool)
        ):
            print(f"[profile][{tool}] Reusing complete plan: {path}")
            return plan
    return None


def _find_reusable_execution_plan(
    context: ResultContext,
    workspace: Path,
    tool: str,
) -> Optional[Dict[str, Any]]:
    for path in _execution_plan_candidates(context, workspace, tool):
        plan = _read_plan(path)
        if (
            plan is not None
            and _plan_matches_context(plan, context)
            and execution_plan_covers_tool(plan, context, tool)
        ):
            print(f"[profile][{tool}] Reusing complete plan: {path}")
            return plan
    return None


def _validate_profiler_runtime(context: ResultContext) -> None:
    # Post-hoc collection deliberately skips packet, RAPL, perf, and workload
    # preflights.  It only needs the same native Docker daemon and model image.
    from acprof.host.preflight import require_native_docker, require_native_linux_host

    require_native_linux_host()
    require_native_docker()
    result = subprocess.run(
        ["docker", "image", "inspect", context.image_tag],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "image not found").strip()
        raise PosthocError(
            f"required model image is unavailable: {context.image_tag}: {detail}"
        )


def _collect_compute_plan(
    context: ResultContext,
    output_dir: Path,
    *,
    tool: str,
    ncu_root: Optional[str],
    torch_repeat: int,
    ncu_repeat: int,
    compute_profile_cpus: Optional[int],
    compute_profile_mem: Optional[int],
    resume_existing_ncu_profiles: bool = True,
) -> Dict[str, Any]:
    from acprof.host.compute_profile import collect_compute_profile_plan

    modes = [
        mode
        for mode in ("off", "on")
        if mode in TOOL_GPU_MODES[tool] and context.cases_for_mode(mode)
    ]
    cases = [case for case in context.resource_cases if case[2] in modes]
    cpus = sorted({case[0] for case in cases}) or [1]
    mems = sorted({case[1] for case in cases}) or [1]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = collect_compute_profile_plan(
        task_info=context.task_info,
        image_tag=context.image_tag,
        cpu_list=cpus,
        mem_list=mems,
        gpu_list=modes,
        output_dir=str(output_dir),
        input_scale_plan_file=str(context.input_scale_plan_path),
        advisor_root=None,
        ncu_root=ncu_root,
        advisor_repeat=1,
        torch_profiler_repeat=torch_repeat,
        ncu_repeat=ncu_repeat,
        keep_profiles=True,
        compute_profile_cpus=compute_profile_cpus,
        compute_profile_mem=compute_profile_mem,
        compute_profile_tool=tool,
        resume_existing_ncu_profiles=resume_existing_ncu_profiles,
    )
    plan = _read_plan(Path(plan_path))
    if plan is None:
        raise PosthocError(
            f"{tool} collector did not produce a valid compute plan: {plan_path}"
        )
    return plan


def _resolve_massif_reference(
    context: ResultContext,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[int, int]:
    cases = [(cpu, mem) for cpu, mem, mode in context.resource_cases if mode == "off"]
    candidates = [
        case
        for case in cases
        if (reference_cpu is None or case[0] == reference_cpu)
        and (reference_mem is None or case[1] == reference_mem)
    ]
    if not candidates:
        raise PosthocError(
            "Massif reference resource is not present in CPU result rows: "
            f"cpu={reference_cpu or 'auto'}, mem={reference_mem or 'auto'}"
        )
    return max(candidates, key=lambda case: (case[0], case[1]))


def expand_representative_massif_plan(
    plan: Mapping[str, Any],
    context: ResultContext,
    *,
    reference_cpu: int,
    reference_mem: int,
) -> Dict[str, Any]:
    expanded = copy.deepcopy(dict(plan))
    source_profile: Optional[Mapping[str, Any]] = None
    for profile in expanded.get("profiles", []):
        if not isinstance(profile, Mapping):
            continue
        if (
            int(profile.get("cpu_cores", -1)) == reference_cpu
            and int(profile.get("mem_cap_gb", -1)) == reference_mem
            and str(profile.get("gpu_mode") or "").strip().lower() == "off"
            and isinstance(profile.get("tools"), Mapping)
            and isinstance(profile["tools"].get("massif"), Mapping)
        ):
            source_profile = profile
            break
    if source_profile is None:
        raise PosthocError("representative Massif plan has no source profile")

    source_tool = copy.deepcopy(source_profile["tools"]["massif"])
    for entry in source_tool.get("entries", []):
        if isinstance(entry, dict):
            entry["profile_source_cpu_cores"] = reference_cpu
            entry["profile_source_mem_cap_gb"] = reference_mem
            entry["profile_sampling_strategy"] = "representative_per_scale"

    expanded_profiles: List[Dict[str, Any]] = []
    for cpu, mem, mode in context.resource_cases:
        if mode != "off":
            continue
        expanded_profiles.append(
            {
                "cpu_cores": cpu,
                "mem_cap_gb": mem,
                "gpu_mode": "off",
                "tools": {"massif": copy.deepcopy(source_tool)},
            }
        )
    expanded["profiles"] = expanded_profiles
    metadata = expanded.setdefault("static_metadata", {})
    if isinstance(metadata, dict):
        metadata.update(
            {
                "massif_sampling_strategy": "representative_per_scale",
                "massif_reference_cpu_cores": reference_cpu,
                "massif_reference_mem_cap_gb": reference_mem,
                "massif_reused_across_resource_cases": True,
            }
        )
    expanded["massif_sampling_strategy"] = "representative_per_scale"
    return expanded


def _collect_execution_plan(
    context: ResultContext,
    output_dir: Path,
    *,
    tool: str,
    massif_sampling: str,
    massif_reference_cpu: Optional[int],
    massif_reference_mem: Optional[int],
    nsys_sampling: str,
    nsys_reference_cpu: Optional[int],
    nsys_reference_mem: Optional[int],
    massif_repeat: int,
    nsys_repeat: int,
    nsys_root: Optional[str],
    resume_existing_profiles: bool = True,
) -> Dict[str, Any]:
    from acprof.host.execution_profile import collect_execution_profile_plan

    mode = TOOL_GPU_MODES[tool][0]
    cases = context.cases_for_mode(mode)
    cpus = sorted({case[0] for case in cases})
    mems = sorted({case[1] for case in cases})

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan_path = collect_execution_profile_plan(
            task_info=context.task_info,
            image_tag=context.image_tag,
            cpu_list=cpus,
            mem_list=mems,
            gpu_list=[mode],
            output_dir=str(output_dir),
            input_scale_plan_file=str(context.input_scale_plan_path),
            project_dir=str(PROJECT_DIR),
            tool_mode=tool,
            massif_sampling=massif_sampling,
            massif_reference_cpu=massif_reference_cpu,
            massif_reference_mem=massif_reference_mem,
            massif_repeat=massif_repeat,
            nsys_sampling=nsys_sampling,
            nsys_reference_cpu=nsys_reference_cpu,
            nsys_reference_mem=nsys_reference_mem,
            nsys_repeat=nsys_repeat,
            nsys_root=nsys_root,
            keep_profiles=True,
            resume_existing_profiles=resume_existing_profiles,
        )
    except ValueError as exc:
        raise PosthocError(str(exc)) from exc
    plan = _read_plan(Path(plan_path))
    if plan is None:
        raise PosthocError(
            f"{tool} collector did not produce a valid plan: {plan_path}"
        )
    return plan


def merge_compute_plans(
    context: ResultContext,
    plans_by_tool: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {}
    static_metadata: Dict[str, Any] = {}
    enabled_tools: List[str] = []
    plan_keys = {"torch": TORCH_PROFILE_KEY, "ncu": NCU_PROFILE_KEY}

    for requested_tool, plan in plans_by_tool.items():
        plan_key = plan_keys[requested_tool]
        metadata = plan.get("static_metadata")
        if isinstance(metadata, Mapping):
            metadata_keys = ["compute_profiles_retained"]
            if requested_tool == "torch":
                metadata_keys.extend(
                    key
                    for key in metadata
                    if str(key).startswith("torch_profiler_eager_")
                )
                metadata_keys.extend(["torch_version", "transformers_version"])
            else:
                metadata_keys.extend(
                    key for key in metadata if str(key).startswith("ncu_")
                )
            metadata_keys.extend(["gpu_compute_capability", "gpu_sm_count"])
            for key in metadata_keys:
                if key in metadata:
                    static_metadata[key] = copy.deepcopy(metadata[key])

        source_profiles = plan.get("profiles")
        if isinstance(source_profiles, Mapping):
            for profile_name in ("cpu", "gpu"):
                source_group = source_profiles.get(profile_name)
                tool_profile = (
                    source_group.get(plan_key)
                    if isinstance(source_group, Mapping)
                    else None
                )
                if isinstance(tool_profile, Mapping):
                    profiles.setdefault(profile_name, {})[plan_key] = copy.deepcopy(
                        dict(tool_profile)
                    )
        if plan_key not in enabled_tools:
            enabled_tools.append(plan_key)

    static_metadata["compute_profile_tools"] = [
        tool
        for tool in (TORCH_PROFILE_KEY, NCU_PROFILE_KEY)
        if tool in enabled_tools
    ]
    static_metadata["compute_profiles_retained"] = True
    static_metadata["compute_profile_provenance"] = "posthoc_backfill"
    return {
        "model_id": context.task_info.model_id,
        "model_revision": context.task_info.model_revision or "main",
        "task_family": context.task_info.task_family,
        "pipeline_tag": context.task_info.pipeline_tag,
        "runtime_backend": context.task_info.runtime_backend,
        "compute_profile_tool_mode": "posthoc",
        "static_metadata": static_metadata,
        "profiles": profiles,
    }


def merge_execution_plans(
    context: ResultContext,
    plans_by_tool: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    profiles: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    static_metadata: Dict[str, Any] = {}
    enabled_tools: List[str] = []
    for requested_tool, plan in plans_by_tool.items():
        metadata = plan.get("static_metadata")
        if isinstance(metadata, Mapping):
            metadata_keys = [
                "execution_profile_schema_version",
                "execution_profiles_retained",
            ]
            if requested_tool == "massif":
                metadata_keys.extend(
                    key
                    for key in metadata
                    if str(key).startswith("massif_")
                )
            elif requested_tool == "nsys":
                metadata_keys.extend(
                    key for key in metadata if str(key).startswith("nsys_")
                )
            for key in metadata_keys:
                if key in metadata:
                    static_metadata[key] = copy.deepcopy(metadata[key])
        for profile in plan.get("profiles", []):
            if not isinstance(profile, Mapping):
                continue
            try:
                key = (
                    int(profile.get("cpu_cores")),
                    int(profile.get("mem_cap_gb")),
                    str(profile.get("gpu_mode") or "").strip().lower(),
                )
            except (TypeError, ValueError):
                continue
            tools = profile.get("tools")
            tool_profile = (
                tools.get(requested_tool) if isinstance(tools, Mapping) else None
            )
            if not isinstance(tool_profile, Mapping):
                continue
            target = profiles.setdefault(
                key,
                {
                    "cpu_cores": key[0],
                    "mem_cap_gb": key[1],
                    "gpu_mode": key[2],
                    "tools": {},
                },
            )
            target["tools"][requested_tool] = copy.deepcopy(dict(tool_profile))
        if requested_tool not in enabled_tools:
            enabled_tools.append(requested_tool)

    static_metadata["execution_profile_tools"] = [
        tool for tool in ("massif", "nsys") if tool in enabled_tools
    ]
    static_metadata["execution_profile_provenance"] = "posthoc_backfill"
    return {
        "schema_version": 1,
        "model_id": context.task_info.model_id,
        "model_revision": context.task_info.model_revision or "main",
        "task_family": context.task_info.task_family,
        "pipeline_tag": context.task_info.pipeline_tag,
        "runtime_backend": context.task_info.runtime_backend,
        "execution_profile_tool_mode": "posthoc",
        "static_metadata": static_metadata,
        "profiles": [profiles[key] for key in sorted(profiles)],
    }
