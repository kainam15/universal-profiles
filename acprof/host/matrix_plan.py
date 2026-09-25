"""Frozen, reproducible resource and input-scale execution order (no collectors)."""
from __future__ import annotations

import hashlib
from itertools import product
import json
import math
from pathlib import Path

from acprof.artifacts import atomic_write_json

MATRIX_PLAN_NAME = "matrix_plan.json"
ALGORITHM_VERSION = "sha256-sort-v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def matrix_identity(task, image, cpus, mems, gpus, scales, *, order, seed,
                    prune, input_plan_file=None) -> dict:
    if order not in {"seeded", "declared"} or type(seed) is not int:
        raise ValueError("matrix order must be seeded/declared and seed must be an integer")
    if any(not values or len(set(values)) != len(values) for values in (cpus, mems, gpus, scales)):
        raise ValueError("matrix resource and scale lists must be non-empty and unique")
    if any(type(value) is not int or value <= 0 for value in [*cpus, *mems]):
        raise ValueError("matrix CPUs and memory must be positive integers")
    if any(gpu not in {"off", "on"} for gpu in gpus):
        raise ValueError("matrix GPU modes must be off/on")
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("matrix input scales must be finite and positive")
    return {"model_id": task.model_id, "model_revision": task.model_revision,
            "image_id": image.tag, "cpus": list(cpus), "mems": list(mems),
            "gpus": list(gpus), "input_scales": list(scales), "order": order,
            "seed": seed, "prune_startup_oom": prune,
            "input_plan_sha256": hashlib.sha256(Path(input_plan_file).read_bytes()).hexdigest()
            if input_plan_file else ""}


def build_matrix_plan(identity: dict, prefixes: dict) -> dict:
    """Domain-separated hash ordering avoids global RNG and Python shuffle drift."""
    cases = []
    for cpu, mem, gpu in product(identity["cpus"], identity["mems"], identity["gpus"]):
        key = [cpu, mem, gpu]
        scale_seed = digest(["input-scale-seed-v1", identity["seed"], key])
        scales = list(identity["input_scales"])
        if identity["order"] == "seeded":
            scales.sort(key=lambda scale: digest(["input-scale-order-v1", scale_seed, scale]))
        cases.append({"cpu_cores": cpu, "mem_cap_gb": mem, "gpu_mode": gpu,
                      "input_scale_seed": scale_seed, "input_scales": scales,
                      "result_origin": "inferred_not_measured" if mem in prefixes.get(gpu, [])
                      else "formal_measurement"})
    if identity["order"] == "seeded":
        cases.sort(key=lambda case: digest(["matrix-case-order-v1", identity["seed"],
                                           case["cpu_cores"], case["mem_cap_gb"], case["gpu_mode"]]))
    plan = {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION,
            "matrix_order": identity["order"], "seed": identity["seed"],
            "identity": identity, "startup_oom_prefixes": prefixes, "cases": cases}
    plan["plan_sha256"] = digest(plan)
    return plan


def load_matrix_plan(path: str | Path, identity: dict) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("unsupported matrix plan schema/algorithm; start a new experiment")
    if plan.get("identity") != identity:
        raise ValueError("matrix plan identity/options changed; use the original options")
    if plan.get("seed") != identity["seed"] or plan.get("matrix_order") != identity["order"]:
        raise ValueError("matrix plan seed/order metadata mismatch")
    if plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("matrix plan hash mismatch")
    expected = set(product(identity["cpus"], identity["mems"], identity["gpus"]))
    cases = plan["cases"]
    keys = [(c["cpu_cores"], c["mem_cap_gb"], c["gpu_mode"]) for c in cases]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("matrix plan case coverage mismatch")
    for case in cases:
        if sorted(case["input_scales"]) != sorted(identity["input_scales"]):
            raise ValueError("matrix plan input-scale coverage mismatch")
        origin = ("inferred_not_measured" if case["mem_cap_gb"] in
                  plan["startup_oom_prefixes"].get(case["gpu_mode"], []) else "formal_measurement")
        if case["result_origin"] != origin:
            raise ValueError("matrix plan pruning evidence mismatch")
    return plan


def freeze_matrix_plan(path: str | Path, identity: dict, prefixes: dict) -> dict:
    if Path(path).exists():
        return load_matrix_plan(path, identity)
    plan = build_matrix_plan(identity, prefixes)
    atomic_write_json(path, plan)
    return plan
