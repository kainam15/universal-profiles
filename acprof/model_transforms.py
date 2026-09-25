"""Bounded JSON input templates. No imports, evaluation, paths or interpolation."""
from __future__ import annotations

import copy
import math
from typing import Any


MAX_DEPTH = 8
MAX_NODES = 256


def input_references(inputs: Any, allowed: set[str]) -> set[str]:
    """Validate the entire adapter with one shared complexity budget."""
    references: set[str] = set()
    remaining = MAX_NODES

    def visit(value, depth=0, *, expression=False, literal=False):
        nonlocal remaining
        remaining -= 1
        if depth > MAX_DEPTH or remaining < 0:
            raise ValueError("input transform exceeds depth/node limits")
        if isinstance(value, dict):
            if len(value) > 32 or any(not isinstance(key, str) or len(key) > 256 for key in value):
                raise ValueError("input transform object exceeds limits")
            if not literal and (expression or set(value) & {"from", "literal", "template"}):
                if len(value) != 1:
                    raise ValueError("input transform must use exactly one operation")
                operation, argument = next(iter(value.items()))
                if operation == "from":
                    if not isinstance(argument, str) or argument not in allowed:
                        raise ValueError(f"unknown input reference: {argument!r}")
                    references.add(argument)
                elif operation in {"literal", "template"}:
                    visit(argument, depth + 1, literal=operation == "literal")
                else:
                    raise ValueError(f"unknown input transform: {operation}")
            else:
                for item in value.values():
                    visit(item, depth + 1, literal=literal)
        elif expression:
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"unknown input reference: {value!r}")
            references.add(value)
        elif isinstance(value, list):
            if len(value) > 32:
                raise ValueError("input transform list exceeds 32 items")
            for item in value:
                visit(item, depth + 1, literal=literal)
        elif value is None or type(value) in {bool, int, float, str}:
            if isinstance(value, str) and len(value) > 4096 or isinstance(value, float) and not math.isfinite(value):
                raise ValueError("input transform literal exceeds limits or is nonfinite")
        else:
            raise ValueError("input transform requires JSON values")

    if (not isinstance(inputs, dict) or not inputs or len(inputs) > 32
            or any(not isinstance(key, str) or not key.isidentifier() for key in inputs)):
        raise ValueError("multimodal inputs must be a bounded target mapping")
    for value in inputs.values():
        visit(value, expression=True)
    if references != allowed:
        raise ValueError(f"multimodal inputs must map all of {sorted(allowed)}; missing {sorted(allowed - references)}")
    return references


def transform_inputs(inputs: dict, values: dict) -> dict:
    input_references(inputs, set(values))

    def render(value):
        if isinstance(value, dict):
            if set(value) == {"from"}:
                return values[value["from"]]
            if set(value) == {"literal"}:
                return copy.deepcopy(value["literal"])
            if set(value) == {"template"}:
                return render(value["template"])
            return {key: render(item) for key, item in value.items()}
        if isinstance(value, list):
            return [render(item) for item in value]
        return value

    return {key: values[value] if isinstance(value, str) else render(value) for key, value in inputs.items()}
