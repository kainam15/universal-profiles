"""Bounded AST facts for Transformers Pipeline methods; no imports or execution."""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
import json
import re
from typing import Any


MAX_SOURCE_BYTES = 256 * 1024
MAX_AST_NODES = 20000
_MISSING = object()


def parse_source(source: str, filename: str) -> ast.Module:
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError(f"{filename}: source exceeds {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ValueError(f"{filename}: cannot parse Python source: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError(f"{filename}: AST exceeds {MAX_AST_NODES} nodes")
    return tree


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return _MISSING
    try:
        value = ast.literal_eval(node)
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, TypeError, RecursionError):
        return _MISSING


def _name(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _signature(method: ast.FunctionDef) -> dict[str, dict[str, Any]]:
    args = [*method.args.posonlyargs, *method.args.args]
    defaults = [None] * (len(args) - len(method.args.defaults)) + list(method.args.defaults)
    result = {}
    for arg, default in [*zip(args[2:], defaults[2:]), *zip(method.args.kwonlyargs, method.args.kw_defaults)]:
        value = _literal(default) if default is not None else _MISSING
        result[arg.arg] = {"required": default is None, "default_known": value is not _MISSING,
                           "default": None if value is _MISSING else value,
                           "positional_only": arg in method.args.posonlyargs}
    return result


def _input_fields(method: ast.FunctionDef) -> tuple[dict[str, dict[str, Any]], list[str]]:
    args = [*method.args.posonlyargs, *method.args.args]
    if len(args) != 2 or any(item["required"] for item in _signature(method).values()):
        return {}, ["preprocess requires one input mapping"]
    parameter = args[1].arg
    fields: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    if any(_name(node, parameter) and isinstance(node.ctx, ast.Store) for node in ast.walk(method)
           if isinstance(node, ast.Name)):
        issues.append("preprocess reassigns the input mapping")
    for node in ast.walk(method):
        key, required, default = None, False, _MISSING
        if isinstance(node, ast.Subscript) and _name(node.value, parameter):
            key, required = _literal(node.slice), True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _name(node.func.value, parameter):
            if node.func.attr != "get" or not 1 <= len(node.args) <= 2 or node.keywords:
                issues.append("preprocess uses a dynamic input mapping operation")
                continue
            key = _literal(node.args[0])
            default = _literal(node.args[1]) if len(node.args) == 2 else None
        else:
            continue
        if not isinstance(key, str) or not key.isidentifier():
            issues.append("preprocess uses a dynamic or invalid input key")
            continue
        item = {"required": required, "default_known": default is not _MISSING,
                "default": None if default is _MISSING else default, "line": node.lineno}
        if key in fields:
            previous = fields[key]
            if previous["default"] != item["default"] and previous["default_known"] and item["default_known"]:
                issues.append(f"preprocess has conflicting defaults for {key}")
            item["required"] = previous["required"] or required
        fields[key] = item
    return fields, issues


def _assignments(method: ast.FunctionDef) -> dict[str, ast.expr]:
    """Only unique, straight-line assignments can supply literal aliases."""
    assignments, writes = {}, {}
    for node in ast.walk(method):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            writes[node.id] = writes.get(node.id, 0) + 1
    for statement in method.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and writes.get(target.id) == 1:
                assignments[target.id] = statement.value
    return assignments


def _sanitized_keys(method: ast.FunctionDef) -> list[str] | None:
    if any(not isinstance(statement, (ast.Assign, ast.Return)) and not (
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    ) for statement in method.body):
        return None
    assignments = _assignments(method)
    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0] not in method.body:
        return None
    result = returns[0].value
    if not isinstance(result, (ast.Tuple, ast.List)) or len(result.elts) != 3:
        return None
    forward = result.elts[1]
    if isinstance(forward, ast.Name):
        forward = assignments.get(forward.id)
    if isinstance(forward, ast.Dict) and all(isinstance(_literal(key), str) for key in forward.keys):
        return [_literal(key) for key in forward.keys]
    if not isinstance(forward, ast.DictComp) or len(forward.generators) != 1 or method.args.kwarg is None:
        return None
    generator = forward.generators[0]
    if not isinstance(generator.target, ast.Name) or len(generator.ifs) != 1 or generator.is_async:
        return None
    key, kwargs = generator.target.id, method.args.kwarg.arg
    condition = generator.ifs[0]
    if (not _name(generator.iter, kwargs) or not _name(forward.key, key)
            or not isinstance(forward.value, ast.Subscript) or not _name(forward.value.value, kwargs)
            or not _name(forward.value.slice, key) or not isinstance(condition, ast.Compare)
            or not _name(condition.left, key) or len(condition.ops) != 1 or not isinstance(condition.ops[0], ast.In)):
        return None
    allowed = condition.comparators[0]
    if isinstance(allowed, ast.Name):
        allowed = assignments.get(allowed.id)
    keys = _literal(allowed)
    return keys if isinstance(keys, list) and all(isinstance(key, str) for key in keys) else None


def _forward_kwargs(method: ast.FunctionDef, signature: dict) -> dict | None:
    """Recognize direct generation and one explicit temperature-to-sampling idiom.

    This is a syntax recognizer, not a Python interpreter. All writes affecting
    the selected parameters must match the recognized straight-line pattern.
    """
    calls: list[ast.Call] = []
    writes: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
            calls.append(node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            writes.add(node.id)
    if len(calls) != 1:
        return None
    call = calls[0]
    parameters = [*method.args.posonlyargs, *method.args.args]
    if len(parameters) < 2 or any(kw.arg is None and not _name(kw.value, parameters[1].arg) for kw in call.keywords):
        return None
    # A conditional/nested generation call cannot prove the executed path.
    if not any((isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Return)) and stmt.value is call)
               for stmt in method.body):
        return None
    keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
    if not _name(keywords.get("max_new_tokens"), "max_new_tokens") or "max_new_tokens" not in signature:
        return None
    if "max_new_tokens" in writes:
        return None
    assignments = _assignments(method)
    sample = keywords.get("do_sample")
    kwargs: dict[str, Any] = {"max_new_tokens": "$max_new_tokens"}
    if _name(sample, "do_sample") and "do_sample" in signature and "do_sample" not in writes:
        kwargs["do_sample"] = False
    elif "temperature" in signature and _name(sample, "do_sample"):
        normalized = assignments.get("temperature")
        condition = assignments.get("do_sample")
        if (not isinstance(normalized, ast.BoolOp) or not isinstance(normalized.op, ast.Or)
                or len(normalized.values) != 2 or not _name(normalized.values[0], "temperature")
                or _literal(normalized.values[1]) is not None
                or not isinstance(condition, ast.Compare) or not _name(condition.left, "temperature")
                or len(condition.ops) != 1 or not isinstance(condition.ops[0], ast.IsNot)
                or _literal(condition.comparators[0]) is not None
                or not normalized.lineno < condition.lineno < call.lineno
                or not _name(keywords.get("temperature"), "temperature")):
            return None
        kwargs["temperature"] = 0
    else:
        return None
    if any(item["positional_only"] or (item["required"] and name not in kwargs)
           for name, item in signature.items()):
        return None
    return kwargs


@lru_cache(maxsize=64)
def _analyze_pipeline(source: str, filename: str, class_name: str) -> dict:
    tree = parse_source(source, filename)
    classes: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            classes.append(node)
    if len(classes) != 1 or classes[0].decorator_list:
        raise ValueError(f"{filename}: pipeline class {class_name} is absent, duplicated or decorated")
    methods = {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    required = {"preprocess", "_forward", "postprocess", "_sanitize_parameters"}
    if not required <= methods.keys() or any(methods[name].decorator_list for name in required):
        raise ValueError(f"{filename}: pipeline methods must be defined locally without decorators")
    fields, issues = _input_fields(methods["preprocess"])
    for method_name in required:
        for node in ast.walk(methods[method_name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "getattr"}:
                issues.append(f"{method_name} uses unsupported dynamic operation {node.func.id}")
    signature = _signature(methods["_forward"])
    sanitized = _sanitized_keys(methods["_sanitize_parameters"])
    kwargs = _forward_kwargs(methods["_forward"], signature)
    if sanitized is None:
        issues.append("_sanitize_parameters forward keys are unresolved")
    if kwargs is None:
        issues.append("_forward bounded deterministic generation is unresolved")
    elif sanitized is not None and not set(kwargs) <= set(sanitized):
        issues.append("_sanitize_parameters conflicts with _forward generation parameters")
        kwargs = None
    registrations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register_pipeline":
            if node.args and any(kw.arg == "pipeline_class" and _name(kw.value, class_name) for kw in node.keywords):
                name = _literal(node.args[0])
                if isinstance(name, str):
                    registrations.append(name)
    return {"inputs": fields, "signature": signature, "sanitized_keys": sanitized,
            "forward_kwargs": kwargs, "issues": issues, "registrations": registrations,
            "methods": {name: methods[name].lineno for name in sorted(required)}}


def analyze_pipeline(source: str, filename: str, class_name: str) -> dict:
    # Callers cannot corrupt the process-local content-addressed analysis cache.
    return copy.deepcopy(_analyze_pipeline(source, filename, class_name))


def _config_value(expression: ast.AST | None, config: dict, model_id: str) -> Any:
    literal = _literal(expression)
    if literal is not _MISSING:
        return literal
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or):
        for operand in expression.values:
            value = _config_value(operand, config, model_id)
            if value is _MISSING or value:
                return value
        return None
    path = []
    while isinstance(expression, ast.Attribute):
        path.insert(0, expression.attr)
        expression = expression.value
    if _name(expression, "config"):
        path.insert(0, "config")
    if "config" in path:
        path = path[path.index("config") + 1:]
        if path == ["_name_or_path"]:
            return model_id
        value = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return _MISSING
            value = value[key]
        return value
    return _MISSING


def dependency_candidates(source: str, filename: str, config: dict, model_id: str) -> list[dict]:
    result = []
    tree = parse_source(source, filename)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "from_pretrained":
            continue
        loader = ast.unparse(node.func.value).rsplit(".", 1)[-1]
        role = ("tokenizer" if loader == "AutoTokenizer" else "processor" if loader in {"AutoProcessor", "AutoFeatureExtractor"}
                else "metadata" if loader == "AutoConfig" else "generation_metadata" if loader == "GenerationConfig"
                else "weights" if loader.startswith("AutoModel") else "unknown")
        expression = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "pretrained_model_name_or_path"), None)
        repo = _config_value(expression, config, model_id)
        if repo == model_id:
            continue
        valid = isinstance(repo, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", repo, re.ASCII)
        revision_node = next((kw.value for kw in node.keywords if kw.arg == "revision"), None)
        requested = _literal(revision_node) if revision_node is not None else "main"
        if any(kw.arg is None for kw in node.keywords):
            requested = None
        ancestor, conditional = node, False
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Try,
                                     getattr(ast, "TryStar", ast.Try), ast.Match, ast.ListComp,
                                     ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension)):
                conditional = True
        result.append({"repo_id": repo if valid else None, "role": role, "loader": loader,
                       "required": "candidate", "state": "derived" if valid else "unresolved",
                       "source": f"{filename}:{node.lineno}", "expression": ast.unparse(expression) if expression else "",
                       "requested_revision": requested if isinstance(requested, str) else None,
                       "conditional": conditional})
    return result


def model_card_examples(text: str) -> list[dict]:
    """Supplementary literal call keys; examples never establish executable types."""
    result = []
    for block in re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL):
        try:
            tree = parse_source(block.group(1), "README.md")
        except ValueError:
            continue
        pipelines = set()
        for statement in tree.body:
            if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                    and ast.unparse(statement.value.func).rsplit(".", 1)[-1] == "pipeline"):
                pipelines.update(target.id for target in statement.targets if isinstance(target, ast.Name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in pipelines:
                payload = node.args[0] if node.args else None
                if isinstance(payload, ast.Dict):
                    keys = [_literal(key) for key in payload.keys]
                    if all(isinstance(key, str) for key in keys):
                        result.append({"inputs": keys, "kwargs": [kw.arg for kw in node.keywords if kw.arg],
                                       "line": text[:block.start(1)].count("\n") + node.lineno})
    return result
