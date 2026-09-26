"""Bounded abstract interpretation of local loader calls, never repository execution.

Only JSON values, explicit local bindings and selected Transformers entrypoints
are understood. Unknown Python stays unknown; inactive calls remain in evidence.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
import re

from acprof.model_source_analysis import parse_source


class Unknown(Enum):
    VALUE = "unknown"


UNKNOWN = Unknown.VALUE
MAX_CALL_DEPTH = 12
MAX_CALL_CONTEXTS = 256
MAX_CANDIDATES = 512
MAX_ANALYSIS_STEPS = 100000


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass
class Instance:
    name: str
    attributes: dict = field(default_factory=dict)


def activation(parent: str, condition) -> str:
    if parent == "inactive" or condition is False:
        return "inactive"
    return parent if condition is True else "unknown"


def known_truth(value):
    return UNKNOWN if value is UNKNOWN or isinstance(value, Symbol) else bool(value)


def merge(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: merge(left.get(key, UNKNOWN), right.get(key, UNKNOWN)) for key in left.keys() | right.keys()}
    if isinstance(left, Instance) and isinstance(right, Instance) and left.name == right.name:
        return Instance(left.name, merge(left.attributes, right.attributes))
    return left if type(left) is type(right) and left == right else UNKNOWN


class DependencyAnalyzer:
    def __init__(self, sources, config, model_id, *, files=(), metadata=None, transformers_version=None):
        self.config = {**copy.deepcopy(config), "_name_or_path": model_id}
        self.model_id = model_id
        self.files = set(files)
        self.metadata = metadata or {}
        self.version = transformers_version
        self.trees = {name: parse_source(source, name) for name, source in sources.items()}
        self.functions = {}
        self.classes = {}
        self.modules = {}
        self.visited = set()
        self.stack = []
        self.candidates = []
        self.contexts = 0
        self.steps = 0
        self.transparent_calls = set()
        self.entries = {}
        for name, tree in self.trees.items():
            module = self.module_name(name)
            env = {"config": copy.deepcopy(self.config)}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = module + "." + node.name
                    self.functions[key] = (name, None, node)
                    env[node.name] = Symbol(key)
                elif isinstance(node, ast.ClassDef):
                    key = module + "." + node.name
                    self.classes[key] = (name, node)
                    env[node.name] = Symbol(key)
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.functions[key + "." + method.name] = (name, key, method)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.import_names(name, node, env)
            self.modules[name] = env
        for key, reference in config.get("auto_map", {}).items():
            if isinstance(reference, str) and reference in self.classes:
                self.entries[reference] = "config" if key == "AutoConfig" else "processor" if key == "AutoProcessor" else "model" if key.startswith("AutoModel") else "other"
        for pipeline in config.get("custom_pipelines", {}).values():
            if isinstance(pipeline, dict) and pipeline.get("impl") in self.classes:
                self.entries[pipeline["impl"]] = "pipeline"

    @staticmethod
    def module_name(filename):
        return filename.removesuffix(".py").replace("/", ".").removesuffix(".__init__")

    def import_names(self, filename, node, env):
        if isinstance(node, ast.Import):
            for alias in node.names:
                env[alias.asname or alias.name.split(".")[0]] = Symbol(alias.name if alias.asname else alias.name.split(".")[0])
            return
        prefix = node.module or ""
        if node.level:
            parent = PurePosixPath(filename).parent
            for _ in range(node.level - 1):
                parent = parent.parent
            prefix = ".".join([*(() if str(parent) == "." else parent.parts), *prefix.split(".")]).strip(".")
        for alias in node.names:
            env[alias.asname or alias.name] = Symbol((prefix + "." + alias.name).strip("."))

    def run(self):
        for filename, tree in self.trees.items():
            self.block(tree.body, self.modules[filename], filename, "active", [], None)
        for key, kind in self.entries.items():
            source_file, class_node = self.classes[key]
            if class_node.decorator_list:
                self.unresolved(source_file, class_node, "unknown", [], None, "declared loader class is decorated")
            obj = Instance(key, {"config": copy.deepcopy(self.config)})
            if kind == "config":
                self.invoke(key + ".__init__", [obj], copy.deepcopy(self.config), "active", [], None)
            elif kind == "pipeline":
                main = next((name for name, role in self.entries.items() if role == "model"), "main_model")
                model = Instance(main, {"config": copy.deepcopy(self.config)})
                self.invoke(key + ".__init__", [obj], {"model": model}, "active", [], None)
                for method in ("preprocess", "_forward", "postprocess", "_sanitize_parameters"):
                    self.invoke(key + "." + method, [obj], {}, "active", [], None)
            elif kind in {"model", "processor"}:
                self.invoke(key + ".from_pretrained", [Symbol(key), self.model_id], {}, "active", [], None)
                if kind == "model":
                    # Pinned Transformers 4.57.6 has two loader phases: construction
                    # under no_init_weights, then its registered initialization hook.
                    # These are API summaries, never arbitrary training-mode guesses.
                    supported = self.version == "4.57.6" and self.hf_model_class(key)
                    self.invoke(key + ".__init__", [obj, copy.deepcopy(self.config)], {}, "active", [], None,
                                init_weights=False if supported else UNKNOWN)
                    obj.attributes["_is_hf_initialized"] = False
                    if key + ".initialize_weights" in self.functions:
                        hook, hook_args = ".initialize_weights", [obj]
                    elif key + "._initialize_weights" in self.functions:
                        hook, hook_args = "._initialize_weights", [obj, obj]
                    else:
                        hook, hook_args = "._init_weights", [obj, obj]
                    self.invoke(key + hook, hook_args, {}, "active" if supported else "unknown", [], None,
                                init_weights=True if supported else UNKNOWN)
        # Unbound loader methods are not silently discarded. They remain review
        # candidates, with unknown arguments and reachability.
        for key, (_, owner, node) in self.functions.items():
            if key not in self.visited and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "from_pretrained" for n in ast.walk(node)):
                args = [Instance(owner)] if owner else []
                self.invoke(key, args, {}, "unknown", [], None, use_defaults=False)
        return self.candidates

    def hf_model_class(self, key):
        filename, node = self.classes[key]
        if node.decorator_list:
            return False
        for base in node.bases:
            value = self.value(base, self.modules[filename], filename, "inactive", [], None)
            if isinstance(value, Symbol) and value.name.startswith("transformers.") and value.name.endswith("PreTrainedModel"):
                return True
        return False

    @staticmethod
    def transparent_wrapper(node):
        if node.decorator_list or isinstance(node, ast.AsyncFunctionDef):
            return False
        body = [item for item in node.body if not (isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            call = body[0].value
        elif (len(body) == 2 and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1
              and isinstance(body[0].targets[0], ast.Name) and isinstance(body[1], ast.Return)
              and isinstance(body[1].value, ast.Name) and body[0].targets[0].id == body[1].value.id):
            call = body[0].value
        else:
            return False
        return isinstance(call, ast.Call) and not any(isinstance(part, ast.Call)
                   for arg in [*call.args, *(kw.value for kw in call.keywords)] for part in ast.walk(arg))

    def invoke(self, key, args, kwargs, state, chain, alternative, *, init_weights=UNKNOWN, use_defaults=True):
        if key not in self.functions:
            return UNKNOWN
        filename, owner, node = self.functions[key]
        location = f"{filename}:{node.lineno}"
        if key in self.stack or len(self.stack) >= MAX_CALL_DEPTH or self.contexts >= MAX_CALL_CONTEXTS:
            self.unresolved(filename, node, state, chain, alternative, "local call recursion/depth/context limit")
            return UNKNOWN
        self.visited.add(key)
        self.contexts += 1
        if (owner and self.classes[owner][1].decorator_list) or isinstance(node, ast.AsyncFunctionDef) or any(
                not isinstance(item, ast.Name) or item.id not in {"classmethod", "staticmethod"}
                for item in node.decorator_list):
            state = activation(state, UNKNOWN)
        env = dict(self.modules[filename])
        parameters = [*node.args.posonlyargs, *node.args.args]
        defaults = [None] * (len(parameters) - len(node.args.defaults)) + list(node.args.defaults)
        for index, (parameter, default) in enumerate(zip(parameters, defaults)):
            env[parameter.arg] = (args[index] if index < len(args) else kwargs.get(parameter.arg, UNKNOWN)
                if parameter.arg in kwargs else self.value(default, env, filename, "inactive", chain, alternative)
                if use_defaults and default is not None and "@unknown_kwargs" not in kwargs else UNKNOWN)
        for parameter, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            env[parameter.arg] = kwargs.get(parameter.arg, self.value(default, env, filename, "inactive", chain, alternative)
                                                   if use_defaults and default is not None and "@unknown_kwargs" not in kwargs else UNKNOWN)
        if node.args.vararg:
            env[node.args.vararg.arg] = args[len(parameters):]
        if node.args.kwarg:
            consumed = {item.arg for item in [*parameters, *node.args.kwonlyargs]}
            env[node.args.kwarg.arg] = {name: value for name, value in kwargs.items() if name not in consumed}
        env["@owner"] = owner
        env["@function"] = node.name
        env["@init_weights"] = init_weights
        bound_mappings = {parameter.arg: env[parameter.arg] for parameter in parameters
                          if isinstance(env[parameter.arg], dict)}
        self.stack.append(key)
        try:
            return self.block(node.body, env, filename, state, [*chain, location], alternative)[0]
        finally:
            for parameter, original in bound_mappings.items():
                updated = env[parameter]
                if isinstance(updated, dict) and updated is not original:
                    # Branch copies may have mutated a caller's mapping. Do not
                    # leave a stale caller-side config proof after a local call.
                    combined = merge(original, updated)
                    original.clear()
                    original.update(combined)
            self.stack.pop()

    def assign(self, target, value, env, filename, state, chain, alternative):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Attribute):
            parent = self.value(target.value, env, filename, "inactive", chain, alternative)
            if isinstance(parent, Instance):
                parent.attributes[target.attr] = UNKNOWN if value is parent else value
            elif isinstance(parent, dict):
                parent[target.attr] = UNKNOWN if value is parent else value
        elif isinstance(target, ast.Subscript):
            parent = self.value(target.value, env, filename, "inactive", chain, alternative)
            key = self.value(target.slice, env, filename, "inactive", chain, alternative)
            if isinstance(parent, dict):
                if isinstance(key, str):
                    parent[key] = UNKNOWN if value is parent else value
                else:
                    parent.update({name: UNKNOWN for name in parent})
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = value if isinstance(value, (tuple, list)) and len(value) == len(target.elts) else [UNKNOWN] * len(target.elts)
            for part, element in zip(target.elts, values):
                self.assign(part, element, env, filename, state, chain, alternative)

    def block(self, statements, env, filename, state, chain, alternative):
        result, returned = UNKNOWN, False
        for node in statements:
            current = "inactive" if returned else state
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Global, ast.Pass)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.value(node.value, env, filename, current, chain, alternative)
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.assign(target, value, env, filename, current, chain, alternative)
            elif isinstance(node, ast.Expr):
                self.value(node.value, env, filename, current, chain, alternative)
            elif isinstance(node, ast.Return):
                result = self.value(node.value, env, filename, current, chain, alternative)
                returned = True
            elif isinstance(node, ast.Raise):
                returned = True
            elif isinstance(node, ast.If):
                condition = self.condition(node.test, env, filename, current, chain, alternative)
                left, right = copy.deepcopy(env), copy.deepcopy(env)
                lv, lr = self.block(node.body, left, filename, activation(current, condition), chain, alternative)
                rv, rr = self.block(node.orelse, right, filename, activation(current, not condition if condition is not UNKNOWN else UNKNOWN), chain, alternative)
                selected = left if condition is True else right if condition is False else merge(left, right)
                env.update(selected)
                if condition is True and lr or condition is False and rr or lr and rr:
                    result = lv if condition is True else rv if condition is False else merge(lv, rv)
                    returned = True
                elif condition is UNKNOWN and (lr or rr):
                    state = activation(current, UNKNOWN)
            elif isinstance(node, ast.Try):
                group = f"{filename}:{node.lineno}"
                before = len(self.candidates)
                primary_env = copy.deepcopy(env)
                self.block(node.body, primary_env, filename, current, chain, {"group": group, "branch": "primary"})
                primary = self.candidates[before:]
                complete = self.complete_primary(node.body, primary, filename)
                others = []
                for handler in node.handlers:
                    branch = copy.deepcopy(env)
                    self.block(handler.body, branch, filename, activation(current, False if complete else UNKNOWN), chain,
                               {"group": group, "branch": "fallback", "primary_sources": [item["source"] for item in primary],
                                "reason": "main snapshot tokenizer files complete" if complete else "primary success unproven"})
                    others.append(branch)
                env.update(primary_env if complete else merge(env, primary_env))
                if not complete:
                    for branch in others:
                        env.update(merge(env, branch))
                self.block(node.orelse, env, filename, activation(current, True if complete else UNKNOWN), chain, alternative)
                self.block(node.finalbody, env, filename, current, chain, alternative)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.block(node.body, env, filename, activation(current, UNKNOWN), chain, alternative)
            else:
                # Loops, comprehensions, match, decorators and unsupported statements
                # never turn potential downloads into active dependencies.
                branch = copy.deepcopy(env)
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr):
                        self.value(child, branch, filename, activation(current, UNKNOWN), chain, alternative)
                    elif isinstance(child, ast.stmt):
                        self.block([child], branch, filename, activation(current, UNKNOWN), chain, alternative)
                    elif isinstance(child, ast.match_case):
                        self.block(child.body, branch, filename, activation(current, UNKNOWN), chain, alternative)
                env.update(merge(env, branch))
        return result, returned

    def condition(self, node, env, filename, state, chain, alternative):
        if isinstance(node, ast.BoolOp):
            values = [self.condition(part, env, filename, activation(state, UNKNOWN) if i else state, chain, alternative)
                      for i, part in enumerate(node.values)]
            decisive = False if isinstance(node.op, ast.And) else True
            if any(value is decisive for value in values):
                return decisive
            return UNKNOWN if any(value is UNKNOWN for value in values) else not decisive
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.condition(node.operand, env, filename, state, chain, alternative)
            return UNKNOWN if value is UNKNOWN else not value
        return known_truth(self.value(node, env, filename, state, chain, alternative))

    def value(self, node, env, filename, state, chain, alternative):
        self.steps += 1
        if self.steps > MAX_ANALYSIS_STEPS:
            raise ValueError("dependency analysis step limit exceeded")
        if node is None:
            return UNKNOWN
        def evaluate(child):
            return self.value(child, env, filename, state, chain, alternative)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, Symbol(node.id))
        if isinstance(node, ast.Attribute):
            base = evaluate(node.value)
            if isinstance(base, Instance):
                return base.attributes.get(node.attr, Symbol(base.name + "." + node.attr))
            if isinstance(base, dict):
                return base.get(node.attr, UNKNOWN)
            if isinstance(base, Symbol):
                name = base.name + "." + node.attr
                if name == "transformers.modeling_utils._init_weights":
                    return env.get("@init_weights", UNKNOWN)
                return Symbol(name)
            return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [evaluate(part) for part in node.elts]
        if isinstance(node, ast.Dict):
            result = {}
            for key, val in zip(node.keys, node.values):
                value = evaluate(val)
                if key is None:
                    if isinstance(value, dict):
                        result.update(value)
                    else:
                        result["@unknown_kwargs"] = UNKNOWN
                else:
                    name = evaluate(key)
                    if isinstance(name, str):
                        result[name] = value
                    else:
                        result["@unknown_kwargs"] = UNKNOWN
            return result
        if isinstance(node, ast.Subscript):
            base, key = evaluate(node.value), evaluate(node.slice)
            if isinstance(base, dict) and isinstance(key, (str, int)):
                return base.get(key, UNKNOWN)
            if isinstance(base, (list, tuple)) and isinstance(key, int) and -len(base) <= key < len(base):
                return base[key]
            return UNKNOWN
        if isinstance(node, ast.BoolOp):
            uncertain = False
            stopped = False
            result = UNKNOWN
            for part in node.values:
                value = self.value(part, env, filename, "inactive" if stopped else activation(state, UNKNOWN)
                                   if uncertain else state, chain, alternative)
                if stopped:
                    continue
                truth = known_truth(value)
                if truth is UNKNOWN:
                    uncertain = True
                if truth == isinstance(node.op, ast.Or):
                    stopped = True
                result = UNKNOWN if uncertain else value
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = known_truth(evaluate(node.operand))
            return UNKNOWN if value is UNKNOWN else not value
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            outcomes = []
            for operator, comparator in zip(node.ops, node.comparators):
                right = evaluate(comparator)
                if left is UNKNOWN or right is UNKNOWN or isinstance(left, Symbol) or isinstance(right, Symbol):
                    outcomes.append(UNKNOWN)
                elif isinstance(operator, (ast.Is, ast.IsNot)):
                    if left is None or right is None or isinstance(left, (bool, Instance)) or isinstance(right, (bool, Instance)):
                        equal = left is right
                        outcomes.append(not equal if isinstance(operator, ast.IsNot) else equal)
                    else:
                        outcomes.append(UNKNOWN)
                elif isinstance(operator, (ast.Eq, ast.NotEq)) and all(
                        value is None or type(value) in (str, int, float, bool) for value in (left, right)):
                    equal = left == right
                    outcomes.append(not equal if isinstance(operator, ast.NotEq) else equal)
                elif isinstance(operator, (ast.In, ast.NotIn)) and isinstance(right, (dict, list, str)):
                    try:
                        present = left in right
                        outcomes.append(not present if isinstance(operator, ast.NotIn) else present)
                    except TypeError:
                        outcomes.append(UNKNOWN)
                else:
                    outcomes.append(UNKNOWN)
                left = right
            return False if False in outcomes else UNKNOWN if UNKNOWN in outcomes else True
        if isinstance(node, ast.IfExp):
            condition = self.condition(node.test, env, filename, state, chain, alternative)
            left = self.value(node.body, env, filename, activation(state, condition), chain, alternative)
            right = self.value(node.orelse, env, filename, activation(state, not condition if condition is not UNKNOWN else UNKNOWN), chain, alternative)
            return left if condition is True else right if condition is False else merge(left, right)
        if isinstance(node, ast.Call):
            return self.call(node, env, filename, state, chain, alternative)
        if isinstance(node, ast.DictComp) and len(node.generators) == 1:
            generator = node.generators[0]
            # A finite literal key filter proves that forwarded kwargs cannot
            # override revision, even when their non-identity values are unknown.
            if (isinstance(node.key, ast.Name) and isinstance(node.value, ast.Name)
                    and isinstance(generator.target, ast.Tuple) and len(generator.target.elts) == 2
                    and all(isinstance(part, ast.Name) for part in generator.target.elts)
                    and [part.id for part in generator.target.elts] == [node.key.id, node.value.id]
                    and isinstance(generator.iter, ast.Call) and isinstance(generator.iter.func, ast.Attribute)
                    and generator.iter.func.attr == "items" and not generator.iter.args and not generator.iter.keywords
                    and not generator.is_async and len(generator.ifs) == 1):
                check = generator.ifs[0]
                if (isinstance(check, ast.Compare) and isinstance(check.left, ast.Name)
                        and check.left.id == node.key.id and len(check.ops) == 1 and isinstance(check.ops[0], ast.In)):
                    allowed = evaluate(check.comparators[0])
                    if isinstance(allowed, list) and all(isinstance(key, str) for key in allowed):
                        return {key: UNKNOWN for key in allowed}
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.value(child, env, filename, activation(state, UNKNOWN), chain, alternative)
        return UNKNOWN

    def arguments(self, node, env, filename, state, chain, alternative):
        args, kwargs = [], {}
        for arg in node.args:
            value = self.value(arg.value if isinstance(arg, ast.Starred) else arg, env, filename, state, chain, alternative)
            if isinstance(arg, ast.Starred):
                args.extend(value if isinstance(value, list) else [UNKNOWN])
            else:
                args.append(value)
        for keyword in node.keywords:
            value = self.value(keyword.value, env, filename, state, chain, alternative)
            if keyword.arg is None:
                if isinstance(value, dict):
                    kwargs.update(value)
                else:
                    kwargs["@unknown_kwargs"] = UNKNOWN
            else:
                kwargs[keyword.arg] = value
        return args, kwargs

    def call(self, node, env, filename, state, chain, alternative):
        args, kwargs = self.arguments(node, env, filename, state, chain, alternative)
        function = self.value(node.func, env, filename, "inactive", chain, alternative)
        name = function.name if isinstance(function, Symbol) else ""
        if isinstance(node.func, ast.Name) and node.func.id == "hasattr" and len(args) == 2:
            if args == [Symbol("transformers.modeling_utils"), "_init_weights"] and self.version == "4.57.6":
                return True
            return UNKNOWN
        if name == "getattr" and len(args) in (2, 3) and isinstance(args[0], Instance) and isinstance(args[1], str):
            return args[0].attributes.get(args[1], args[2] if len(args) == 3 else UNKNOWN)
        if isinstance(node.func, ast.Attribute):
            base = self.value(node.func.value, env, filename, "inactive", chain, alternative)
            if isinstance(base, dict):
                if node.func.attr == "get" and args and isinstance(args[0], str):
                    default = UNKNOWN if "@unknown_kwargs" in base else args[1] if len(args) > 1 else None
                    return base.get(args[0], default)
                if node.func.attr in {"items", "keys", "values", "copy"}:
                    return UNKNOWN
                if node.func.attr == "update" and all(isinstance(value, dict) for value in args):
                    for value in args:
                        base.update(value)
                    base.update(kwargs)
                    return None
                base.update({key: UNKNOWN for key in base})
                base["@unknown_kwargs"] = UNKNOWN
                return UNKNOWN
        if name in self.classes:
            obj = Instance(name, {"config": args[0] if args and isinstance(args[0], dict) else UNKNOWN})
            self.invoke(name + ".__init__", [obj, *args], kwargs, state, [*chain, f"{filename}:{node.lineno}"], alternative,
                        init_weights=env.get("@init_weights", UNKNOWN))
            return obj
        if name in self.functions:
            _, owner, method = self.functions[name]
            if self.transparent_wrapper(method):
                self.transparent_calls.add(f"{filename}:{node.lineno}")
            if owner:
                base = self.value(node.func.value, env, filename, "inactive", chain, alternative)
                if not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in method.decorator_list):
                    args = [base, *args]
            return self.invoke(name, args, kwargs, state, [*chain, f"{filename}:{node.lineno}"], alternative,
                               init_weights=env.get("@init_weights", UNKNOWN))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            loader = name.rsplit(".", 2)[-2] if name else ast.unparse(node.func.value)
            owner = env.get("@owner")
            parent_forward = (isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name)
                              and node.func.value.func.id == "super")
            main_forward = parent_forward and self.entries.get(owner) == "model" and env.get("@function") == "from_pretrained" and self.hf_model_class(owner)
            role = ("weights" if main_forward or loader.startswith("AutoModel") else "tokenizer" if loader == "AutoTokenizer"
                    else "processor" if loader in {"AutoProcessor", "AutoFeatureExtractor"} else "metadata" if loader == "AutoConfig"
                    else "generation_metadata" if loader == "GenerationConfig" else "unknown")
            if not main_forward and (not isinstance(function, Symbol) or not (
                    name.startswith("transformers.") or name == loader + ".from_pretrained")):
                role = "unknown"
            repo = args[0] if args else kwargs.get("pretrained_model_name_or_path", UNKNOWN)
            revision = UNKNOWN if "@unknown_kwargs" in kwargs else kwargs.get("revision", "main")
            kind = "main_model" if (repo == self.model_id and role != "unknown" and revision == "main"
                                     and (not parent_forward or main_forward)) else "external"
            valid = isinstance(repo, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", repo, re.ASCII)
            expression = ast.unparse(node.args[0]) if node.args else ast.unparse(node)
            item = {"repo_id": repo if valid else None, "role": role, "loader": "super()" if parent_forward else loader,
                    "required": "candidate", "state": "derived" if valid else "unresolved", "activation": state,
                    "dependency_kind": kind, "source": f"{filename}:{node.lineno}", "expression": expression,
                    "requested_revision": revision if isinstance(revision, str) else None,
                    "call_chain": [*chain, f"{filename}:{node.lineno}"]}
            if env.get("@init_weights", UNKNOWN) is not UNKNOWN:
                item["loader_context"] = {"transformers_version": self.version,
                                          "phase": "weight_initialization" if env["@init_weights"] else "construction"}
            if alternative:
                item["alternative"] = alternative
            if kind == "main_model" and role == "tokenizer":
                safe_options = all(key in {"local_files_only", "trust_remote_code"} and type(value) is bool
                                   for key, value in kwargs.items())
                item["file_availability"] = ("complete" if len(args) == 1 and safe_options
                    and self.main_tokenizer_files_complete() else "unknown")
            self.append(item)
            if role == "metadata":
                return copy.deepcopy(self.config) if kind == "main_model" else {"_name_or_path": repo}
            return Instance(owner or loader, {"config": copy.deepcopy(self.config)})
        # No arbitrary function is evaluated, including eval/getattr/dynamic repo
        # functions. Local code reached through an unknown dispatch stays unbound.
        return UNKNOWN

    def main_tokenizer_files_complete(self):
        metadata = self.metadata.get("tokenizer_config.json", {})
        return ({"tokenizer.json", "tokenizer_config.json"} <= self.files and not metadata.get("auto_map")
                and metadata.get("tokenizer_class") in {"PreTrainedTokenizer", "PreTrainedTokenizerFast"})

    def complete_primary(self, statements, candidates, filename):
        if len(statements) != 1 or not isinstance(statements[0], (ast.Assign, ast.AnnAssign, ast.Expr)):
            return False
        node = statements[0].value
        if not isinstance(node, ast.Call):
            return False
        if len(candidates) != 1:
            return False
        item = candidates[0]
        location = f"{filename}:{node.lineno}"
        if item["source"] != location and location not in self.transparent_calls:
            return False
        # Every wrapper below this try's call site must consist solely of
        # forwarding and returning the loader result, without extra operations.
        path = item["call_chain"]
        if location not in path:
            return False
        tail = path[path.index(location) + 1:]
        definitions = {f"{name}:{method.lineno}": method for name, _, method in self.functions.values()}
        if any(not self.transparent_wrapper(definitions[source]) for source in tail if source in definitions):
            return False
        return (item["dependency_kind"] == "main_model" and item["role"] == "tokenizer"
                and item["activation"] == "active" and item["requested_revision"] == "main"
                and item.get("file_availability") == "complete")

    def append(self, candidate):
        if len(self.candidates) >= MAX_CANDIDATES:
            raise ValueError("dependency candidate limit exceeded")
        if candidate not in self.candidates:
            self.candidates.append(candidate)

    def unresolved(self, filename, node, state, chain, alternative, reason):
        item = {"repo_id": None, "role": "unknown", "loader": None, "required": "candidate",
                "activation": state, "dependency_kind": "external", "state": "unresolved",
                "source": f"{filename}:{node.lineno}", "expression": reason, "call_chain": chain,
                "requested_revision": None}
        if alternative:
            item["alternative"] = alternative
        self.append(item)


def analyze_dependencies(sources, config, model_id, *, files=(), metadata=None, transformers_version=None):
    try:
        return DependencyAnalyzer(sources, config, model_id, files=files, metadata=metadata,
                                  transformers_version=transformers_version).run()
    except RecursionError as exc:
        raise ValueError("dependency expression nesting exceeds analysis limit") from exc
