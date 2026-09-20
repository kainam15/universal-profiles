"""从锁定版本的官方 modeling_auto.py 导出静态注册表；不执行上游代码。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def export_support(source: bytes, version: str) -> dict:
    mappings = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], "id", "")
        if not name.endswith("_MAPPING_NAMES"):
            continue
        if not isinstance(node.value, ast.Call) or getattr(node.value.func, "id", "") != "OrderedDict" or len(node.value.args) != 1:
            raise ValueError(f"unsupported Auto registry definition: {name}")
        if node.value.keywords or not isinstance(node.value.args[0], (ast.List, ast.Tuple)):
            raise ValueError(f"unsupported Auto registry definition: {name}")
        pairs = []
        for item in node.value.args[0].elts:
            if isinstance(item, ast.Starred):
                # Upstream composes multimodal mappings as *list(OTHER.items()).
                call = item.value
                if not (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "list"
                        and len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Call)
                        and isinstance(call.args[0].func, ast.Attribute)
                        and call.args[0].func.attr == "items" and not call.args[0].args and not call.args[0].keywords):
                    raise ValueError("unsupported Auto registry expansion")
                referenced = getattr(call.args[0].func.value, "id", "")
                if referenced not in mappings:
                    raise ValueError(f"unsupported Auto registry reference: {referenced}")
                pairs.extend(mappings[referenced].items())
            else:
                pairs.append(ast.literal_eval(item))
        mappings[name] = dict(pairs)
    if not mappings.get("MODEL_MAPPING_NAMES") or not mappings.get("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"):
        raise ValueError("upstream Auto registry layout is unsupported; no partial catalog is published")
    return {"schema_version": 1, "version": version,
            "source": f"https://github.com/huggingface/transformers/blob/v{version}/src/transformers/models/auto/modeling_auto.py",
            "source_sha256": hashlib.sha256(source).hexdigest(), "mappings": mappings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = export_support(args.source.read_bytes(), args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
