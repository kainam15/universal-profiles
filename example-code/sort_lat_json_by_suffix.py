# sort_lat_json_by_suffix.py
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

SUFFIX_RE = re.compile(r":(\d+)$")

def suffix_num(k: str):
    """
    从 key 末尾提取 :xxxxx 的数字。
    提取不到就返回 None。
    """
    m = SUFFIX_RE.search(k)
    return int(m.group(1)) if m else None

def sort_items(items):
    """
    先按 suffix 数字升序；没有 suffix 的放最后；
    suffix 相同再按 key 字典序稳定排序。
    """
    def key_fn(kv):
        k, _ = kv
        n = suffix_num(k)
        return (0, n, k) if n is not None else (1, 10**30, k)
    return sorted(items, key=key_fn)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sort_lat_json_by_suffix.py <input.json> [output.json]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path.with_suffix(".sorted.json")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("输入 JSON 顶层不是 dict，无法按 key 排序。")

    sorted_data = OrderedDict(sort_items(data.items()))

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)

    print(f"OK: wrote sorted json to {out_path}")

if __name__ == "__main__":
    main()