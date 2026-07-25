#!/usr/bin/env python3
import json
import subprocess
import sys
from typing import Sequence


# 用法：
#   python3 -m acprof.packet.sniff_parse_pcap <pcap> <port>
#
# 输出 JSON:
#   {"<sniff_group_id>:<req_frame>": latency_s, ...}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr.strip() or p.stdout.strip())
    return p.stdout


def extract_group_id_from_request_lines(req_lines: str) -> str:
    """
    req_lines 通常包含多条 request line（请求行+headers）
    例如： 'POST /predict HTTP/1.1,Host: ...,Connection: close,X-Req-Id: abc:123,...'
    我们从中找到 X-Req-Id，并取其冒号前缀作为 group_id。
    """
    if not req_lines:
        return "group"

    # tshark 对重复字段通常用逗号连接
    parts = req_lines.split(",")
    for p in parts:
        p = p.strip()
        if p.lower().startswith("x-req-id:"):
            v = p.split(":", 1)[1].strip()  # 取 header value
            # client: "{sniff_group_id}:{k}"
            if ":" in v:
                return v.split(":", 1)[0].strip()
            return v.strip() or "group"
    return "group"


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m acprof.packet.sniff_parse_pcap <pcap> <port>"
        )

    pcap = args[0]
    port = args[1] if len(args) > 1 else "8002"

    # 1) request: /predict 的 request frame -> time + group_id
    req_cmd = [
        "tshark", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={port},http",
        "-Y", f"tcp.port=={port} && http.request && http.request.uri==\"/predict\"",
        "-T", "fields", "-E", "separator=\t",
        "-e", "frame.number",
        "-e", "frame.time_epoch",
        "-e", "http.request.line",
    ]

    req_time = {}
    req_gid = {}
    for line in run(req_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        fn = parts[0].strip()
        t = parts[1].strip()
        req_lines = parts[2] if len(parts) >= 3 else ""

        if not fn.isdigit():
            continue
        try:
            req_fn = int(fn)
            req_time[req_fn] = float(t)
            req_gid[req_fn] = extract_group_id_from_request_lines(req_lines)
        except Exception:
            pass

    # 2) response: http.request_in 指回对应 request frame -> response time
    resp_cmd = [
        "tshark", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={port},http",
        "-Y", f"tcp.port=={port} && http.response",
        "-T", "fields", "-E", "separator=\t",
        "-e", "http.request_in",
        "-e", "frame.time_epoch",
        "-e", "http.response.code",
    ]

    out = {}
    for line in run(resp_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        req_in, t = parts[0].strip(), parts[1].strip()
        if not req_in.isdigit():
            continue

        req_fn = int(req_in)
        if req_fn in req_time:
            try:
                dt = float(t) - float(req_time[req_fn])
            except Exception:
                continue
            if dt >= 0:
                gid = req_gid.get(req_fn, "group")
                out[f"{gid}:{req_fn}"] = dt

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
