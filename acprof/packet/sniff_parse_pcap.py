#!/usr/bin/env python3
import json
import subprocess
import sys
from typing import Sequence


# 用法：
#   python3 -m acprof.packet.sniff_parse_pcap <pcap> <port>
#
# 输出 JSON schema v2：每个请求包含时延、线上帧字节、TCP payload 和
# L2/L3/L4 协议开销。连接由客户端显式关闭，因此一个 tcp.stream 对应
# 一个 /predict 请求。


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


def _parse_int_field(raw: str) -> int | None:
    value = str(raw or "").split(",", 1)[0].strip()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m acprof.packet.sniff_parse_pcap <pcap> <port>"
        )

    pcap = args[0]
    port = args[1] if len(args) > 1 else "8002"
    server_port = _parse_int_field(port)
    if server_port is None:
        raise SystemExit(f"invalid TCP port: {port!r}")

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
        "-e", "tcp.stream",
    ]

    req_time = {}
    req_gid = {}
    req_stream = {}
    for line in run(req_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        fn = parts[0].strip()
        t = parts[1].strip()
        req_lines = parts[2] if len(parts) >= 3 else ""
        stream = parts[3].strip() if len(parts) >= 4 else ""

        if not fn.isdigit():
            continue
        try:
            req_fn = int(fn)
            req_time[req_fn] = float(t)
            req_gid[req_fn] = extract_group_id_from_request_lines(req_lines)
            stream_id = _parse_int_field(stream)
            if stream_id is not None:
                req_stream[req_fn] = stream_id
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

    latency_by_request = {}
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
                latency_by_request[f"{gid}:{req_fn}"] = dt

    # 3) Sum captured wire bytes and TCP payload for each /predict stream.
    # frame.len includes the captured link-layer frame; tcp.len excludes
    # Ethernet/IP/TCP headers. Retransmissions are intentionally retained.
    stream_cmd = [
        "tshark", "-r", pcap,
        "-Y", f"tcp.port=={port}",
        "-T", "fields", "-E", "separator=\t",
        "-e", "tcp.stream",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "frame.len",
        "-e", "tcp.len",
    ]
    stream_stats = {}
    for line in run(stream_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        stream_id = _parse_int_field(parts[0])
        src_port = _parse_int_field(parts[1])
        dst_port = _parse_int_field(parts[2])
        frame_len = _parse_int_field(parts[3])
        tcp_payload_len = _parse_int_field(parts[4])
        if stream_id is None or frame_len is None or tcp_payload_len is None:
            continue
        stats = stream_stats.setdefault(
            stream_id,
            {
                "request_wire_bytes": 0,
                "response_wire_bytes": 0,
                "tcp_payload_bytes": 0,
            },
        )
        if dst_port == server_port:
            stats["request_wire_bytes"] += frame_len
        elif src_port == server_port:
            stats["response_wire_bytes"] += frame_len
        else:
            continue
        stats["tcp_payload_bytes"] += tcp_payload_len

    requests = {}
    for request_id, latency_s in latency_by_request.items():
        req_fn = int(request_id.rsplit(":", 1)[1])
        stream_id = req_stream.get(req_fn)
        stats = stream_stats.get(stream_id) if stream_id is not None else None
        record = {"latency_s": latency_s}
        if stats is not None:
            request_wire = int(stats.get("request_wire_bytes", 0))
            response_wire = int(stats.get("response_wire_bytes", 0))
            tcp_payload = int(stats.get("tcp_payload_bytes", 0))
            total_wire = request_wire + response_wire
            protocol_overhead = max(0, total_wire - tcp_payload)
            record.update({
                "request_wire_bytes": request_wire,
                "response_wire_bytes": response_wire,
                "total_wire_bytes": total_wire,
                "tcp_payload_bytes": tcp_payload,
                "protocol_overhead_bytes": protocol_overhead,
                "protocol_overhead_ratio": (
                    protocol_overhead / total_wire if total_wire > 0 else None
                ),
            })
        requests[request_id] = record

    print(json.dumps({"schema_version": 2, "requests": requests}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
