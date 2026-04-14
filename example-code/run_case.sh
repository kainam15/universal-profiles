#!/bin/bash
set -euo pipefail

CPU=$1
MEM=$2
GPU=$3

NAME="case_${CPU}c_${MEM}g_${GPU}"
IMAGE="chronos-bolt-base_server"

HOST_PORT=$((8002 + CPU*100 + MEM))
export SNIFF_IFACE=docker0

export MODEL_NAME="amazon/chronos-bolt-base"
export CPU_CORES="$CPU"
export MEM_CAP_GB="$MEM"
export GPU_MODE="$GPU"
export CASE_NAME="$NAME"

export BASE_URL="http://127.0.0.1:${HOST_PORT}"
export ENDPOINT="/predict"

export BATCH_SIZE="128"
export CONTEXT_LENS="64,128,256,512,1024,2048"
export PRED_LEN="64"
export WARMUP="2"
export REPEAT="5"
export REPEAT_IN_WINDOW="${REPEAT_IN_WINDOW:-20}"

export SAMPLE_HZ="20"
export IDLE_SECONDS="3"

# ✅ 每个 case 单独写一个 CSV，避免被后续 case 追加写坏/覆盖
export OUT_CSV="result_${NAME}.csv"

export HF_HOME="$(pwd)/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_TELEMETRY=1

MODEL_DIR="$HF_HOME/models--amazon--chronos-bolt-base"

if [ ! -d "$MODEL_DIR" ]; then
  echo "[PREWARM] Cache missing: $MODEL_DIR"
  unset HF_HUB_OFFLINE || true
  unset TRANSFORMERS_OFFLINE || true
  python -c "import huggingface_hub" >/dev/null 2>&1 || python -m pip install -U huggingface_hub
  HF_HOME="$HF_HOME" python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = "amazon/chronos-bolt-base"
cache_dir = os.environ["HF_HOME"]
print("[PREWARM] snapshot_download:", repo_id, "cache_dir=", cache_dir)
snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True)
print("[PREWARM] done")
PY
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting $NAME on ${HOST_PORT} ..."

if [ "$GPU" == "on" ]; then
  GPU_FLAG="--gpus all"
  USE_GPU=1
else
  GPU_FLAG=""
  USE_GPU=0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

t0_ms=$(date +%s%3N)
docker run -d \
  --name "$NAME" \
  --cpus="$CPU" \
  --memory="${MEM}g" \
  $GPU_FLAG \
  -e MODEL_ID="$MODEL_NAME" \
  -e USE_GPU="$USE_GPU" \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e HF_HOME=/models/hf \
  -e HF_HUB_CACHE=/models/hf \
  -e TRANSFORMERS_CACHE=/models/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_ENDPOINT=https://huggingface.co \
  -v "$(pwd)/hf-cache:/models/hf" \
  -p ${HOST_PORT}:8002 \
  "$IMAGE"

READY_OK=0
for _ in {1..600}; do
  if curl -sf "http://127.0.0.1:${HOST_PORT}/ready" >/dev/null; then
    READY_OK=1
    break
  fi
  sleep 0.1
done

t1_ms=$(date +%s%3N)
cold_start_s=$(awk "BEGIN{printf \"%.3f\", ($t1_ms-$t0_ms)/1000}")
export COLD_START_S=$cold_start_s

if [ "$READY_OK" -ne 1 ]; then
  echo "[ERROR] Server not ready after wait. cold_start_s=$cold_start_s"
  docker logs "$NAME" --tail 200 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 1
fi

# ----------------------------
# docker0 packet sniffer
# ----------------------------
IFACE="${SNIFF_IFACE:-docker0}"
PCAP="sniff_${NAME}.pcap"
LAT_JSON="lat_${NAME}.json"

echo "[SNIFF] tcpdump on iface=$IFACE (pcap=$PCAP)"
sudo tcpdump -i "$IFACE" -s 0 -B 4096 -w "$PCAP" "tcp port 8002" >/dev/null 2>&1 &
TCPDUMP_PID=$!

cleanup_sniff() {
  if ps -p "$TCPDUMP_PID" >/dev/null 2>&1; then
    sudo kill -INT "$TCPDUMP_PID" >/dev/null 2>&1 || true
    wait "$TCPDUMP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_sniff EXIT

# ✅ 等 tcpdump 真正开始抓
sleep 0.3

echo "Running benchmark..."
python3 client.py

# ✅ 关键：停抓包前等最后一波包回来（ctx=2048 特别需要）
sleep 1.0

cleanup_sniff
trap - EXIT

# ✅ 只需要很短的写盘缓冲
sleep 0.2

echo "[SNIFF] parsing pcap -> packet latencies..."
set -x
python3 sniff_parse_pcap.py "$PCAP" 8002 > "$LAT_JSON"
set +x

if [ ! -s "$LAT_JSON" ]; then
  echo "[SNIFF][WARN] $LAT_JSON empty; writing {}"
  echo "{}" > "$LAT_JSON"
fi

echo "[SNIFF] merging packet latency back to CSV..."
python3 merge_packet_latency.py "$OUT_CSV" "$LAT_JSON" "result_${NAME}_sniff.csv"
mv "result_${NAME}_sniff.csv" "$OUT_CSV"

docker stop "$NAME" >/dev/null
docker rm "$NAME" >/dev/null

echo "Done. Wrote: $OUT_CSV"