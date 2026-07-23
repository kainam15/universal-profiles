"""AC-Prof Universal Flask Server - dynamic handler routing by task family."""

from __future__ import annotations

import os
import time

# Inference containers must never resolve model artifacts over the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Environment-driven configuration
# ─────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
TASK_FAMILY = os.getenv("TASK_FAMILY", "nlp")
TASK_TYPE = os.getenv("TASK_TYPE", "text-generation")
RUNTIME_BACKEND = os.getenv("RUNTIME_BACKEND", "transformers_pipeline")
USE_GPU = int(os.getenv("USE_GPU", "0"))

device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Load handler and model
# ─────────────────────────────────────────────
from acprof.container.handlers import HandlerRegistry, resolve_model_source  # noqa: E402

handler = HandlerRegistry.get(TASK_FAMILY, RUNTIME_BACKEND)
MODEL_SOURCE = resolve_model_source(MODEL_ID, os.getenv("MODEL_LOCAL_PATH"))

print(f"[server] Loading model: {MODEL_ID}")
print(f"[server] Revision: {MODEL_REVISION}")
print(f"[server] Source: {MODEL_SOURCE}")
print(f"[server] Task: {TASK_TYPE} (family={TASK_FAMILY}, backend={RUNTIME_BACKEND})")
print(f"[server] Device: {device}")

t_load_start = time.perf_counter()
model_ctx = handler.load(MODEL_SOURCE, TASK_TYPE, RUNTIME_BACKEND, device, MODEL_REVISION)
t_load_end = time.perf_counter()
load_time_s = t_load_end - t_load_start

print(f"[server] Model loaded in {load_time_s:.2f}s")


def _extract_probe_metadata(processed: object) -> dict:
    if not isinstance(processed, dict):
        return {}
    return {
        "effective_input_scale": processed.get("_effective_input_scale"),
        "truncated_by_limit": processed.get("_truncated_by_limit"),
        "reason": processed.get("_probe_reason", ""),
    }


def _request_json_body() -> dict:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {}

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.route("/ready")
def ready():
    return jsonify({
        "status": "ok",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": device,
        "load_time_s": round(load_time_s, 3),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = _request_json_body()
    try:
        processed = handler.preprocess(model_ctx, data)
        with torch.inference_mode():
            output = handler.predict(model_ctx, processed)
        result = handler.postprocess(model_ctx, output)
        metadata = _extract_probe_metadata(processed)
        effective_input_scale = metadata.get("effective_input_scale")
        if effective_input_scale is not None:
            result["effective_input_scale"] = effective_input_scale
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/probe", methods=["POST"])
def probe():
    data = _request_json_body()
    try:
        processed = handler.preprocess(model_ctx, data)
        return jsonify(_extract_probe_metadata(processed))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scale_meta", methods=["GET", "POST"])
def scale_meta():
    data = _request_json_body()
    try:
        metadata = handler.get_scale_metadata(model_ctx, data)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/meta")
def meta():
    return jsonify({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "task_family": TASK_FAMILY,
        "task_type": TASK_TYPE,
        "runtime_backend": RUNTIME_BACKEND,
        "device": device,
        "load_time_s": round(load_time_s, 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
