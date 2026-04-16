"""AC-Prof Universal Flask Server - dynamic handler routing by task family."""

from __future__ import annotations

import os
import time

import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Environment-driven configuration
# ─────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "")
TASK_FAMILY = os.getenv("TASK_FAMILY", "nlp")
TASK_TYPE = os.getenv("TASK_TYPE", "text-generation")
RUNTIME_BACKEND = os.getenv("RUNTIME_BACKEND", "transformers_pipeline")
USE_GPU = int(os.getenv("USE_GPU", "0"))

device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Load handler and model
# ─────────────────────────────────────────────
from handlers import HandlerRegistry  # noqa: E402

handler = HandlerRegistry.get(TASK_FAMILY, RUNTIME_BACKEND)

print(f"[server] Loading model: {MODEL_ID}")
print(f"[server] Task: {TASK_TYPE} (family={TASK_FAMILY}, backend={RUNTIME_BACKEND})")
print(f"[server] Device: {device}")

t_load_start = time.perf_counter()
model_ctx = handler.load(MODEL_ID, TASK_TYPE, RUNTIME_BACKEND, device)
t_load_end = time.perf_counter()
load_time_s = t_load_end - t_load_start

print(f"[server] Model loaded in {load_time_s:.2f}s")

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.route("/ready")
def ready():
    return jsonify({
        "status": "ok",
        "model_id": MODEL_ID,
        "device": device,
        "load_time_s": round(load_time_s, 3),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        processed = handler.preprocess(model_ctx, data)
        with torch.inference_mode():
            output = handler.predict(model_ctx, processed)
        result = handler.postprocess(model_ctx, output)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/meta")
def meta():
    return jsonify({
        "model_id": MODEL_ID,
        "task_family": TASK_FAMILY,
        "task_type": TASK_TYPE,
        "runtime_backend": RUNTIME_BACKEND,
        "device": device,
        "load_time_s": round(load_time_s, 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
