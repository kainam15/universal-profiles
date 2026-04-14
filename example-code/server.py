from flask import Flask, request, jsonify
import torch
from chronos import ChronosBoltPipeline
import os

app = Flask(__name__)

MODEL_ID = os.getenv("MODEL_ID", "amazon/chronos-bolt-base")
USE_GPU = int(os.getenv("USE_GPU", "0"))

device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

pipeline = ChronosBoltPipeline.from_pretrained(
    MODEL_ID,
    device_map=device,
    local_files_only=True
)

@app.route("/ready")
def ready():
    return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    pred_len = int(data["prediction_length"])

    raw = data["context"]

    # raw 可以是 [L] 或 [[L], [L], ...]
    context = torch.tensor(raw, dtype=torch.float32, device=device)
    if context.ndim == 1:
        # [L] -> [1, L]
        context = context.unsqueeze(0)
    elif context.ndim == 2:
        # [B, L] OK
        pass
    else:
        return jsonify({"error": f"invalid context ndim={context.ndim}"}), 400

    with torch.inference_mode():
        forecast = pipeline.predict(context, prediction_length=pred_len)

    return jsonify({"forecast_shape": list(forecast.shape)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)