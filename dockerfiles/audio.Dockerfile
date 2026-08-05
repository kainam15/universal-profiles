# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# Audio-specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    'transformers>=4.40' \
    torchaudio \
    librosa \
    soundfile \
    accelerate

# Recent PyTorch/Triton releases JIT-compile a small CUDA launcher during
# eager GPU inference.  The slim Python base image has no C toolchain, which
# makes the Torch FLOP probe fail before profiling starts.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download model weights (baked into image layer)
ARG MODEL_ID
ARG MODEL_REVISION=main
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_REVISION=${MODEL_REVISION}
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then \
        export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    fi; \
    python -m acprof.container.download_model

CMD ["python", "-m", "acprof.container.server"]
