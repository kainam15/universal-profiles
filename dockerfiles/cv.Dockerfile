# syntax=docker/dockerfile:1

ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

# CV-specific dependencies
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'torch>=2.2' \
    'transformers>=4.40' \
    torchvision \
    Pillow \
    accelerate

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
