# syntax=docker/dockerfile:1
ARG BASE_IMAGE=acprof-base:latest
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGE_SPEC=torch>=2.7

RUN pip install --no-cache-dir "${TORCH_PACKAGE_SPEC}" --index-url ${TORCH_INDEX_URL}

# sklearn/skops is CPU-only. No training or simulator stack.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    'scikit-learn==1.7.2' 'skops==0.13.0'

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
