FROM python:3.10-slim

WORKDIR /app

# 固定 cache 目录（镜像里）
ENV MODEL_CACHE_DIR=/models
ENV HF_HOME=/models/hf
ENV SENTENCE_TRANSFORMERS_HOME=/models

# ✅ 走 hf-mirror（大陆更稳）
ENV HF_ENDPOINT=https://hf-mirror.com

# ✅ 避免并发下载导致偶发问题（可选，但很推荐）
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# ✅ 预下载权重（固化进镜像 layer）
COPY download_model.py .
RUN python download_model.py

COPY server.py .

EXPOSE 8002
CMD ["python", "server.py"]