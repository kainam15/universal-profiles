# download_model.py  (Chronos 版本)
import os
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "amazon/chronos-bolt-base")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")

os.makedirs(CACHE_DIR, exist_ok=True)

print("Downloading:", MODEL_ID)
snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR, resume_download=True)
print("Done.")