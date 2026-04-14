"""Download and cache HuggingFace model weights.

Used both during Docker image build and host-side pre-warming.
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.getenv("HF_HOME", "/models/hf"))

if not MODEL_ID:
    raise SystemExit("MODEL_ID environment variable is required")

os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Downloading: {MODEL_ID} -> {CACHE_DIR}")
snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR, resume_download=True)
print("Done.")
