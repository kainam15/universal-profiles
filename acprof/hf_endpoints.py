"""Shared, explicit Hugging Face Hub endpoint policy (standard library only)."""
from __future__ import annotations

import os
from typing import Mapping
from urllib.parse import urlsplit

HF_DEFAULT_ENDPOINT = "https://huggingface.co"


def normalize_hf_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(char.isspace() or char in ",;" for char in endpoint))
        parsed.port  # Validate the port without including the supplied URL in errors.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("HF endpoint must be an HTTP(S) URL without credentials, query or fragment")
    return endpoint


def hf_endpoints(environ: Mapping[str, str] | None = None) -> list[str]:
    """Use the official Hub unless the caller explicitly selects another endpoint."""
    env = os.environ if environ is None else environ
    primary = (env.get("HF_ENDPOINT", "").strip()
               or env.get("HF_HUB_ENDPOINT", "").strip() or HF_DEFAULT_ENDPOINT)
    endpoints = [normalize_hf_endpoint(primary)]
    for value in env.get("HF_FALLBACK_ENDPOINTS", "").replace(";", ",").split(","):
        if value.strip():
            endpoint = normalize_hf_endpoint(value)
            if endpoint not in endpoints:
                endpoints.append(endpoint)
    return endpoints
