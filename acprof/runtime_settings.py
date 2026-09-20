"""Small shared runtime settings; Docker quota and affinity remain separate."""
from __future__ import annotations

import math
import os


RUNTIME_ENV_NAMES = (
    'ACPROF_RUNTIME_THREADS', 'ACPROF_ONNX_INTRA_OP_THREADS',
    'ACPROF_ONNX_INTER_OP_THREADS', 'ACPROF_ONNX_PROVIDERS',
    'ACPROF_REQUEST_TIMEOUT_S', 'TORCH_NUM_THREADS',
)


def runtime_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in RUNTIME_ENV_NAMES if name in os.environ}


def runtime_docker_env_args() -> list[str]:
    return [part for name, value in runtime_environment().items() for part in ('-e', f'{name}={value}')]


def _integer(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer >= {minimum}') from exc
    if value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def runtime_threads(default: int = 0) -> int:
    name = 'ACPROF_RUNTIME_THREADS' if 'ACPROF_RUNTIME_THREADS' in os.environ else 'TORCH_NUM_THREADS'
    return _integer(name, default)


def request_timeout_s() -> float | None:
    name = 'ACPROF_REQUEST_TIMEOUT_S'
    raw = os.environ.get(name, '300')
    if raw.strip().lower() == 'none':
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def onnx_runtime_parameters() -> dict:
    intra_name = next((name for name in ('ACPROF_ONNX_INTRA_OP_THREADS', 'ACPROF_RUNTIME_THREADS',
                                       'TORCH_NUM_THREADS') if name in os.environ), None)
    # Legacy zero meant one ORT thread; retain that default. Explicit ORT values must be positive.
    intra = max(1, _integer(intra_name, 1, minimum=1 if intra_name == 'ACPROF_ONNX_INTRA_OP_THREADS' else 0)) if intra_name else 1
    inter_name = 'ACPROF_ONNX_INTER_OP_THREADS'
    inter = _integer(inter_name, 1, minimum=1)
    raw_providers = os.environ.get('ACPROF_ONNX_PROVIDERS', 'CPUExecutionProvider')
    providers = [name.strip() for name in raw_providers.split(',')]
    if providers != ['CPUExecutionProvider']:
        raise ValueError('unsupported ONNX providers: this CPU profile requires CPUExecutionProvider; fallback is disabled')
    return {
        'requested': {'runtime': 'onnxruntime', 'device': 'cpu', 'providers': providers,
                      'intra_op_threads': int(os.environ[intra_name]) if intra_name else None,
                      'inter_op_threads': int(os.environ[inter_name]) if inter_name in os.environ else None,
                      'allow_fallback': False},
        'effective': {'runtime': 'onnxruntime', 'device': 'cpu', 'providers': providers,
                      'threads': intra, 'intra_op_threads': intra, 'inter_op_threads': inter},
        'source': {'intra_op_threads': intra_name or 'default',
                   'inter_op_threads': inter_name if inter_name in os.environ else 'default'},
    }
