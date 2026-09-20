"""仅由声明使用 Torch 的运行时按需加载。"""

import torch
import time

from acprof.runtime_settings import runtime_environment


def prepare_device(use_gpu: bool, threads: int = 0) -> str:
    if threads > 0:
        torch.set_num_threads(threads)
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('请求 GPU，但容器内 CUDA 不可用')
        torch.cuda.init()
        return 'cuda'
    return 'cpu'


def inference_context():
    return torch.inference_mode()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wait_for_completion(model_ctx: dict, output, *, timeout_s: float | None):
    """Wait for the request's current CUDA stream, without a device-wide barrier.

    Handlers using auxiliary streams must join them onto their current stream
    before returning, or provide their own execution module/completion hook.
    """
    if str(model_ctx.get('device', 'cpu')).startswith('cuda'):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=model_ctx['device']))
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not event.query():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('request completion timed out on the CUDA stream')
            time.sleep(0.001)
    return output


def metadata() -> dict:
    requests = runtime_environment()
    source = next((key for key in ('ACPROF_RUNTIME_THREADS', 'TORCH_NUM_THREADS') if key in requests), None)
    return {'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'runtime_parameters': {
                'requested': {'runtime': 'torch', 'threads': int(requests[source]) if source else None},
                'effective': {'runtime': 'torch', 'threads': torch.get_num_threads()},
                'source': {'threads': source or 'runtime_default'},
            }}
