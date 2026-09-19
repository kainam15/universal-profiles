"""仅由声明使用 Torch 的运行时按需加载。"""

import torch


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


def metadata() -> dict:
    return {'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda}
