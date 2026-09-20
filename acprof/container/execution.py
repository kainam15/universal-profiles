"""可选的运行时执行钩子；默认 CPU 执行无需任何推理框架。"""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from importlib import import_module
from inspect import isawaitable
import sys

from acprof.runtime_settings import request_timeout_s


def prepare_device(use_gpu: bool, threads: int = 0) -> str:
    if use_gpu:
        raise ValueError('unsupported: selected execution runtime has no CUDA implementation')
    return 'cpu'


def inference_context():
    return nullcontext()


def synchronize() -> None:
    pass


def metadata() -> dict:
    return {}


def complete_prediction(runtime, model_ctx: dict, output):
    """Resolve this request before postprocess, response, or the timing window ends.

    Existing runtimes without the optional hook promise synchronous predict.
    Async runtimes must implement wait_for_completion and propagate backend errors
    and timeout; the hook returns the completed raw output. No validation runs here.
    """
    wait = getattr(runtime, 'wait_for_completion', None)
    if wait is not None:
        timeout = request_timeout_s()
        try:
            output = wait(model_ctx, output, timeout_s=timeout)
        except (TimeoutError, FutureTimeoutError) as exc:
            limit = f'after {timeout:g}s' if timeout is not None else '(no configured deadline)'
            raise TimeoutError(f'request completion timed out {limit}: {exc}') from exc
    if isinstance(output, Future) or isawaitable(output):
        raise TypeError('execution must resolve asynchronous output with wait_for_completion before returning')
    return output


def configured_execution(family: str, backend: str, *, use_gpu: bool = False,
                         threads: int = 0, adapter: str = 'family-default'):
    from acprof.extensions import get_extension
    from acprof.container.handlers import HandlerDependencyMissingError, HandlerModuleImportError

    declaration = get_extension(family, backend, adapter=adapter)
    device = 'cuda' if use_gpu else 'cpu'
    if declaration.execution.get(device) == 'unsupported':
        raise ValueError(f'unsupported: backend={backend} device={device}')
    module_name = declaration.execution_entrypoint
    try:
        runtime = import_module(module_name) if module_name else sys.modules[__name__]
    except ModuleNotFoundError as exc:
        raise HandlerDependencyMissingError(
            f'backend: {backend}; module: {module_name}; missing dependency: {exc.name}; '
            f'original exception: {type(exc).__name__}: {exc}'
        ) from exc
    except Exception as exc:
        raise HandlerModuleImportError(
            f'backend: {backend}; module: {module_name}; original exception: {type(exc).__name__}: {exc}'
        ) from exc
    return runtime, runtime.prepare_device(use_gpu, threads)
