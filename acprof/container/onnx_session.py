"""Shared ONNX session and named tensor checks; task meaning stays in handlers."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import onnxruntime as ort

from acprof.container.handlers.structured import _artifact_path, _local_snapshot
from acprof.runtime_settings import onnx_runtime_parameters


TENSOR_DTYPES = {
    'tensor(float)': np.dtype('float32'), 'tensor(double)': np.dtype('float64'),
    'tensor(int64)': np.dtype('int64'), 'tensor(int32)': np.dtype('int32'),
    'tensor(bool)': np.dtype('bool'),
}


def load_session(model_source: str, task_type: str, backend: str, device: str,
                 model_revision: str = 'main', load_options: dict | None = None, *,
                 runtime: Any = None) -> dict:
    if backend != 'onnxruntime' or device != 'cpu':
        raise ValueError('unsupported: ONNX Runtime extension is CPU only and requires backend=onnxruntime')
    if load_options:
        raise ValueError('unsupported: ONNX Runtime does not implement Torch attention/profiler options')
    root = _local_snapshot(model_source, model_revision)
    manifest_file = root / 'acprof_model.json'
    manifest = {}
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
                or isinstance(manifest.get('schema_version'), bool)
                or manifest.get('format') != 'onnxruntime' or manifest.get('task') != task_type):
            raise ValueError('acprof_model.json must declare schema_version=1, selected task, format=onnxruntime')
        artifact = _artifact_path(root, manifest.get('model_file'))
    else:
        artifacts = sorted(root.glob('*.onnx'))
        if len(artifacts) != 1:
            raise ValueError('ONNX requires exactly one *.onnx file or acprof_model.json selecting model_file')
        artifact = artifacts[0]
    runtime = ort if runtime is None else runtime
    parameters = onnx_runtime_parameters()
    options = runtime.SessionOptions()
    options.intra_op_num_threads = parameters['effective']['intra_op_threads']
    options.inter_op_num_threads = parameters['effective']['inter_op_threads']
    options.execution_mode = (runtime.ExecutionMode.ORT_PARALLEL if options.inter_op_num_threads > 1
                              else runtime.ExecutionMode.ORT_SEQUENTIAL)
    session = runtime.InferenceSession(str(artifact), sess_options=options,
                                       providers=parameters['requested']['providers'])
    actual_providers = session.get_providers()
    if actual_providers != parameters['requested']['providers']:
        raise ValueError('ONNX CPU provider binding differs from declaration; fallback is disabled')
    parameters['effective']['providers'] = actual_providers
    parameters['effective']['operator_device_assignment'] = 'cpu_only_provider'
    actual_options = session.get_session_options()
    parameters['effective'].update(
        intra_op_threads=actual_options.intra_op_num_threads,
        inter_op_threads=actual_options.inter_op_num_threads,
        threads=actual_options.intra_op_num_threads,
        execution_mode=str(actual_options.execution_mode).rsplit('.', 1)[-1],
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    for kind, specs in (('input', inputs), ('output', outputs)):
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise ValueError(f'ONNX {kind} signature must have unique named tensors')
        for spec in specs:
            if spec.type not in TENSOR_DTYPES:
                raise ValueError(f'unsupported ONNX {kind} dtype: {spec.name}={spec.type}')
        parameters[kind + 's'] = [{'name': spec.name, 'type': spec.type, 'shape': list(spec.shape)}
                                  for spec in specs]
    declared_hash = manifest.get('artifact_sha256')
    if declared_hash is not None and (not isinstance(declared_hash, str) or len(declared_hash) != 64
                                      or any(c not in '0123456789abcdef' for c in declared_hash)):
        raise ValueError('artifact_sha256 must be a lowercase SHA256 hex digest')
    artifact_info = {'model_file': str(artifact.relative_to(root)), 'sha256': declared_hash,
                     'verification': 'declared' if declared_hash else 'unknown'}
    parameters['artifact'] = artifact_info
    return {'model': session, 'task_type': task_type, 'backend': backend, 'device': device,
            'model_revision': model_revision, 'runtime_version': runtime.__version__,
            'model_format': 'onnxruntime', 'dtype': 'float32', 'model_root': root,
            'manifest': manifest, 'input_specs': inputs, 'output_specs': outputs,
            'artifact_path': artifact, 'artifact': artifact_info, 'runtime_parameters': parameters}


def validate_artifact(context: dict) -> None:
    """Audit bytes only in independent output validation, outside load and measurement."""
    import onnx

    path = context['artifact_path']
    model = onnx.load_model(str(path), load_external_data=False)

    def reject_external(message):
        if isinstance(message, onnx.TensorProto) and (message.data_location == onnx.TensorProto.EXTERNAL
                                                      or message.external_data):
            raise ValueError('unsupported ONNX external tensor data; artifact audit requires a self-contained graph')
        for descriptor, value in message.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                if descriptor.is_repeated:
                    for child in value:
                        reject_external(child)
                else:
                    reject_external(value)

    reject_external(model)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    expected = context['manifest'].get('artifact_sha256')
    if expected is not None and digest.hexdigest() != expected:
        raise ValueError('ONNX artifact SHA256 differs from acprof_model.json')
    context['artifact'].update(sha256=digest.hexdigest(), verification='verified', scope='self_contained_onnx_graph')
    context['artifact_sha256'] = digest.hexdigest()


def _validate_tensors(specs: list, tensors: dict, *, kind: str) -> None:
    expected = {spec.name for spec in specs}
    if set(tensors) != expected:
        raise ValueError(f'ONNX named {kind}s must match signature: expected={sorted(expected)}, actual={sorted(tensors)}')
    dimensions: dict[str, int] = {}
    for spec in specs:
        value = tensors[spec.name]
        if not isinstance(value, np.ndarray) or value.dtype != TENSOR_DTYPES[spec.type]:
            raise ValueError(f'ONNX {kind} {spec.name} dtype must be {TENSOR_DTYPES[spec.type]}')
        if len(value.shape) != len(spec.shape) or any(size <= 0 for size in value.shape):
            raise ValueError(f'ONNX {kind} {spec.name} shape must match {spec.shape}; received {list(value.shape)}')
        for declared, actual in zip(spec.shape, value.shape):
            if isinstance(declared, int) and declared > 0 and actual != declared:
                raise ValueError(f'ONNX fixed {kind} shape {spec.shape} for {spec.name}; received {list(value.shape)}')
            if isinstance(declared, str) and declared:
                if declared in dimensions and dimensions[declared] != actual:
                    raise ValueError(f'ONNX symbolic shape {declared!r} differs across named {kind}s')
                dimensions[declared] = actual


def validate_inputs(context: dict, inputs: dict) -> None:
    _validate_tensors(context['input_specs'], inputs, kind='input')


def tensor_metadata(inputs: dict) -> dict:
    """Describe already materialized arrays without reading their values."""
    return {name: {'dtype': str(value.dtype), 'shape': list(value.shape)} for name, value in inputs.items()}


def validate_outputs(context: dict, outputs: dict) -> None:
    _validate_tensors(context['output_specs'], outputs, kind='output')


def run_session(context: dict, inputs: dict) -> dict:
    names = [spec.name for spec in context['output_specs']]
    values = context['model'].run(names, inputs)
    if len(values) != len(names):
        raise ValueError('ONNX returned an incomplete set of named outputs')
    return dict(zip(names, values))
