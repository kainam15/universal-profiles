"""小型 CPU ONNX dense tabular 接口；只有选择此扩展才导入 ORT。"""

from __future__ import annotations

from typing import Any

import onnxruntime as ort

from acprof.container.handlers import BaseHandler
from acprof.container.handlers.structured import _dense_matrix, _positive_integer
from acprof.container.onnx_session import load_session, run_session, tensor_metadata, validate_artifact, validate_inputs, validate_outputs


class ONNXRuntimeHandler(BaseHandler):
    def load(self, model_source: str, task_type: str, backend: str, device: str,
             model_revision: str = 'main', load_options: dict | None = None) -> dict:
        if backend != 'onnxruntime' or task_type not in {'tabular-regression', 'tabular-classification'}:
            raise ValueError('unsupported: ONNX Runtime extension accepts dense tabular tasks')
        context = load_session(model_source, task_type, backend, device, model_revision,
                               load_options, runtime=ort)
        inputs, outputs = context['input_specs'], context['output_specs']
        manifest = context['manifest']
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('unsupported ONNX signature: requires one dense input and one tensor output')
        input_spec, output_spec = inputs[0], outputs[0]
        if input_spec.type != 'tensor(float)' or len(input_spec.shape) != 2:
            raise ValueError('ONNX input must be float32 with shape [rows, feature_dim]')
        width = _positive_integer(input_spec.shape[1], 'ONNX feature_dim')
        if manifest.get('feature_dim', width) != width:
            raise ValueError('acprof_model.json feature_dim differs from ONNX input shape')
        if output_spec.type not in {'tensor(float)', 'tensor(double)', 'tensor(int64)', 'tensor(int32)'}:
            raise ValueError('ONNX output must be a dense numeric tensor')
        context.update(feature_dim=width, input_name=input_spec.name, output_name=output_spec.name,
                       input_shape=input_spec.shape, output_signature=output_spec.shape)
        return context

    def preprocess(self, model_ctx: dict, raw_input: dict) -> dict:
        scale = _positive_integer(raw_input.get('input_scale'), 'input_scale')
        batch = _positive_integer(raw_input.get('batch_size', 1), 'batch_size')
        features = _dense_matrix(raw_input.get('features'), model_ctx['feature_dim'], 'features')
        if len(features) != scale * batch:
            raise ValueError('input_scale * batch_size must match actual feature rows')
        fixed = model_ctx['input_shape'][0]
        if isinstance(fixed, int) and fixed > 0 and len(features) != fixed:
            raise ValueError(f'ONNX fixed input shape requires {fixed} rows; received {len(features)}')
        inputs = {model_ctx['input_name']: features}
        validate_inputs(model_ctx, inputs)
        return {'features': features, '_effective_input_scale': float(scale),
                '_truncated_by_limit': False, '_probe_reason': 'ONNX input shape verified',
                '_workload': {'input': {'tensors': tensor_metadata(inputs)}}}

    def predict(self, model_ctx: dict, processed_input: Any) -> Any:
        output = run_session(model_ctx, {model_ctx['input_name']: processed_input['features']})[model_ctx['output_name']]
        if not output.shape or output.shape[0] != len(processed_input['features']):
            raise ValueError('ONNX output rows differ from the actual input rows')
        return output

    def postprocess(self, model_ctx: dict, raw_output: Any) -> dict:
        shape = list(raw_output.shape)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError('ONNX output must have a non-empty result dimension')
        return {'task': model_ctx['task_type'],
                'output_type': 'classification' if model_ctx['task_type'] == 'tabular-classification' else 'regression',
                'output_shape': shape, 'n_results': shape[0]}

    def validate_output(self, model_ctx: dict, raw_input: dict, processed_input: Any,
                        raw_output: Any, response: dict) -> dict:
        result = super().validate_output(model_ctx, raw_input, processed_input, raw_output, response)
        validate_outputs(model_ctx, {model_ctx['output_name']: raw_output})
        validate_artifact(model_ctx)
        declared = model_ctx['output_signature']
        if len(raw_output.shape) != len(declared) or any(
            isinstance(size, int) and size > 0 and raw_output.shape[index] != size
            for index, size in enumerate(declared)
        ):
            raise ValueError('ONNX output shape differs from model signature')
        result['task']['checks'].append('onnx_model_output_signature')
        return result

    def get_scale_metadata(self, model_ctx: dict, raw_input: dict) -> dict:
        fixed = model_ctx['input_shape'][0]
        return {'feature_dim': model_ctx['feature_dim'], 'model_format': 'onnxruntime',
                'fixed_rows': fixed if isinstance(fixed, int) else None}
