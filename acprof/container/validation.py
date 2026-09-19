"""独立于测量窗口的输出协议与任务 sanity validation。"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from numbers import Number
from typing import Any

from acprof.extensions import CATALOG
from acprof.workloads.contract import workload_contract


class OutputValidationError(ValueError):
    """The executed workload does not satisfy its declared protocol."""


def _finite(value: Any, path: str = 'output') -> None:
    if isinstance(value, Number):
        if not math.isfinite(value):
            raise OutputValidationError(f'{path} must be finite')
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f'{path}[{index}]')
    elif callable(getattr(value, 'isfinite', None)):
        if not bool(value.isfinite().all().item()):
            raise OutputValidationError(f'{path} must be finite')
    elif hasattr(value, 'dtype') and hasattr(value, 'shape'):
        import numpy as np
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise OutputValidationError(f'{path} must be finite')
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _finite(getattr(value, field.name), f'{path}.{field.name}')


def _texts(raw_output: Any) -> list[str]:
    if isinstance(raw_output, str):
        return [raw_output]
    if isinstance(raw_output, Mapping):
        found = [value for key, value in raw_output.items()
                 if key in {'generated_text', 'summary_text', 'translation_text', 'text', 'answer'} and isinstance(value, str)]
        found.extend(text for key in ('texts', 'captions', 'answers')
                     for text in _texts(raw_output.get(key)))
        if found:
            return found
    if isinstance(raw_output, (list, tuple)):
        return [text for value in raw_output for text in _texts(value)]
    return []


def validate_output(model_ctx: dict, raw_input: dict, processed_input: Any,
                    raw_output: Any, response: dict) -> dict:
    """Validate actual outputs once before collection; never starts another inference."""
    if not isinstance(response, dict) or not response:
        raise OutputValidationError('postprocess must return a non-empty JSON object')
    try:
        json.dumps(response, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise OutputValidationError(f'output serialization must be finite JSON: {exc}') from exc
    task = model_ctx.get('task_type')
    if not task or response.get('task') != task:
        raise OutputValidationError('output task is missing or differs from the selected task')
    if response.get('error'):
        raise OutputValidationError(f"output contains error: {response['error']}")
    if raw_output is None:
        raise OutputValidationError('raw output is missing')
    family = CATALOG.task_families.get(task)
    if family != 'timeseries' and (not isinstance(response.get('output_type'), str) or not response['output_type']):
        raise OutputValidationError('output_type is required')
    if family in {'nlp', 'cv'}:
        count = response.get('n_results')
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise OutputValidationError('n_results must be a non-negative integer')
    _finite(raw_output)
    prepared = processed_input if isinstance(processed_input, Mapping) else {}
    effective = prepared.get('_effective_input_scale')
    planned = raw_input.get('input_scale', raw_input.get('resolution'))
    if (isinstance(effective, bool) or not isinstance(effective, Number)
            or not math.isfinite(effective) or effective <= 0):
        raise OutputValidationError('actual input scale evidence must be finite and positive')
    if planned is not None and not math.isclose(float(planned), float(effective), abs_tol=1e-6):
        raise OutputValidationError(f'actual input scale differs from plan: {effective} != {planned}')
    if prepared.get('_truncated_by_limit'):
        raise OutputValidationError('workload was truncated by the model limit')
    checks = ['type', 'task', 'finite', 'actual_scale', 'serialization']
    shape_evidence = {'status': 'unavailable', 'detail': 'output protocol does not expose tensor dimensions'}
    shape = next((response[key] for key in ('output_shape', 'forecast_shape', 'depth_shape', 'feature_shape', 'audio_shape')
                  if key in response), None)
    if shape is not None:
        if not isinstance(shape, list) or not shape or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
            raise OutputValidationError('output shape must contain positive integer dimensions')
        actual_shape = getattr(raw_output, 'shape', None)
        if actual_shape is not None and list(actual_shape) != shape:
            raise OutputValidationError('reported output shape differs from actual tensor shape')
        checks.append('shape')
        shape_evidence = {'status': 'verified' if actual_shape is not None else 'available',
                          'detail': 'actual tensor dimensions' if actual_shape is not None else 'postprocess dimensions'}
    task_checks = []
    if task.startswith('tabular-') or task in {'reinforcement-learning', 'robotics', 'graph-ml'}:
        if 'output_shape' not in response:
            raise OutputValidationError('structured output requires output_shape')
        count = response.get('n_results')
        expected = len(raw_input.get('features', raw_input.get('observations', raw_input.get('graphs', []))))
        if count != expected or shape[0] != expected:
            raise OutputValidationError(f'output shape/rows differ from executed input: expected {expected}')
        task_checks.append('one_result_per_input_row_or_graph')
    if task == 'time-series-forecasting':
        if shape is None:
            raise OutputValidationError('forecast_shape is required')
        horizon = raw_input.get('prediction_length', prepared.get('prediction_length'))
        if horizon is not None and shape[-1] != horizon:
            raise OutputValidationError('forecast shape differs from prediction_length')
        task_checks.append('forecast_horizon')
    if task in {'text-generation', 'text2text-generation', 'summarization', 'translation',
                'image-to-text', 'image-text-to-text', 'audio-text-to-text', 'video-text-to-text', 'any-to-any'}:
        texts = _texts(raw_output) or _texts(response)
        if texts and not any(text.strip() for text in texts):
            raise OutputValidationError('text generation sanity check returned empty text')
        actual_tokens = response.get('actual_output_tokens')
        if not texts and (isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens <= 0):
            raise OutputValidationError('generation has no non-empty text or observed output token evidence')
        task_checks.append('non_empty_text' if texts else 'observed_output_tokens')
    if task in {'automatic-speech-recognition', 'automatic-speech-recognition-long-form'}:
        if not isinstance(response.get('text'), str):
            raise OutputValidationError('ASR requires transcript text')
        if not response['text'].strip():
            raise OutputValidationError('ASR sanity check returned an empty transcript')
        task_checks.append('non_empty_transcript')
    if not task_checks and family is not None:
        if not (isinstance(raw_output, (Mapping, list, tuple)) or hasattr(raw_output, 'shape')
                or dataclasses.is_dataclass(raw_output)):
            raise OutputValidationError('task sanity requires actual prediction records or tensors')
        if isinstance(raw_output, Mapping) and not raw_output:
            raise OutputValidationError('task sanity returned empty prediction evidence')
        if isinstance(raw_output, (list, tuple)) and not raw_output and response.get('n_results', 0) != 0:
            raise OutputValidationError('n_results differs from empty actual predictions')
        task_checks.append('prediction_records_or_tensors')
    contract = workload_contract(model_ctx, raw_input, processed_input, response)
    json.dumps(contract, allow_nan=False)
    return {'protocol': {'status': 'verified', 'checks': checks, 'aspects': {'shape': shape_evidence}},
            'task': {'status': 'verified' if task_checks else 'unavailable', 'checks': task_checks,
                     'scope': 'sanity_only_not_accuracy_benchmark'},
            'workload_contract': contract}
