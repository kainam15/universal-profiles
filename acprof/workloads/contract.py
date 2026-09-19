"""实际请求工作量摘要；不重新分词、解码媒体或触发推理。"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, 'shape', None)
    return [int(item) for item in shape] if shape is not None else None


def summarize_workload_contracts(contracts: list[dict | None]) -> dict:
    """Count identical facts after the window; preserve unknown and variable outputs.

    Fast models can execute thousands of requests per row. Repeating the full
    JSON for each one exceeds ordinary CSV field limits without adding evidence.
    """
    variants: dict[str, dict] = {}
    for contract in contracts:
        key = json.dumps(contract, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if key not in variants:
            variants[key] = {'count': 0, 'contract': contract}
        variants[key]['count'] += 1
    return {'schema_version': 1, 'request_count': len(contracts), 'variants': list(variants.values())}


def workload_contract(model_ctx: dict, raw_input: dict, processed: Any, response: dict) -> dict:
    """Keep requested limits separate from observed counts; unknown is JSON null."""
    prepared = processed if isinstance(processed, Mapping) else {}
    task = model_ctx.get('task_type', response.get('task', ''))
    params = raw_input.get('params') or {}
    samples = raw_input.get('samples')
    sample = samples[0] if isinstance(samples, list) and samples and isinstance(samples[0], dict) else raw_input
    contract: dict[str, Any] = {
        'schema_version': 1, 'task': task,
        'batch_size': raw_input.get('batch_size', 1),
        'scenario': {'type': 'serial'},
        'input': {'planned_scale': raw_input.get('input_scale', raw_input.get('resolution')),
                  'actual_scale': prepared.get('_effective_input_scale')},
        'output': {'type': response.get('output_type'),
                   'shape': response.get('output_shape', response.get('forecast_shape')),
                   'count': response.get('n_results')},
    }
    inputs = contract['input']
    for key in ('features', 'observations'):
        if isinstance(raw_input.get(key), list):
            rows = raw_input[key]
            inputs.update(rows=len(rows), feature_dim=len(rows[0]) if rows else 0)
    if isinstance(raw_input.get('graphs'), list):
        graphs = raw_input['graphs']
        inputs['graphs'] = {'count': len(graphs),
                            'nodes': sum(len(g.get('node_features', [])) for g in graphs)}
    if 'text' in sample or task in {'text-generation', 'text2text-generation', 'summarization', 'translation'}:
        actual_input = response.get('actual_input_tokens')
        inputs['text'] = {'tokens': actual_input if actual_input is not None else prepared.get('_actual_input_tokens',
                                                prepared.get('prompt_length', prepared.get('_effective_input_scale'))),
                          'token_scope': 'model_input_including_special_tokens' if actual_input is not None else
                                         prepared.get('_input_token_scope', 'effective_scale_excludes_special_tokens')}
    if task in {'text-generation', 'text2text-generation', 'summarization', 'translation',
                'image-to-text', 'image-text-to-text', 'audio-text-to-text', 'video-text-to-text', 'any-to-any'}:
        actual = response.get('actual_output_tokens')
        contract['generation'] = {
            'max_output_tokens': params.get('max_new_tokens'),
            'actual_output_tokens': actual,
            'actual_output_tokens_per_sequence': response.get('actual_output_tokens_per_sequence'),
            'actual_output_tokens_status': 'available' if actual is not None else 'unavailable',
            'stop_reason': response.get('stop_reason'),
            'decoded_text_token_count': response.get('output_token_count'),
        }
    if 'audio' in prepared or 'audio_base64' in raw_input:
        audio = prepared.get('audio')
        rate = prepared.get('sample_rate', raw_input.get('sample_rate'))
        samples = len(audio) if audio is not None and hasattr(audio, '__len__') else None
        duration = prepared.get('_duration_s')
        inputs['audio'] = {
            'audio_seconds': duration, 'sample_rate': rate,
            'channels': prepared.get('_channels', 1 if audio is not None else None),
            'processor_input_duration': samples / rate if samples is not None and rate else None,
            'processed_duration': None,
            'processed_duration_status': 'unavailable',
            'source_sample_rate': prepared.get('_source_sample_rate', raw_input.get('sample_rate')),
        }
        if 'text' in response:
            contract['output']['transcript_non_empty'] = bool(response['text'].strip())
    image = prepared.get('image')
    tensor_inputs = prepared.get('inputs')
    pixels = tensor_inputs.get('pixel_values') if isinstance(tensor_inputs, Mapping) else None
    pixel_shape = _shape(pixels)
    if image is not None or 'image_base64' in raw_input or 'frames_base64' in raw_input:
        resolution = list(image.size) if image is not None and hasattr(image, 'size') else None
        inputs['images'] = {
            'count': len(raw_input.get('frames_base64', [])) or 1,
            'original_resolution': prepared.get('_original_resolution', resolution),
            'processed_resolution': pixel_shape[-2:][::-1] if pixel_shape and len(pixel_shape) >= 3 else None,
            'processed_shape': pixel_shape,
            'processed_resolution_status': 'available' if pixel_shape else 'unavailable',
        }
    if 'num_inference_steps' in params or 'resolution' in prepared:
        generation = contract.setdefault('generation', {})
        generation.update(width=response.get('image_width', response.get('video_width')),
                          height=response.get('image_height', response.get('video_height')),
                          steps=params.get('num_inference_steps'),
                          frames=response.get('video_frame_count'))
    # A handler may provide measured modality facts; this is data, not a second adapter.
    for section, facts in prepared.get('_workload', {}).items():
        if isinstance(facts, dict):
            target = contract.setdefault(section, {})
            for name, value in facts.items():
                if isinstance(value, dict) and isinstance(target.get(name), dict):
                    target[name].update(value)
                else:
                    target[name] = value
    return contract
