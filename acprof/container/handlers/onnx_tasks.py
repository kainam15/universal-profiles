"""Explicit image and text classification adapters using the shared CPU ORT session."""
from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np

from acprof.container.handlers import BaseHandler, InputLimitError
from acprof.container.handlers.structured import _artifact_path, _positive_integer
from acprof.container.onnx_session import load_session, run_session, tensor_metadata, validate_artifact, validate_inputs, validate_outputs


def _config(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f'{name} must be an explicit object with supported keys: {sorted(allowed)}')
    return dict(value)


class _ClassificationHandler(BaseHandler):
    output_type = 'classification'

    def _load(self, model_source, task_type, backend, device, model_revision, load_options):
        context = load_session(model_source, task_type, backend, device, model_revision, load_options)
        outputs = {spec.name: spec for spec in context['output_specs']}
        selected = context['manifest'].get('output_name')
        if selected is None and len(outputs) == 1:
            selected = next(iter(outputs))
        if selected not in outputs:
            raise ValueError('classification requires explicit output_name selecting a named ONNX output')
        spec = outputs[selected]
        if spec.type != 'tensor(float)' or len(spec.shape) != 2:
            raise ValueError('classification output must be float32 [batch, classes_or_scores]')
        context['output_name'] = selected
        return context

    def predict(self, model_ctx: dict, processed_input: dict) -> dict:
        outputs = run_session(model_ctx, processed_input['inputs'])
        scores = outputs[model_ctx['output_name']]
        batch = next(iter(processed_input['inputs'].values())).shape[0]
        if scores.ndim != 2 or scores.shape[0] != batch or scores.shape[1] < 1:
            raise ValueError('classification output shape must contain one non-empty score row per actual input')
        return outputs

    def postprocess(self, model_ctx: dict, raw_output: dict) -> dict:
        scores = raw_output[model_ctx['output_name']]
        response = {'task': model_ctx['task_type'], 'output_type': self.output_type,
                    'output_shape': list(scores.shape), 'n_results': int(scores.shape[0]),
                    'score_interpretation': 'single_score' if scores.shape[1] == 1 else 'class_scores'}
        # A single score does not establish a threshold, label mapping, or probability.
        if scores.shape[1] > 1:
            response['predicted_classes'] = scores.argmax(axis=1).tolist()
        return response

    def validate_output(self, model_ctx: dict, raw_input: dict, processed_input: dict,
                        raw_output: dict, response: dict) -> dict:
        validate_outputs(model_ctx, raw_output)
        scores = raw_output[model_ctx['output_name']]
        if response.get('output_shape') != list(scores.shape):
            raise ValueError('reported classification output shape differs from actual selected output')
        result = super().validate_output(model_ctx, raw_input, processed_input, raw_output, response)
        batch = next(iter(processed_input['inputs'].values())).shape[0]
        if scores.shape[0] != batch or response.get('n_results') != batch:
            raise ValueError('classification result count differs from actual input batch')
        result['protocol']['aspects']['shape'] = {'status': 'verified', 'detail': 'actual named ONNX output dimensions'}
        result['task']['checks'].extend(['onnx_model_output_signature', 'one_score_row_per_input'])
        validate_artifact(model_ctx)
        return result


class ONNXImageClassificationHandler(_ClassificationHandler):
    def load(self, model_source: str, task_type: str, backend: str, device: str,
             model_revision: str = 'main', load_options: dict | None = None) -> dict:
        if task_type != 'image-classification':
            raise ValueError('unsupported: ONNX image adapter supports image-classification')
        context = self._load(model_source, task_type, backend, device, model_revision, load_options)
        config = _config(context['manifest'].get('image_processing'), 'image_processing',
                         {'input_name', 'layout', 'mode', 'rescale_factor', 'mean', 'std', 'resize'})
        if config.get('layout') != 'NCHW' or config.get('mode') not in {'RGB', 'L'}:
            raise ValueError('image_processing requires layout=NCHW and mode=RGB or L')
        inputs = context['input_specs']
        if len(inputs) != 1 or inputs[0].name != config.get('input_name'):
            raise ValueError('image_processing input_name must select the single image tensor')
        if inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
            raise ValueError('image classification input must be float32 NCHW')
        channels = 3 if config['mode'] == 'RGB' else 1
        for name, default in [('mean', [0.] * channels), ('std', [1.] * channels)]:
            values = config.get(name, default)
            if (not isinstance(values, list) or len(values) != channels
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or (name == 'std' and v <= 0) for v in values)):
                raise ValueError(f'image_processing {name} must have {channels} finite channel values')
            config[name] = values
        factor = config.get('rescale_factor')
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
            raise ValueError('image_processing rescale_factor must be explicit, finite and positive')
        if 'resize' in config:
            resize = _config(config['resize'], 'image_processing resize', {'width', 'height', 'resample'})
            resize['width'] = _positive_integer(resize.get('width'), 'resize width')
            resize['height'] = _positive_integer(resize.get('height'), 'resize height')
            if resize.get('resample') not in {'nearest', 'bilinear'}:
                raise ValueError('image_processing resize resample must be nearest or bilinear')
            config['resize'] = resize
        context['image_processing'] = config
        context['runtime_parameters']['image_processing'] = config
        return context

    def preprocess(self, model_ctx: dict, raw_input: dict) -> dict:
        from PIL import Image

        if raw_input.get('params'):
            raise ValueError('unsupported image-classification params')
        if _positive_integer(raw_input.get('batch_size', 1), 'batch_size') != 1:
            raise ValueError('ONNX image-classification requires batch_size=1; image replication is unsupported')
        encoded = raw_input.get('image_base64')
        if not isinstance(encoded, str) or not encoded:
            raise ValueError('image_base64 must contain one image')
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as source:
            original_size = list(source.size)
            image = source.convert(model_ctx['image_processing']['mode'])
        scale = raw_input.get('input_scale', original_size[0] / 224)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('input_scale must be a finite positive resolution multiplier')
        if 'input_scale' in raw_input and original_size != [max(1, int(224 * scale))] * 2:
            raise ValueError('input_scale does not match actual image resolution')
        config = model_ctx['image_processing']
        if 'resize' in config:
            resize = config['resize']
            resample = Image.Resampling.NEAREST if resize['resample'] == 'nearest' else Image.Resampling.BILINEAR
            image = image.resize((resize['width'], resize['height']), resample)
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        pixels = ((pixels * np.float32(config['rescale_factor']) - np.asarray(config['mean'], dtype=np.float32))
                  / np.asarray(config['std'], dtype=np.float32))
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        inputs = {config['input_name']: tensor}
        validate_inputs(model_ctx, inputs)
        return {'inputs': inputs, 'image': image, '_effective_input_scale': float(scale),
                '_truncated_by_limit': False, '_original_resolution': original_size,
                '_probe_reason': 'raw resolution verified; explicit ONNX image processing applied',
                '_workload': {'input': {'tensors': tensor_metadata(inputs),
                    'images': {'count': 1, 'original_resolution': original_size,
                    'processed_resolution': list(image.size), 'processed_shape': list(tensor.shape),
                    'processed_resolution_status': 'available', 'mode': config['mode'], 'layout': 'NCHW'}}}}


class ONNXTextClassificationHandler(_ClassificationHandler):
    output_type = 'label'

    def load(self, model_source: str, task_type: str, backend: str, device: str,
             model_revision: str = 'main', load_options: dict | None = None) -> dict:
        from tokenizers import Tokenizer

        if task_type != 'text-classification':
            raise ValueError('unsupported: ONNX text adapter supports text-classification')
        context = self._load(model_source, task_type, backend, device, model_revision, load_options)
        config = _config(context['manifest'].get('tokenizer'), 'tokenizer',
                         {'file', 'max_length', 'padding', 'truncation', 'add_special_tokens'})
        if config.get('padding') != 'none' or config.get('truncation') != 'reject':
            raise ValueError('tokenizer requires explicit padding=none and truncation=reject')
        if not isinstance(config.get('add_special_tokens'), bool):
            raise ValueError('tokenizer add_special_tokens must be explicit boolean')
        config['max_length'] = _positive_integer(config.get('max_length'), 'tokenizer max_length')
        tokenizer = Tokenizer.from_file(str(_artifact_path(context['model_root'], config.get('file'))))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        names = {spec.name for spec in context['input_specs']}
        if 'input_ids' not in names or names - {'input_ids', 'attention_mask', 'token_type_ids'}:
            raise ValueError('unsupported text named inputs; expected input_ids, optional attention_mask/token_type_ids')
        if any(spec.type != 'tensor(int64)' or len(spec.shape) != 2 for spec in context['input_specs']):
            raise ValueError('ONNX text inputs must be int64 [batch, sequence]')
        context.update(tokenizer=tokenizer, tokenizer_config=config)
        context['runtime_parameters']['tokenizer'] = config
        return context

    def preprocess(self, model_ctx: dict, raw_input: dict) -> dict:
        if raw_input.get('params'):
            raise ValueError('unsupported text-classification params')
        batch = _positive_integer(raw_input.get('batch_size', 1), 'batch_size')
        texts = raw_input.get('text')
        if isinstance(texts, str):
            if batch != 1:
                raise ValueError('scalar text requires batch_size=1; automatic sample replication is unsupported')
            texts = [texts]
        if (not isinstance(texts, list) or len(texts) != batch
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError('text must contain exactly batch_size non-empty strings')
        config = model_ctx['tokenizer_config']
        encodings = model_ctx['tokenizer'].encode_batch(texts, add_special_tokens=config['add_special_tokens'])
        counts = [len(item.ids) for item in encodings]
        if len(set(counts)) != 1:
            raise ValueError('different text lengths require padding, which is unsupported by this task contract')
        content_counts = [sum(not special for special in item.special_tokens_mask) for item in encodings]
        if any(count <= 0 for count in content_counts):
            raise ValueError('text requires at least one non-special input token')
        if any(count > config['max_length'] for count in counts):
            raise InputLimitError('tokenized input exceeds tokenizer max_length; truncation is rejected',
                                  effective_input_scale=max(content_counts))
        if 'input_scale' in raw_input:
            planned = raw_input['input_scale']
            if (isinstance(planned, bool) or not isinstance(planned, (int, float))
                    or not math.isfinite(planned) or planned != max(content_counts)):
                raise ValueError('input_scale does not match actual non-special input tokens')
        values = {'input_ids': [item.ids for item in encodings],
                  'attention_mask': [item.attention_mask for item in encodings],
                  'token_type_ids': [item.type_ids for item in encodings]}
        inputs = {spec.name: np.asarray(values[spec.name], dtype=np.int64) for spec in model_ctx['input_specs']}
        validate_inputs(model_ctx, inputs)
        return {'inputs': inputs, '_effective_input_scale': float(max(content_counts)),
                '_actual_input_tokens': sum(counts), '_input_token_scope': 'model_input_including_special_tokens',
                '_truncated_by_limit': False, '_probe_reason': 'all tokens preserved; padding disabled',
                '_workload': {'input': {'tensors': tensor_metadata(inputs),
                    'text': {'tokens': sum(counts), 'actual_tokens_per_sample': counts,
                    'content_tokens_per_sample': content_counts, 'padding': 'none', 'truncation': 'reject',
                    'add_special_tokens': config['add_special_tokens'], 'max_length': config['max_length']}}}}

    def get_scale_metadata(self, model_ctx: dict, raw_input: dict) -> dict:
        config = model_ctx['tokenizer_config']
        special = model_ctx['tokenizer'].num_special_tokens_to_add(False) if config['add_special_tokens'] else 0
        return {'input_scale_type': 'seq_length', 'max_effective_input_scale': config['max_length'] - special,
                'reason': 'tokenizer max_length includes special tokens; truncation and padding disabled'}
