"""确定性离线 ONNX 接口制品；不代表预训练模型质量。"""
from __future__ import annotations

import json


BASIC_SCENARIOS = {
    'tabular': {
        'family': 'structured', 'task': 'tabular-regression', 'batch_size': 2, 'input_scale': 2,
        'output_shape': [4, 1], 'n_results': 4, 'fixture_options': {},
        'expected_workload': {'task': 'tabular-regression', 'batch_size': 2,
                              'input': {'rows': 4, 'feature_dim': 8}, 'output': {'shape': [4, 1]}},
        'reference': {'prediction': [[28.25], [92.25], [156.25], [220.25]]},
    },
    'image': {
        'family': 'cv', 'task': 'image-classification', 'batch_size': 1, 'input_scale': 0.125,
        'output_shape': [1, 2], 'n_results': 1, 'fixture_options': {'fixed_size': 14, 'resize': 14},
        'expected_workload': {'task': 'image-classification', 'batch_size': 1,
                              'input': {'images': {'count': 1, 'original_resolution': [28, 28],
                                                  'processed_resolution': [14, 14],
                                                  'processed_shape': [1, 3, 14, 14]}},
                              'output': {'shape': [1, 2]}},
        'reference': {'scores': [[1 / 3, -1 / 3]]},
    },
    'text': {
        'family': 'nlp', 'task': 'text-classification', 'batch_size': 1, 'input_scale': 2,
        'output_shape': [1, 2], 'n_results': 1, 'fixture_options': {},
        'expected_workload': {'task': 'text-classification', 'batch_size': 1,
                              'input': {'text': {'tokens': 4, 'actual_tokens_per_sample': [4],
                                                 'content_tokens_per_sample': [2], 'padding': 'none',
                                                 'truncation': 'reject'}},
                              'output': {'shape': [1, 2]}},
        'reference': {'logits': [[14., -14.]], 'token_count': [[4]]},
    },
}


def create_image_fixture(root, *, fixed_size=None, resize=None):
    import onnx
    from onnx import TensorProto, helper
    root.mkdir(parents=True, exist_ok=True)
    size = [fixed_size, fixed_size] if fixed_size else ['height', 'width']
    graph = helper.make_graph([
        helper.make_node('ReduceMean', ['images'], ['mean'], axes=[1, 2, 3], keepdims=0),
        helper.make_node('Unsqueeze', ['mean', 'axis'], ['positive']),
        helper.make_node('Neg', ['positive'], ['negative']),
        helper.make_node('Concat', ['positive', 'negative'], ['scores'], axis=1),
    ], 'image-mean', [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['batch', 3, *size])],
        [helper.make_tensor_value_info('scores', TensorProto.FLOAT, ['batch', 2])],
        [helper.make_tensor('axis', TensorProto.INT64, [1], [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, root / 'model.onnx')
    processing = {'input_name': 'images', 'layout': 'NCHW', 'mode': 'RGB',
                  'rescale_factor': 1 / 255, 'mean': [0, 0, 0], 'std': [1, 1, 1]}
    if resize:
        processing['resize'] = {'width': resize, 'height': resize, 'resample': 'nearest'}
    (root / 'acprof_model.json').write_text(json.dumps({
        'schema_version': 1, 'format': 'onnxruntime', 'task': 'image-classification',
        'model_file': 'model.onnx', 'output_name': 'scores', 'image_processing': processing,
    }))
    return root

def create_text_fixture(root, *, max_length=16, fixed_length=None):
    import onnx
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.WordLevel(
        {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'hello': 4, 'world': 5},
        unk_token='[UNK]',
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)],
    )
    tokenizer.save(str(root / 'tokenizer.json'))
    shape = ['batch', fixed_length or 'sequence']
    graph = helper.make_graph([
        helper.make_node('Mul', ['input_ids', 'attention_mask'], ['masked_ids']),
        helper.make_node('Add', ['masked_ids', 'token_type_ids'], ['combined']),
        helper.make_node('Cast', ['combined'], ['floats'], to=TensorProto.FLOAT),
        helper.make_node('ReduceSum', ['floats', 'axis'], ['positive'], keepdims=1),
        helper.make_node('Neg', ['positive'], ['negative']),
        helper.make_node('Concat', ['positive', 'negative'], ['logits'], axis=1),
        helper.make_node('ReduceSum', ['attention_mask', 'axis'], ['token_count'], keepdims=1),
    ], 'token-sum', [helper.make_tensor_value_info(name, TensorProto.INT64, shape)
                    for name in ('input_ids', 'attention_mask', 'token_type_ids')],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 2]),
         helper.make_tensor_value_info('token_count', TensorProto.INT64, ['batch', 1])],
        [helper.make_tensor('axis', TensorProto.INT64, [1], [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, root / 'model.onnx')
    (root / 'acprof_model.json').write_text(json.dumps({
        'schema_version': 1, 'format': 'onnxruntime', 'task': 'text-classification',
        'model_file': 'model.onnx', 'output_name': 'logits',
        'tokenizer': {'file': 'tokenizer.json', 'max_length': max_length,
                      'padding': 'none', 'truncation': 'reject', 'add_special_tokens': True},
    }))
    return root


def _red_image_payload():
    import base64
    import io
    from PIL import Image

    stream = io.BytesIO()
    Image.new('RGB', (28, 28), (255, 0, 0)).save(stream, format='PNG')
    return {'image_base64': base64.b64encode(stream.getvalue()).decode(),
            'input_scale': 0.125, 'batch_size': 1}


def prepare_basic_scenario(name, directory):
    """Prepare and independently check fixture values before any measurement starts."""
    import hashlib
    import importlib.util
    import numpy as np
    import onnx
    from acprof.artifacts import atomic_write_json
    from acprof.container.handlers import HandlerRegistry
    from acprof.container.runtime_validate import validate
    from examples.onnxruntime.smoke import create_linear_fixture

    if any(importlib.util.find_spec(module) is not None for module in ('torch', 'transformers')):
        raise AssertionError('basic ONNX fixture preparation requires Torch and Transformers to be absent')
    specification = BASIC_SCENARIOS[name]
    creators = {'tabular': create_linear_fixture, 'image': create_image_fixture, 'text': create_text_fixture}
    payloads = {
        'tabular': lambda: {'input_scale': 2, 'batch_size': 2,
                           'features': [[float(i + offset) for i in range(8)] for offset in (0, 8, 16, 24)]},
        'image': _red_image_payload,
        'text': lambda: {'text': 'hello world', 'input_scale': 2, 'batch_size': 1},
    }
    root = creators[name](directory / 'model', **specification['fixture_options'])
    payload = payloads[name]()
    handler = HandlerRegistry.get(specification['family'], 'onnxruntime')
    context = handler.load(str(root), specification['task'], 'onnxruntime', 'cpu')
    raw = handler.predict(context, handler.preprocess(context, payload))
    named_output = raw if isinstance(raw, dict) else {'prediction': raw}
    for key, expected in specification['reference'].items():
        np.testing.assert_allclose(named_output[key], expected, rtol=0, atol=1e-6)
    validation = validate(payload)
    artifact = root / 'model.onnx'
    model = onnx.load(str(artifact))
    report = {
        'reference': {'known_reference_passed': True, 'scope': 'fixture_analytical_values',
                      'checked_outputs': list(specification['reference'])},
        'runtime_validation': validation, 'artifact_kind': 'synthetic_fixture',
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'opset': model.opset_import[0].version, 'ir_version': model.ir_version,
        'preparation': {'generator': 'onnx.helper', 'version': onnx.__version__, 'source': 'fixture-v1'},
        'torch_installed': False, 'transformers_installed': False,
    }
    atomic_write_json(directory / 'payload.json', payload)
    atomic_write_json(directory / 'input_scale_plan.json', {
        'schema_version': 2, 'entries': [{'input_scale': specification['input_scale'],
                                        'scale_label': name, 'payload': payload}],
    })
    atomic_write_json(directory / 'output_validation.json', report)
    return report
