"""Prepare pinned small pretrained artifacts, then validate them offline without Torch.

Reference checks compare one sample with ONNX's independent reference evaluator.
They do not establish task accuracy, a performance baseline, or full profiling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = {
    'mnist': {
        'repository': 'onnxmodelzoo/mnist-12',
        'revision': '1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5',
        'license': {'card_metadata': 'apache-2.0', 'card_body': 'MIT'},
        'family': 'cv', 'task': 'image-classification', 'model_file': 'mnist-12.onnx',
        'files': {
            'mnist-12.onnx': '5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd',
            'README.md': 'f6032329d4b8a93d8350ddc7489d9139a606fe3e984e261466afc51737cda4ac',
        },
        'options': {
            'output_name': 'Plus214_Output_0',
            'image_processing': {'input_name': 'Input3', 'layout': 'NCHW', 'mode': 'L',
                                 'rescale_factor': 1 / 255, 'mean': [0.], 'std': [1.],
                                 'resize': {'width': 28, 'height': 28, 'resample': 'bilinear'}},
        },
    },
    'bert-tiny': {
        'repository': 'onnx-community/BERT-tiny-RAID-ONNX',
        'revision': '8f5741a3d45781899100c9a299b25624b5afa914', 'license': {'card_metadata': 'MIT'},
        'family': 'nlp', 'task': 'text-classification', 'model_file': 'onnx/model.onnx',
        'files': {
            'onnx/model.onnx': 'b7b83d30c66fa5aef49fb3019ff294e451983c9a810e9999c8480c25af514d24',
            'tokenizer.json': 'cb374d6bc042c22455946f4e09a89d29882a199fdaf8fb25be00dc8b8857a448',
            'config.json': 'c72b4a25288ad495a2dc582aa836eec7a3ee54dd278a6765cd80bc44135c1036',
            'tokenizer_config.json': 'fe7b9e43fbb955c7a6ad00d597382e484dbb4dc14fb8cdb3f75e859de1e24cf7',
            'README.md': '7f060912ce9065cd36a65e4fc8e60ddfd219ec6e870fb59128eab0b80d9999f1',
        },
        'options': {'output_name': 'logits',
                    'tokenizer': {'file': 'tokenizer.json', 'max_length': 512, 'padding': 'none',
                                  'truncation': 'reject', 'add_special_tokens': True}},
    },
}


def prepare(name: str, directory: Path) -> dict:
    declaration = MODELS[name]
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('preparation directory must be empty; existing artifacts are not overwritten')
    files = {}
    for relative, expected in declaration['files'].items():
        url = f"https://huggingface.co/{declaration['repository']}/resolve/{declaration['revision']}/{relative}"
        with urlopen(url, timeout=60) as response:
            content = response.read(30 * 1024 * 1024 + 1)
        if len(content) > 30 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'artifact size/hash mismatch: {relative}')
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative] = {'sha256': expected, 'bytes': len(content)}
    provenance = {key: declaration[key] for key in ('repository', 'revision', 'license')}
    provenance.update(files=files, conversion={
        'performed_locally': False, 'source': 'published upstream ONNX artifact',
        'tool_version': 'unknown', 'parameters': 'unknown',
    })
    manifest = {'schema_version': 1, 'format': 'onnxruntime', 'task': declaration['task'],
                'model_file': declaration['model_file'],
                'artifact_sha256': declaration['files'][declaration['model_file']],
                'provenance': provenance, **declaration['options']}
    (directory / 'acprof_model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (directory / 'source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def validate(name: str, directory: Path) -> dict:
    declaration = MODELS[name]
    files = {}
    for relative, expected in declaration['files'].items():
        content = (directory / relative).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f'{relative}: prepared file hash differs from the pinned artifact')
        files[relative] = {'sha256': actual, 'bytes': len(content)}
    provenance = {key: declaration[key] for key in ('repository', 'revision', 'license')}
    provenance.update(files=files, conversion={
        'performed_locally': False, 'source': 'published upstream ONNX artifact',
        'tool_version': 'unknown', 'parameters': 'unknown',
    })
    if json.loads((directory / 'source_provenance.json').read_text()) != provenance:
        raise ValueError('preparation provenance differs from the verified pinned files')

    import numpy as np
    import onnx
    import onnxruntime
    from onnx.reference import ReferenceEvaluator
    from acprof.container.handlers import HandlerRegistry
    from acprof.workloads import get_generator

    if any(importlib.util.find_spec(module) is not None for module in ('torch', 'transformers')):
        raise AssertionError('real ONNX checks require an environment without Torch or Transformers installed')
    handler = HandlerRegistry.get(declaration['family'], 'onnxruntime')
    context = handler.load(str(directory), declaration['task'], 'onnxruntime', 'cpu', declaration['revision'])
    generator = get_generator(declaration['family'], declaration['repository'], declaration['task'], 1)
    payload = generator.generate(1.0 if name == 'mnist' else 8)
    processed = handler.preprocess(context, payload)
    output = handler.predict(context, processed)
    response = handler.postprocess(context, output)
    validation = handler.validate_output(context, payload, processed, output, response)
    model = onnx.load(str(directory / declaration['model_file']))
    names = [spec.name for spec in context['output_specs']]
    reference = ReferenceEvaluator(model).run(names, processed['inputs'])
    errors = {}
    for key, expected in zip(names, reference):
        np.testing.assert_allclose(output[key], expected, rtol=1e-4, atol=1e-4)
        errors[key] = float(np.max(np.abs(output[key] - expected)))
    return {
        'successful': True, 'scope': 'pretrained_model_cpu_interface_not_accuracy_or_performance',
        'model': name, 'torch_installed': False, 'transformers_installed': False,
        'runtime_version': onnxruntime.__version__, 'artifact': context['artifact'],
        'runtime_parameters': context['runtime_parameters'], 'response': response,
        'validation': validation, 'source': provenance,
        'reference': {'status': 'verified', 'scope': 'one_sample_independent_onnx_reference_evaluator',
                      'onnx_version': onnx.__version__, 'rtol': 1e-4, 'atol': 1e-4,
                      'maximum_absolute_errors': errors},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate'))
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = prepare(args.model, args.directory) if args.action == 'prepare' else validate(args.model, args.directory)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.report:
        args.report.write_text(encoded)
    else:
        print(encoded, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
