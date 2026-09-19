"""离线创建已知数值的小图，验证无 Torch ONNX Handler 的完整接口。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


def create_linear_fixture(directory: Path, *, fixed_rows: int | None = None) -> Path:
    """y = sum(x) + 0.25；模型由 ONNX API 创建，无导出框架或 Hub 下载。"""
    import onnx
    from onnx import TensorProto, helper

    directory.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['features', 'weight'], ['linear']),
         helper.make_node('Add', ['linear', 'bias'], ['prediction'])],
        'acprof-linear-sanity',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, [fixed_rows or 'rows', 8])],
        [helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [fixed_rows or 'rows', 1])],
        [helper.make_tensor('weight', TensorProto.FLOAT, [8, 1], [1.0] * 8),
         helper.make_tensor('bias', TensorProto.FLOAT, [1], [0.25])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, directory / 'model.onnx')
    (directory / 'acprof_model.json').write_text(json.dumps({
        'schema_version': 1, 'task': 'tabular-regression', 'format': 'onnxruntime',
        'model_file': 'model.onnx', 'feature_dim': 8,
    }) + '\n')
    return directory


def exercise(directory: Path) -> dict:
    import numpy as np
    import onnxruntime
    from acprof.container.handlers import HandlerRegistry

    if importlib.util.find_spec('torch') is not None:
        raise AssertionError('This smoke test must run in an environment without Torch installed')
    handler = HandlerRegistry.get('structured', 'onnxruntime')
    context = handler.load(str(directory), 'tabular-regression', 'onnxruntime', 'cpu')
    payload = {'input_scale': 2, 'batch_size': 2,
               'features': [[float(i + offset) for i in range(8)] for offset in (0, 8, 16, 24)]}
    processed = handler.preprocess(context, payload)
    raw = handler.predict(context, processed)
    response = handler.postprocess(context, raw)
    validation = handler.validate_output(context, payload, processed, raw, response)
    np.testing.assert_allclose(raw, [[28.25], [92.25], [156.25], [220.25]], rtol=0, atol=1e-6)
    assert validation['protocol']['status'] == 'verified'
    assert validation['task']['status'] == 'verified'
    assert validation['workload_contract']['input']['rows'] == 4
    assert validation['workload_contract']['input']['feature_dim'] == 8
    return {'successful': True, 'runtime_version': onnxruntime.__version__, 'torch_installed': False,
            'known_reference_passed': True, 'response': response, 'validation': validation}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='acprof-onnx-smoke-') as temporary:
        report = exercise(create_linear_fixture(Path(temporary)))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
