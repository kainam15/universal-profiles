"""Run a tiny randomly initialized ChronosBolt fixture, without model downloads.

    PYTHONPATH=. python examples/structured/chronos_smoke.py /tmp/chronos-fixture

Requires chronos-forecasting and its runtime dependencies. This fixture checks
loading and inference plumbing; it is not an accuracy benchmark or trained model.
"""

import argparse
import json
from pathlib import Path

import torch
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from transformers import T5Config

from acprof.container.handlers.timeseries import ChronosHandler
from acprof.workloads.timeseries import TimeseriesWorkloadGenerator


def main(directory: Path) -> None:
    torch.manual_seed(12345)
    config = T5Config(d_model=16, d_kv=8, d_ff=32, num_heads=2, num_layers=1,
                      num_decoder_layers=1, dropout_rate=0.0, decoder_start_token_id=0)
    config.chronos_config = {"context_length": 32, "prediction_length": 4,
                            "input_patch_size": 4, "input_patch_stride": 4,
                            "quantiles": [0.1, 0.5, 0.9], "use_reg_token": True}
    model = ChronosBoltModelForForecasting(config).eval()
    model.save_pretrained(directory)
    handler = ChronosHandler()
    context = handler.load(str(directory), "time-series-forecasting", "chronos", "cpu")
    assert context["pipeline_type"] == "bolt"
    assert handler.get_scale_metadata(context, {})["max_effective_input_scale"] == 32
    generator = TimeseriesWorkloadGenerator("chronos-fixture", "time-series-forecasting", 2)
    for scale in (8, 16, 32):
        payload = generator.generate(scale)
        payload["prediction_length"] = 4
        processed = handler.preprocess(context, payload)
        output = handler.predict(context, processed)
        result = handler.postprocess(context, output)
        assert result["forecast_shape"] == [2, 3, 4], result
        assert processed["_effective_input_scale"] == scale
        print(json.dumps({"torch_version": torch.__version__, "scale": scale, **result}))
    try:
        handler.preprocess(context, generator.generate(33))
    except ValueError as error:
        print(json.dumps({"over_limit_rejected": str(error)}))
    else:
        raise AssertionError("Chronos context truncation was not rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    main(parser.parse_args().model_dir)
