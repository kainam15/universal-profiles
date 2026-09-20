"""在 timeseries 镜像中验证真实 ChronosBolt 加载与预测。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in (
    "torch", "chronos", "transformers",
)), "requires the timeseries container")
class TimeseriesRuntimeTests(unittest.TestCase):
    def test_chronos2_native_dispatch_and_batched_forecast_match_upstream(self):
        import torch
        from chronos import BaseChronosPipeline
        from chronos.chronos2 import Chronos2Model
        from acprof.container.handlers.timeseries import ChronosHandler

        torch.set_num_threads(1)
        config = Chronos2Model.config_class(d_model=16, d_kv=8, d_ff=32, num_heads=2, num_layers=1,
                                            dropout_rate=0.)
        config.chronos_config = {"context_length": 32, "output_patch_size": 4, "input_patch_size": 4,
                                "input_patch_stride": 4, "quantiles": [index / 10 for index in range(1, 10)], "use_reg_token": True,
                                "max_output_patches": 2}
        config.chronos_pipeline_class = "Chronos2Pipeline"
        with tempfile.TemporaryDirectory() as directory:
            Chronos2Model(config).save_pretrained(directory)
            handler = ChronosHandler()
            context = handler.load(directory, "time-series-forecasting", "chronos", "cpu")
            self.assertEqual(context["pipeline_type"], "Chronos2Pipeline")
            payload = {"context": [[1., 2., 3., 4.] * 2, [3., 4., 2., 1.] * 2], "prediction_length": 4}
            processed = handler.preprocess(context, payload)
            output = handler.predict(context, processed)
            reference = BaseChronosPipeline.from_pretrained(directory, local_files_only=True, device_map="cpu")
            expected = torch.cat(reference.predict([torch.tensor(row) for row in payload["context"]], prediction_length=4), dim=0)
            torch.testing.assert_close(output, expected)
            self.assertEqual(handler.postprocess(context, output)["forecast_shape"], [2, 9, 4])

    def test_chronos_bolt_offline(self):
        from examples.structured.chronos_smoke import main
        with tempfile.TemporaryDirectory() as directory:
            main(Path(directory))
