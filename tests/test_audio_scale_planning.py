import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host import orchestrator
from acprof.host.detect import TaskInfo


RESAMPLING_POLICY = "scipy.signal.resample_poly_if_required_in_preprocess"


class AudioScalePlanningTests(unittest.TestCase):
    def _plan(self, task, constraints, directory):
        info = TaskInfo("example/audio", task, "audio", "transformers_model", "transformers", "fixed", "unit")
        with patch.object(orchestrator, "_start_probe_session", return_value=SimpleNamespace(name="audio-probe")), patch.object(
            orchestrator, "_stop_container_session"
        ), patch.object(orchestrator, "_request_scale_meta", return_value=constraints), patch.object(
            orchestrator, "_post_probe_payload", return_value={"effective_input_scale": 1.0, "truncated_by_limit": False, "reason": "valid waveform"}
        ):
            return orchestrator._plan_audio_scales(
                task_info=info, image_info=orchestrator.ImageInfo("unused"),
                cpu_list=[1], mem_list=[2], gpu_list=["off"], scales=[1.0], batch_size=1,
                output_dir=directory, source="manual", workload_spec_path=None,
            )

    def _constraints(self, rate=24000):
        return {
            "input_scale_type": "duration_s", "required_sampling_rate": rate,
            "source_sampling_rate": 16000, "resampling_policy": RESAMPLING_POLICY,
            "short_form_fixed_padding": False, "model_type": "encodec", "reason": "audio seconds",
        }

    def test_codec_and_classification_accept_declared_resampling_and_persist_source(self):
        for task, rate in (("audio-to-audio", 24000), ("audio-classification", 32000)):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                plan = self._plan(task, self._constraints(rate), directory)
                data = json.loads(Path(plan.plan_file).read_text())
                constraints = data["model_constraints"]
                self.assertEqual(constraints["source_sampling_rate"], 16000)
                self.assertEqual(constraints["required_sampling_rate"], rate)
                self.assertEqual(constraints["resampling_policy"], RESAMPLING_POLICY)
                self.assertEqual(data["entries"][0]["payload"]["sample_rate"], 16000)
                self.assertEqual(data["entries"][0]["input_metadata"]["input_num_samples"], 16000)
                self.assertEqual(plan.scales, [1.0])

    def test_asr_keeps_strict_sampling_rate_even_if_a_resampling_policy_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "sampling rate does not match"):
            self._plan("automatic-speech-recognition", self._constraints(), directory)

    def test_legacy_image_without_resampling_contract_still_rejects_rate_mismatch(self):
        constraints = self._constraints()
        del constraints["resampling_policy"]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "sampling rate does not match"):
            self._plan("audio-to-audio", constraints, directory)

    def test_resampling_contract_must_match_the_actual_source_rate(self):
        constraints = self._constraints()
        constraints["source_sampling_rate"] = 8000
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "sampling rate does not match"):
            self._plan("audio-to-audio", constraints, directory)


if __name__ == "__main__":
    unittest.main()
