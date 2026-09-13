"""旧参数和旧产物必须在执行或写入前明确失败。"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.cli.run_args import build_parser
from acprof.container.handlers import HandlerRegistry, resolve_model_source
from acprof.host.compute_profile import _resolve_ncu_metrics, _select_ncu_flop_metrics
from acprof.host.profiler_common import _load_input_scale_plan_entries
from acprof.host.runtime_images import runtime_fingerprint
from acprof.host.runtime_validation import validate_runtime
from acprof.host.static_metadata import enrich_static_meta_from_input_plan
from acprof.packet.merge_packet_latency import _request_records
from acprof.plotting.data import prepare_df, read_static_meta
from acprof.runtime_profiles import RuntimeProfile
from acprof.tui.settings import load_settings


class CurrentContractTests(unittest.TestCase):
    def test_retired_tui_imports_fail(self):
        for suffix in ("core", "settings", "i18n", "input", "log", "scrollbar", "themes"):
            with self.subTest(suffix=suffix), self.assertRaises(ModuleNotFoundError):
                importlib.import_module("acprof.cli.tui_" + suffix)

    def test_current_version_with_retired_settings_field_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tui.json"
            path.write_text(json.dumps({"version": 4, "run_defaults": {"allow_cgroup_v1": False}}))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "allow_cgroup_v1"):
                load_settings(path, Path(temporary))
            self.assertEqual(path.read_bytes(), original)

    def test_missing_baked_snapshot_does_not_fall_back_to_hub(self):
        with self.assertRaisesRegex(FileNotFoundError, "snapshot"):
            resolve_model_source("example/model", "/missing/acprof/model-snapshot")

    def test_unknown_backend_does_not_choose_another_family_handler(self):
        with patch.dict(HandlerRegistry._handlers, {"test:current": object()}, clear=True):
            with self.assertRaisesRegex(ValueError, "backend"):
                HandlerRegistry.get("test", "retired")

    def test_old_ncu_counters_are_not_selected(self):
        self.assertEqual(_select_ncu_flop_metrics(["flop_count_sp", "flop_count_dp"]), [])

    def test_failed_ncu_query_does_not_guess_a_metric_list(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="query unavailable")
        with patch("acprof.host.compute_profile._run", return_value=failed):
            metrics, error = _resolve_ncu_metrics("ncu")
        self.assertEqual(metrics, [])
        self.assertIn("query unavailable", error)

    def test_removed_cli_options_fail_during_argument_parsing(self):
        for arguments in (["--no-compute-profile"], ["--compute-profile-tool", "auto"],
                          ["--allow-cgroup-v1"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    build_parser().parse_args(["--model", "example/model", *arguments])
                self.assertEqual(caught.exception.code, 2)

    def test_old_settings_fail_without_overwriting_the_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tui.json"
            for payload in ({}, {"version": 1}, {"version": 2}, {"version": 3}):
                path.write_text(json.dumps(payload))
                original = path.read_bytes()
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "version|版本"):
                    load_settings(path, root)
                self.assertEqual(path.read_bytes(), original)

    def test_old_input_plan_is_rejected_and_current_payload_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input_scale_plan.json"
            entry = {"input_scale": 1, "payload": {"text": "exact input"}, "input_metadata": {}}
            for version in (None, 1, True, 99):
                payload = {"entries": [entry]}
                if version is not None:
                    payload["schema_version"] = version
                path.write_text(json.dumps(payload))
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema_version"):
                    _load_input_scale_plan_entries(str(path))
            path.write_text(json.dumps({"schema_version": 2, "entries": [entry]}))
            self.assertEqual(_load_input_scale_plan_entries(str(path))[0]["payload"], entry["payload"])

    def test_unlocked_runtime_is_rejected_before_building(self):
        with self.assertRaisesRegex(ValueError, "锁|lock"):
            runtime_fingerprint(RuntimeProfile("unlocked", "nlp"))

    def test_unmanaged_image_cannot_skip_runtime_validation(self):
        with patch("subprocess.run") as run, self.assertRaisesRegex(ValueError, "runtime_environment|运行环境"):
            validate_runtime(task_info=None, image_info=SimpleNamespace(runtime_environment={}),
                             planned=None, cpu_list=[1], mem_list=[4], gpu_list=["off"], output_dir="unused")
        run.assert_not_called()

    def test_static_metadata_stand_in_is_rejected(self):
        with self.assertRaises(TypeError):
            enrich_static_meta_from_input_plan(object(), SimpleNamespace(workload={}, plan_sha256=""))

    def test_packet_flat_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version|schema v2"):
            _request_records({"request-1": 0.25})
        self.assertEqual(_request_records({"schema_version": 2, "requests": {
            "request-1": {"latency_s": 0.25}}}), {"request-1": {"latency_s": 0.25}})

    def test_legacy_csv_fields_and_static_csv_fail_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result_all.csv"
            original = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,status,warmup,energy_eff_j\n1,4,on,1,ok,0,2\n"
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, "energy_eff_j"):
                prepare_df(str(path))
            self.assertEqual(path.read_text(), original)
            (path.parent / "static_meta.csv").write_text("batch_size\n1\n")
            with self.assertRaisesRegex(ValueError, "static_meta.csv"):
                read_static_meta(str(path))


if __name__ == "__main__":
    unittest.main()
