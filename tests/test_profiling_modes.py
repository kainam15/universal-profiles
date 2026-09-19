import csv
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from acprof.host import client, orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import ImageInfo
from acprof.host.docker_runtime import RunningContainer


class ProfilingModeTests(unittest.TestCase):
    def test_basic_refuses_missing_required_resource_collector(self):
        with tempfile.TemporaryDirectory() as root, patch.object(client, "OUT_CSV", str(Path(root) / "result.csv")), patch.object(client, "PROFILING_MODE", "basic"), patch.object(client, "resource_usage_mod", None), patch.object(
            client, "input_scale_entries", [{"input_scale": 1.0, "scale_label": "one", "payload": {}}]
        ), patch.object(client.requests, "get", side_effect=AssertionError("must fail before server request")):
            with self.assertRaisesRegex(RuntimeError, "CPU.*memory"):
                client.main()

    def test_basic_matrix_preserves_resource_sweep_and_skips_capture(self):
        task = TaskInfo("test/model", "fill-mask", "nlp", "transformers_pipeline", "transformers", "main", "unit")
        captured = []
        output = io.StringIO()
        def fake_run(command, **kwargs):
            if command[-2:] == ["-m", "acprof.host.client"]:
                captured.append(kwargs["env"])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            stack.enter_context(patch.object(orchestrator, "_start_container_session", return_value=RunningContainer("test", "http://localhost", 8000, 0.1)))
            stack.enter_context(patch.object(orchestrator, "_stop_container_session"))
            stack.enter_context(patch.object(orchestrator, "_run", side_effect=fake_run))
            stack.enter_context(patch.object(orchestrator, "_resolve_packet_latency_runtime", side_effect=AssertionError("basic must not resolve packet capture")))
            stack.enter_context(patch.object(orchestrator, "_check_case_cpu_idle_power_stable", side_effect=AssertionError("basic must not require RAPL")))
            stack.enter_context(redirect_stdout(output))
            paths = orchestrator.run_matrix(task, ImageInfo(tag="test"), [1, 2], [1], ["off"], root, ".", profiling_mode="basic", input_scales="1")
        self.assertEqual(len(paths), 2)
        self.assertEqual([env["CPU_CORES"] for env in captured], ["1", "2"])
        self.assertTrue(all(env["USE_MIPS"] == "0" and env["PROFILING_MODE"] == "basic" for env in captured))
        self.assertIn("packet_latency=not_requested (profiling_mode=basic)", output.getvalue())
        self.assertNotIn("tcpdump/tshark unavailable", output.getvalue())

    def test_resume_legacy_options_mean_full_and_reject_basic(self):
        from acprof.host.run_state import RunState, RunStateError
        with tempfile.TemporaryDirectory() as root, patch("acprof.host.run_state.host_identity", return_value={}):
            state = RunState(root, {"model": "test"}, resume=False, project_dir=".")
            state.close()
            state = RunState(root, {"model": "test", "profiling_mode": "full"}, resume=True, project_dir=".")
            state.close()
            with self.assertRaisesRegex(RunStateError, "参数"):
                RunState(root, {"model": "test", "profiling_mode": "basic"}, resume=True, project_dir=".")

    def test_basic_client_skips_energy_and_perf_but_preserves_app_latency(self):
        events = []
        class ResourceMonitor:
            def __init__(self, **kwargs):
                pass
            def start(self):
                events.append("resource_start")
            def stop(self):
                from acprof.monitors.resource_usage import _nan_result
                events.append("resource_stop")
                result = _nan_result()
                result.container_cpu_util_avg_pct = 25.0
                result.container_mem_usage_avg_bytes = 1024.0
                return result, "", []
            def close(self):
                events.append("resource_close")
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            path = Path(root) / "case.csv"
            settings = {
                "OUT_CSV": str(path), "PROFILING_MODE": "basic", "WARMUP": 0,
                "REPEAT": 1, "REPEAT_IN_WINDOW": 1, "USE_ENERGY": True,
                "GPU_MODE": "on", "USE_MIPS": False, "BATCH_SIZE": 1,
                "resource_usage_mod": SimpleNamespace(ResourceUsageMonitor=ResourceMonitor),
                "energy_mod": SimpleNamespace(GPUEnergyMonitor=Mock(side_effect=AssertionError("basic must not start NVML energy"))),
                "cpu_energy_mod": SimpleNamespace(CPUEnergyMonitor=Mock(side_effect=AssertionError("basic must not start RAPL"))),
                "input_scale_entries": [{"input_scale": 1.0, "scale_label": "one", "payload": {}}],
            }
            for name, value in settings.items():
                stack.enter_context(patch.object(client, name, value, create=True))
            stack.enter_context(patch.object(client.requests, "get", return_value=SimpleNamespace(status_code=200, text="ok")))
            stack.enter_context(patch.object(client, "_one_request", return_value={
                "latency_app_s": 0.5, "effective_input_scale": 1.0,
                "workload_contract": {"schema_version": 1, "actual_rows": 1},
            }))
            stack.enter_context(redirect_stdout(io.StringIO()))
            client.main()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "ok", row["error"])
        self.assertEqual(row["latency_app_s"], "0.500000")
        self.assertEqual(row["latency_s"], "nan")
        self.assertEqual(row["throughput_samples_per_s"], "2.000000")
        self.assertEqual(row["container_mem_usage_avg_bytes"], "1024.000000")
        self.assertEqual(json.loads(row["workload_contract"]), {
            "schema_version": 1, "request_count": 1,
            "variants": [{"count": 1, "contract": {"schema_version": 1, "actual_rows": 1}}],
        })
        self.assertEqual(row["cpu_energy_total_j"], "nan")
        self.assertEqual(row["gpu_energy_total_j"], "nan")
        self.assertEqual(events, ["resource_start", "resource_stop", "resource_close"])

    def test_basic_cli_does_not_probe_packet_rapl_or_perf(self):
        from acprof.cli import run
        with ExitStack() as stack:
            stack.enter_context(patch.object(run.sys, "argv", ["run.py", "--model", "test", "--profiling-mode", "basic", "--notify", "none"]))
            for name in ("bootstrap_project_env", "require_native_linux_host", "require_native_docker"):
                stack.enter_context(patch.object(run, name))
            stack.enter_context(patch.object(run, "require_cgroup_prerequisites", return_value="v2"))
            for name in ("require_packet_latency_prerequisites", "require_cpu_energy_prerequisites", "require_mips_prerequisites"):
                stack.enter_context(patch.object(run, name, side_effect=AssertionError("basic called " + name)))
            stack.enter_context(patch("acprof.host.detect.detect_task", side_effect=RuntimeError("reached task detection")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "reached task detection"):
                run._run_main()


if __name__ == "__main__":
    unittest.main()
