import csv
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from acprof.cli import run
from acprof.config import CSV_FIELDS
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.static_metadata import StaticMeta


class RunRecoveryTests(unittest.TestCase):
    def test_seeded_resume_reuses_frozen_order_without_plan_generation(self):
        seen = []
        def interrupted(**kwargs):
            seen.append(kwargs['cpu'])
            if len(seen) == 2:
                raise KeyboardInterrupt()
            return self.write_case(**kwargs)
        with self.assertRaises(KeyboardInterrupt):
            self.invoke('--matrix-order', 'seeded', '--matrix-seed', '37', case=interrupted)
        path = self.directory / 'matrix_plan.json'
        original = path.read_bytes()
        plan = json.loads(original)
        self.assertEqual(seen, [c['cpu_cores'] for c in plan['cases']])
        with patch('acprof.host.matrix_plan.build_matrix_plan', side_effect=AssertionError('reshuffle')):
            self.invoke('--matrix-order', 'seeded', '--matrix-seed', '37', '--resume')
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.calls, [c['cpu_cores'] for c in plan['cases']])

    def test_resume_rejects_tampered_frozen_plan(self):
        self.invoke()
        path = self.directory / 'matrix_plan.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(SystemExit):
            self.invoke('--resume')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "org--model"
        self.task = TaskInfo("org/model", "fill-mask", "nlp", "transformers_pipeline",
                             "transformers", "a" * 40, "manual")
        self.image = ImageInfo(tag="sha256:" + "b" * 64)
        self.calls = []

    def write_case(self, **kwargs):
        self.calls.append(kwargs["cpu"])
        path = self.directory / f'result_case_org--model_{kwargs["cpu"]}c_4g_off.csv'
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores=str(kwargs["cpu"]), mem_cap_gb="4", gpu_mode="off",
                   input_scale="64", warmup="0", repeat_idx="0", status="ok", error="",
                   latency_app_s="0.1", latency_s="0.09", throughput_samples_per_s="10",
                   container_cpu_util_avg_pct="0", container_mem_usage_avg_bytes="1024",
                   cpu_energy_total_j="1", vcpu_energy_total_j="0", cpu_instructions_per_request="0")
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        return str(path)

    def prepare_plan(self, **kwargs):
        path = Path(kwargs["output_dir"]) / "input_scale_plan.json"
        path.write_text(json.dumps({"schema_version": 2, "entries": [
            {"input_scale": 64, "payload": {"text": "test input"}}
        ]}))
        return PlannedInputScales([64.0], "manual", str(path), {},
                                  hashlib.sha256(path.read_bytes()).hexdigest())

    def invoke(self, *extra, case=None):
        metadata = StaticMeta(
            model_name="org/model", model_revision="a" * 40, task_family="nlp",
            pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
            image_tag=self.image.tag, batch_size=1, input_scale_type="seq_length",
            run_command="fixture", model_download_url="https://huggingface.co/org/model",
            gpu="none", gpu_mem_total_bytes=None, model_cache_bytes=1, docker_image_bytes=1,
            environment="linux", cpu_power_source="rapl", vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance", cpu_boost="off", cgroup_version="v2",
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [
                "run.py", "--model", "org/model", "--cpus", "1,2", "--mems", "4",
                "--gpus", "off", "--input-scales", "64", "--warmup", "0", "--repeat", "1",
                "--repeat-in-window", "1", "--notify", "none", "--no-prune-startup-oom",
                "--output-dir", str(self.root), "--matrix-order", "declared", *extra,
            ]))
            for name in ("bootstrap_project_env", "require_native_linux_host", "require_native_docker",
                         "require_packet_latency_prerequisites", "require_cpu_energy_prerequisites",
                         "require_mips_prerequisites"):
                stack.enter_context(patch.object(run, name))
            stack.enter_context(patch.object(run, "_start_tmux_terminal_log", return_value=None))
            stack.enter_context(patch.object(run, "require_cgroup_prerequisites", return_value="v2"))
            stack.enter_context(patch("acprof.host.detect.detect_task", return_value=self.task))
            stack.enter_context(patch("acprof.host.docker_runtime.prepare_image", return_value=self.image))
            stack.enter_context(patch("acprof.host.docker_runtime.require_image_identity"))
            stack.enter_context(patch("acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}))
            stack.enter_context(patch("acprof.host.static_metadata.collect_static_meta", return_value=metadata))
            stack.enter_context(patch("acprof.host.input_plan.plan_input_scales", side_effect=self.prepare_plan))
            stack.enter_context(patch("acprof.host.orchestrator.run_single_case", side_effect=case or self.write_case))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            run.main()

    def test_existing_results_are_not_overwritten_without_resume(self):
        self.directory.mkdir()
        metadata = self.directory / "static_meta.json"
        original = b'{"cgroup_version":"v2","preserve":"original"}'
        metadata.write_bytes(original)
        with self.assertRaises((RuntimeError, SystemExit)):
            self.invoke()
        self.assertEqual(metadata.read_bytes(), original)
        self.assertEqual(self.calls, [])

    def assert_missing_required_measurements_rejected(self, mode):
        def missing_memory(**kwargs):
            path = Path(self.write_case(**kwargs))
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if kwargs["cpu"] == 2:
                rows[-1]["container_mem_usage_avg_bytes"] = "nan"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return str(path)
        with self.assertRaises(SystemExit) as caught:
            self.invoke("--profiling-mode", mode, case=missing_memory)
        self.assertEqual(caught.exception.code, 1)
        report = json.loads((self.directory / "capability_report.json").read_text())
        self.assertEqual(report["measurement"]["container_memory"]["status"], "unavailable")
        self.assertFalse(report["full_profile_complete"])
        self.assertTrue((self.directory / "result_all.csv").is_file())
        self.assertEqual(len(list(self.directory.glob("result_case_*.csv"))), 2)
        state = json.loads((self.directory / "run_state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["artifacts"]["static_meta.json"], hashlib.sha256((self.directory / "static_meta.json").read_bytes()).hexdigest())

    def test_full_rejects_required_measurement_missing_after_preflight(self):
        self.assert_missing_required_measurements_rejected("full")

    def test_basic_rejects_required_measurement_missing_after_preflight(self):
        self.assert_missing_required_measurements_rejected("basic")

    def test_resume_keeps_completed_case_and_restarts_interrupted_case(self):
        def interrupted(**kwargs):
            path = self.write_case(**kwargs)
            if kwargs["cpu"] == 2:
                Path(str(path) + ".requests.jsonl").write_text('{"partial":true}\n')
                raise KeyboardInterrupt()
            return path
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(case=interrupted)
        original_meta = (self.directory / "static_meta.json").read_bytes()
        self.calls.clear()
        self.invoke("--resume")
        self.assertEqual(self.calls, [2])
        self.assertEqual((self.directory / "static_meta.json").read_bytes(), original_meta)
        with (self.directory / "result_all.csv").open() as stream:
            self.assertEqual([row["cpu_cores"] for row in csv.DictReader(stream)], ["1", "2"])
        self.assertTrue(list((self.directory / "interrupted_cases").rglob("*.csv")))
        archived_samples = list((self.directory / "interrupted_cases").rglob("*.requests.jsonl"))
        self.assertEqual(len(archived_samples), 1)
        self.assertEqual(archived_samples[0].read_text(), '{"partial":true}\n')

    def interrupt_after_first(self):
        def interrupted(**kwargs):
            if kwargs["cpu"] == 2:
                raise KeyboardInterrupt()
            return self.write_case(**kwargs)
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(case=interrupted)
        self.calls.clear()

    def test_resume_rejects_changed_measurement_parameters(self):
        self.interrupt_after_first()
        before = (self.directory / "run_state.json").read_bytes()
        with self.assertRaises(SystemExit):
            self.invoke("--resume", "--repeat", "2")
        self.assertEqual(self.calls, [])
        self.assertEqual((self.directory / "run_state.json").read_bytes(), before)

    def test_resume_rejects_changed_input_plan_without_overwriting_it(self):
        self.interrupt_after_first()
        path = self.directory / "input_scale_plan.json"
        path.write_text('{"changed":true}')
        with self.assertRaises(SystemExit):
            self.invoke("--resume")
        self.assertEqual(path.read_text(), '{"changed":true}')
        self.assertEqual(self.calls, [])

    def test_completed_resume_does_not_repeat_measurements_or_rewrite_result(self):
        self.invoke()
        result = self.directory / "result_all.csv"
        before = result.read_bytes(), result.stat().st_mtime_ns
        self.calls.clear()
        self.invoke("--resume")
        self.assertEqual(self.calls, [])
        self.assertEqual((result.read_bytes(), result.stat().st_mtime_ns), before)

    def test_changed_completed_case_is_rejected(self):
        self.interrupt_after_first()
        source = self.directory / "result_case_org--model_1c_4g_off.csv"
        source.write_bytes(source.read_bytes() + b"corrupt,row\n")
        before = source.read_bytes()
        with self.assertRaises(SystemExit):
            self.invoke("--resume")
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_directory_lock_rejects_second_writer_and_releases_on_exit(self):
        from acprof.host.run_state import ResultDirectoryLock, RunStateError
        with ResultDirectoryLock(self.directory):
            with self.assertRaises(RunStateError):
                with ResultDirectoryLock(self.directory):
                    self.fail("two writers acquired the same directory")
        with ResultDirectoryLock(self.directory):
            pass

    def test_different_output_directories_cannot_measure_concurrently(self):
        from acprof.host.run_state import RunState, RunStateError
        first = RunState(self.directory, {}, resume=False, project_dir=str(Path(__file__).resolve().parents[1]))
        self.addCleanup(first.close)
        with self.assertRaises(RunStateError):
            second = RunState(self.directory.parent / "other", {}, resume=False,
                              project_dir=str(Path(__file__).resolve().parents[1]))
            second.close()


if __name__ == "__main__":
    unittest.main()
