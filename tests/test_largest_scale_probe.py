import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from acprof.cli.probe import main as probe_main
from acprof.host.detect import TaskInfo
from acprof.host.largest_scale_probe import (
    PROBE_SUMMARY_NAME,
    load_largest_scale_entry,
    run_largest_scale_probe,
    select_minimum_resources,
)
from acprof.host.orchestrator import ImageInfo, PlannedInputScales, RunningContainer


def _task_info() -> TaskInfo:
    return TaskInfo(
        model_id="demo/model",
        pipeline_tag="fill-mask",
        task_family="nlp",
        runtime_backend="transformers_pipeline",
        library_name="transformers",
        model_revision="revision-1",
        detection_method="manual",
    )


def _write_plan(root: Path) -> Path:
    path = root / "input_scale_plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "entries": [
                    {
                        "input_scale": 64,
                        "scale_label": "tokens",
                        "payload": {"text": "small"},
                    },
                    {
                        "input_scale": 512,
                        "scale_label": "tokens",
                        "payload": {"text": "largest"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class LargestScaleProbeTests(unittest.TestCase):
    def test_minimum_resource_selection_prefers_cpu_only(self) -> None:
        selected = select_minimum_resources(
            [8, 1, 4],
            [16, 2, 8],
            ["on", "off"],
        )
        self.assertEqual(selected, (1, 2, "off"))
        self.assertEqual(
            select_minimum_resources([2], [8], ["on"]),
            (2, 8, "on"),
        )

    def test_largest_materialized_scale_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            plan = _write_plan(Path(temporary_dir))

            entry = load_largest_scale_entry(plan)

        self.assertEqual(entry["input_scale"], 512.0)
        self.assertEqual(entry["payload"], {"text": "largest"})

    @patch("acprof.host.largest_scale_probe._stop_container_session")
    @patch("acprof.host.largest_scale_probe._start_container_session")
    @patch("acprof.host.largest_scale_probe.requests.post")
    def test_probe_times_exactly_one_largest_request_and_writes_summary(
        self,
        post: Mock,
        start_container: Mock,
        stop_container: Mock,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.text = ""
        response.json.return_value = {
            "effective_input_scale": 512,
            "output_length": 1,
        }
        post.return_value = response
        start_container.return_value = RunningContainer(
            name="probe-container",
            base_url="http://127.0.0.1:8002",
            host_port=8002,
            cold_start_s=12.5,
            cold_start_container_launch_s=0.5,
            cold_start_server_setup_s=1.0,
            cold_start_cuda_init_s=0.0,
            cold_start_model_load_s=10.0,
            cold_start_ready_wait_s=1.0,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            plan = _write_plan(output_dir)
            planned = PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan),
            )

            summary = run_largest_scale_probe(
                task_info=_task_info(),
                image_info=ImageInfo(tag="acprof-nlp-demo--model:latest"),
                planned_input_scales=planned,
                cpu_list=[4, 1],
                mem_list=[8, 2],
                gpu_list=["on", "off"],
                batch_size=1,
                output_dir=output_dir,
                timeout_seconds=30,
            )
            saved_text = (output_dir / PROBE_SUMMARY_NAME).read_text(
                encoding="utf-8"
            )
            saved = json.loads(saved_text)

        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8002/predict")
        self.assertEqual(post.call_args.kwargs["json"], {"text": "largest"})
        self.assertEqual(post.call_args.kwargs["timeout"], 30.0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["resource"]["cpu_cores"], 1)
        self.assertEqual(summary["resource"]["mem_gb"], 2)
        self.assertEqual(summary["resource"]["gpu_mode"], "off")
        self.assertEqual(summary["input"]["planned_scale"], 512.0)
        self.assertEqual(summary["input"]["effective_scale"], 512.0)
        self.assertEqual(summary["input"]["batch_size"], 1)
        self.assertEqual(summary["cold_start"]["total_s"], 12.5)
        self.assertGreater(summary["timing"]["request_s"], 0.0)
        self.assertGreater(summary["timing"]["ready_plus_request_s"], 12.5)
        self.assertEqual(saved["status"], "ok")
        self.assertNotIn("NaN", saved_text)
        stop_container.assert_called_once_with(
            "probe-container",
            log_prefix="[largest-probe]",
        )

    @patch("acprof.host.largest_scale_probe._stop_container_session")
    @patch("acprof.host.largest_scale_probe._start_container_session")
    @patch("acprof.host.largest_scale_probe.requests.post")
    def test_request_timeout_is_persisted_without_formal_csv(
        self,
        post: Mock,
        start_container: Mock,
        stop_container: Mock,
    ) -> None:
        post.side_effect = requests.Timeout("too slow")
        start_container.return_value = RunningContainer(
            name="probe-container",
            base_url="http://127.0.0.1:8002",
            host_port=8002,
            cold_start_s=3.0,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            plan = _write_plan(output_dir)
            summary = run_largest_scale_probe(
                task_info=_task_info(),
                image_info=ImageInfo(tag="image:latest"),
                planned_input_scales=PlannedInputScales(
                    scales=[64.0, 512.0],
                    source="manual",
                    plan_file=str(plan),
                ),
                cpu_list=[1],
                mem_list=[2],
                gpu_list=["off"],
                batch_size=1,
                output_dir=output_dir,
                timeout_seconds=0.01,
            )

            self.assertEqual(summary["status"], "timeout")
            self.assertIn("超过 0.01 秒", summary["error"])
            self.assertTrue((output_dir / PROBE_SUMMARY_NAME).is_file())
            self.assertFalse((output_dir / "result_all.csv").exists())
        stop_container.assert_called_once()

    @patch("acprof.host.largest_scale_probe._stop_container_session")
    @patch("acprof.host.largest_scale_probe._start_container_session")
    @patch("acprof.host.largest_scale_probe.requests.post")
    def test_startup_oom_advances_to_first_viable_memory(
        self,
        post: Mock,
        start_container: Mock,
        stop_container: Mock,
    ) -> None:
        start_container.side_effect = [
            RuntimeError("container_oom_killed during startup"),
            RunningContainer(
                name="probe-4gb",
                base_url="http://127.0.0.1:8002",
                host_port=8002,
                cold_start_s=7.0,
            ),
        ]
        response = Mock(status_code=200, text="")
        response.json.return_value = {"effective_input_scale": 512}
        post.return_value = response

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            plan = _write_plan(output_dir)
            summary = run_largest_scale_probe(
                task_info=_task_info(),
                image_info=ImageInfo(tag="image:latest"),
                planned_input_scales=PlannedInputScales(
                    scales=[64.0, 512.0],
                    source="manual",
                    plan_file=str(plan),
                ),
                cpu_list=[1, 4],
                mem_list=[8, 2, 4],
                gpu_list=["off", "on"],
                batch_size=1,
                output_dir=output_dir,
                timeout_seconds=30,
            )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["resource"]["mem_gb"], 4)
        self.assertEqual(summary["memory_probe"]["minimum_viable_mem_gb"], 4)
        self.assertEqual(
            [item["status"] for item in summary["memory_probe"]["attempts"]],
            ["startup_oom", "ok"],
        )
        self.assertEqual(
            [call.kwargs["mem"] for call in start_container.call_args_list],
            [2, 4],
        )
        self.assertEqual(post.call_count, 1)
        stop_container.assert_called_once_with(
            "probe-4gb",
            log_prefix="[largest-probe]",
        )

    @patch("acprof.host.largest_scale_probe._stop_container_session")
    @patch("acprof.host.largest_scale_probe._start_container_session")
    @patch("acprof.host.largest_scale_probe.requests.post")
    def test_cuda_oom_advances_and_successful_attempt_supplies_timing(
        self,
        post: Mock,
        start_container: Mock,
        stop_container: Mock,
    ) -> None:
        start_container.side_effect = [
            RunningContainer(
                name="probe-2gb",
                base_url="http://127.0.0.1:8002",
                host_port=8002,
                cold_start_s=3.0,
            ),
            RunningContainer(
                name="probe-4gb",
                base_url="http://127.0.0.1:8004",
                host_port=8004,
                cold_start_s=5.0,
            ),
        ]
        oom_response = Mock(
            status_code=500,
            text='{"error":"CUDA out of memory"}',
        )
        success_response = Mock(status_code=200, text="")
        success_response.json.return_value = {"effective_input_scale": 512}
        post.side_effect = [oom_response, success_response]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            plan = _write_plan(output_dir)
            summary = run_largest_scale_probe(
                task_info=_task_info(),
                image_info=ImageInfo(tag="image:latest"),
                planned_input_scales=PlannedInputScales(
                    scales=[64.0, 512.0],
                    source="manual",
                    plan_file=str(plan),
                ),
                cpu_list=[1],
                mem_list=[2, 4, 8],
                gpu_list=["on"],
                batch_size=1,
                output_dir=output_dir,
                timeout_seconds=30,
            )

        self.assertEqual(summary["resource"]["mem_gb"], 4)
        self.assertEqual(
            [item["status"] for item in summary["memory_probe"]["attempts"]],
            ["cuda_oom", "ok"],
        )
        self.assertEqual(summary["cold_start"]["total_s"], 5.0)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(stop_container.call_count, 2)

    @patch("acprof.host.largest_scale_probe._stop_container_session")
    @patch("acprof.host.largest_scale_probe._start_container_session")
    @patch("acprof.host.largest_scale_probe.requests.post")
    def test_all_memory_candidates_oom_without_claiming_a_minimum(
        self,
        post: Mock,
        start_container: Mock,
        stop_container: Mock,
    ) -> None:
        start_container.side_effect = [
            RuntimeError("container_oom_killed during startup"),
            RuntimeError("container_oom_killed during startup"),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            plan = _write_plan(output_dir)
            summary = run_largest_scale_probe(
                task_info=_task_info(),
                image_info=ImageInfo(tag="image:latest"),
                planned_input_scales=PlannedInputScales(
                    scales=[64.0, 512.0],
                    source="manual",
                    plan_file=str(plan),
                ),
                cpu_list=[1],
                mem_list=[2, 4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=output_dir,
                timeout_seconds=30,
            )

        self.assertEqual(summary["status"], "oom")
        self.assertIsNone(summary["resource"]["mem_gb"])
        self.assertIsNone(summary["memory_probe"]["minimum_viable_mem_gb"])
        self.assertEqual(summary["resource"]["last_attempt_mem_gb"], 4)
        self.assertIn("2GB,4GB", summary["error"])
        post.assert_not_called()
        stop_container.assert_not_called()

    @patch("acprof.cli.probe.require_cgroup_prerequisites", return_value="v2")
    @patch("acprof.cli.probe.require_native_docker")
    @patch("acprof.cli.probe.require_native_linux_host")
    @patch("acprof.cli.probe.bootstrap_project_env")
    @patch("acprof.host.detect.detect_task", return_value=_task_info())
    @patch("acprof.cli.probe.plan_input_scales")
    @patch("acprof.cli.probe.run_largest_scale_probe")
    def test_cli_delegates_planning_and_probe_with_requested_matrix(
        self,
        run_probe: Mock,
        plan_scales: Mock,
        _detect_task: Mock,
        _bootstrap: Mock,
        _native_linux: Mock,
        _native_docker: Mock,
        _cgroup: Mock,
    ) -> None:
        def fake_run(**kwargs):
            summary_path = Path(kwargs["output_dir"]) / PROBE_SUMMARY_NAME
            return {
                "status": "ok",
                "timing": {},
                "artifacts": {"summary": str(summary_path)},
            }

        run_probe.side_effect = fake_run

        with tempfile.TemporaryDirectory() as temporary_dir:
            plan_path = Path(temporary_dir) / "planned.json"
            plan_scales.return_value = PlannedInputScales(
                scales=[64.0, 512.0],
                source="manual",
                plan_file=str(plan_path),
            )

            returncode = probe_main(
                [
                    "--model",
                    "demo/model",
                    "--cpus",
                    "1,4",
                    "--mems",
                    "2,8",
                    "--gpus",
                    "off,on",
                    "--batch-size",
                    "3",
                    "--input-scales",
                    "64,512",
                    "--output-dir",
                    temporary_dir,
                    "--skip-build",
                ]
            )
            summary_path = Path(run_probe.call_args.kwargs["output_dir"]) / PROBE_SUMMARY_NAME
            saved = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertEqual(plan_scales.call_args.kwargs["cpu_list"], [1, 4])
        self.assertEqual(plan_scales.call_args.kwargs["mem_list"], [2, 8])
        self.assertEqual(plan_scales.call_args.kwargs["gpu_list"], ["off", "on"])
        self.assertEqual(run_probe.call_args.kwargs["batch_size"], 3)
        self.assertGreaterEqual(saved["timing"]["command_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
