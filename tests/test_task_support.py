import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Button, DataTable, Static

from acprof.cli import probe, run
from acprof.cli.tui import AcprofTui, PendingLaunch
from acprof.cli.tui_core import RunConfig, RunProgressTracker
from acprof.cli.tui_i18n import translate
from acprof.cli.tui_log import SelectableLog
from acprof.host import detect
from acprof.host.task_support import TaskSupportError, require_task_support


def task_info(tag="image-text-to-text", family="unknown"):
    return detect.TaskInfo(
        model_id="example/caption-model",
        pipeline_tag=tag,
        task_family=family,
        runtime_backend="transformers_pipeline",
        library_name="transformers",
        model_revision="test-revision",
        detection_method="hub_api",
    )


class TaskSupportTests(unittest.TestCase):
    def test_run_and_probe_reject_unsupported_tasks_before_build_or_results(self):
        for module in (run, probe):
            for tag, family, batch_size in (
                ("image-to-text", "nlp", 1),
                ("video-classification", "unknown", 1),
                ("image-to-text", "cv", 2),
            ):
                with self.subTest(entrypoint=module.__name__, task=tag):
                    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                        root = Path(tmp)
                        existing = root / "example--caption-model"
                        existing.mkdir()
                        csv = existing / "result_all.csv"
                        csv.write_text("existing measurement\n", encoding="utf-8")
                        stack.enter_context(patch.object(module, "bootstrap_project_env"))
                        for guard in ("require_native_linux_host", "require_native_docker", "require_cgroup_prerequisites"):
                            stack.enter_context(patch.object(module, guard, return_value="v2"))
                        if module is run:
                            for guard in ("require_packet_latency_prerequisites", "require_cpu_energy_prerequisites", "require_mips_prerequisites", "_start_tmux_terminal_log"):
                                stack.enter_context(patch.object(run, guard, return_value=None))
                        stack.enter_context(patch("acprof.host.detect.detect_task", return_value=task_info(tag, family)))
                        build_target = "acprof.host.orchestrator.prepare_image" if module is run else "acprof.cli.probe.prepare_image"
                        build = stack.enter_context(patch(build_target, side_effect=AssertionError("unsupported task reached image preparation")))
                        stderr = stack.enter_context(redirect_stderr(io.StringIO()))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        argv = ["--model", "example/caption-model", "--output-dir", tmp, "--skip-build", "--batch-size", str(batch_size)]
                        if module is run:
                            stack.enter_context(patch.object(sys, "argv", ["run.py", *argv, "--notify", "none"]))
                            with self.assertRaises(SystemExit) as caught:
                                run.main()
                            code = caught.exception.code
                        else:
                            code = probe.main(argv)
                        self.assertEqual(code, 2)
                        build.assert_not_called()
                        output = stderr.getvalue()
                        self.assertIn("[task-support][ERROR]", output)
                        self.assertIn(tag, output)
                        self.assertIn("example/caption-model", output)
                        self.assertIn("原因", output)
                        self.assertIn("解决办法", output)
                        self.assertIn("README.md", output)
                        self.assertNotIn("Traceback", output)
                        self.assertNotIn("startup_oom", output)
                        if batch_size != 1:
                            self.assertIn("--batch-size 1", output)
                        elif tag == "image-to-text":
                            self.assertIn("--task-family", output)
                        self.assertEqual(csv.read_text(encoding="utf-8"), "existing measurement\n")
                        self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob("*")), ["example--caption-model", "example--caption-model/result_all.csv"])

    def test_hub_task_without_adapter_is_preserved_instead_of_guessed_as_nlp(self):
        hub = SimpleNamespace(pipeline_tag="image-text-to-text", library_name="transformers", sha="rev")
        with patch("huggingface_hub.model_info", return_value=hub), patch.object(
            detect, "_detect_from_config", return_value=task_info("text2text-generation", "nlp")
        ) as fallback:
            info = detect.detect_task("example/multimodal")
        self.assertEqual(info.pipeline_tag, "image-text-to-text")
        self.assertEqual(info.task_family, "unknown")
        fallback.assert_not_called()

    def test_manual_override_can_correct_hub_metadata(self):
        hub = SimpleNamespace(pipeline_tag="unregistered-task", library_name="transformers", sha="rev")
        with patch("huggingface_hub.model_info", return_value=hub):
            info = detect.detect_task("example/model", override_tag="image-classification")
        self.assertEqual((info.pipeline_tag, info.task_family), ("image-classification", "cv"))

    def test_blip_architecture_fallback_does_not_treat_images_as_nlp(self):
        for architecture in (
            "BlipForConditionalGeneration", "Blip2ForConditionalGeneration",
            "InstructBlipForConditionalGeneration", "VisionEncoderDecoderModel",
        ):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "config.json"
                config.write_text(json.dumps({"architectures": [architecture]}), encoding="utf-8")
                with patch("huggingface_hub.hf_hub_download", return_value=str(config)):
                    info = detect._detect_from_config("example/caption-model")
            self.assertEqual((info.pipeline_tag, info.task_family), ("image-to-text", "cv"))

    def test_supported_families_remain_available_and_wrong_family_is_actionable(self):
        for task, family in (
            ("fill-mask", "nlp"), ("text-generation", "nlp"),
            ("image-classification", "cv"), ("object-detection", "cv"),
            ("image-to-text", "cv"),
            ("automatic-speech-recognition", "audio"),
            ("time-series-forecasting", "timeseries"), ("text-to-image", "diffusion"),
        ):
            with self.subTest(task=task):
                require_task_support(task_info(task, family))
        with self.assertRaises(TaskSupportError) as caught:
            require_task_support(task_info("image-classification", "nlp"))
        self.assertIn("任务族 nlp 与之不匹配", str(caught.exception))
        self.assertIn("--task-family", str(caught.exception))

    def test_manual_caption_task_is_supported_but_requires_single_image_batch(self):
        with patch.object(detect, "_detect_from_hub", return_value=task_info("fill-mask", "nlp")):
            info = detect.detect_task("example/model", override_tag="image-to-text")
        require_task_support(info)
        with self.assertRaisesRegex(TaskSupportError, "--batch-size 1"):
            require_task_support(info, batch_size=2)

    def test_progress_preserves_unsupported_task_and_points_to_remedies(self):
        tracker = RunProgressTracker()
        state = tracker.feed("[task-support][ERROR] Unsupported collection task: image-text-to-text")
        self.assertEqual(state.stage, "任务不支持")
        self.assertIn("image-text-to-text", state.detail)
        self.assertIn("解决办法见日志", state.detail)
        self.assertFalse(state.measurement_active)
        self.assertEqual(state.errors, 1)
        state = tracker.feed("  原因：当前项目尚未登记该任务类型的采集适配。")
        self.assertEqual(state.stage, "任务不支持")
        self.assertIn("image-text-to-text", translate(state.detail, "en"))
        self.assertIn("log", translate(state.detail, "en"))


class TaskSupportTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_and_probe_keep_actionable_failure_visible_in_both_languages(self):
        with self.assertRaises(TaskSupportError) as caught:
            require_task_support(task_info())
        lines = str(caught.exception).splitlines()
        for size in ((80, 24), (120, 30)):
            with tempfile.TemporaryDirectory() as tmp:
                config = replace(RunConfig.smoke("example/caption-model"), output_dir=tmp)
                old_csv = config.result_csv(Path.cwd())
                old_csv.parent.mkdir(parents=True)
                old_csv.write_text("existing measurement\n", encoding="utf-8")
                app = AcprofTui(config, settings_path=Path(tmp) / "tui.json")
                async with app.run_test(size=size) as pilot:
                    for kind in ("run", "probe"):
                        for language in ("zh", "en"):
                            with self.subTest(size=size, kind=kind, language=language):
                                app.ui_preferences = replace(app.ui_preferences, language=language)
                                app._apply_language()
                                with patch.object(app, "_execute_command"):
                                    app._launch(PendingLaunch(("unused",), kind, config))
                                await pilot.pause()
                                tracker = RunProgressTracker()
                                for line in lines:
                                    previous = tracker.snapshot
                                    state = tracker.feed(line)
                                    app._consume_process_line(line, state, previous != state)
                                with patch.object(app, "notify") as notify, patch.object(
                                    app, "_update_result_summary"
                                ) as read_old_results:
                                    app._process_finished(kind, 2, tracker.snapshot, "")
                                await pilot.pause()
                                self.assertEqual(app._latest_snapshot.stage, "任务不支持")
                                self.assertEqual(
                                    app.query_one("#status-stage", Static).content,
                                    "任务不支持" if language == "zh" else "Unsupported task",
                                )
                                stage = app.query_one("#status-stage", Static)
                                self.assertGreater(stage.region.width, 0)
                                self.assertGreater(stage.region.height, 0)
                                self.assertLess(stage.region.bottom, size[1])
                                self.assertTrue(stage.is_on_screen)
                                self.assertIn("image-text-to-text", app._latest_snapshot.detail)
                                self.assertIn("解决办法", app.query_one("#run-log", SelectableLog).text)
                                self.assertIn(
                                    "处理办法" if language == "zh" else "Remedies:",
                                    app.query_one("#run-log", SelectableLog).text.splitlines()[-1],
                                )
                                self.assertIn("image-text-to-text", str(notify.call_args.args[0]))
                                self.assertFalse(app.query_one("#start-run", Button).disabled)
                                self.assertFalse(app.query_one("#probe-largest", Button).disabled)
                                self.assertEqual(app.query_one("#matrix-table", DataTable).row_count, 0)
                                self.assertFalse(app._latest_snapshot.measurement_active)
                                read_old_results.assert_not_called()
                                self.assertEqual(old_csv.read_text(encoding="utf-8"), "existing measurement\n")


if __name__ == "__main__":
    unittest.main()
