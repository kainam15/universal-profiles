import dataclasses
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from acprof.host.detect import TaskInfo
from acprof.host import docker_runtime
from acprof.container.handlers import HandlerRegistry
from acprof.workloads import get_generator
from acprof.runtime_profiles import ARCHITECTURE_PROFILES, PROFILES, RuntimeProfile, select_runtime_profile
from acprof.host.runtime_images import build_fingerprint


def moss_task():
    return TaskInfo(
        model_id="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        pipeline_tag="audio-text-to-text", task_family="multimodal",
        runtime_backend="transformers_model", library_name="transformers",
        model_revision="704aa4a9c304e8520be88901e0d1960158ef5b15",
        detection_method="unit",
    )


class RuntimeProfileRegressionTests(unittest.TestCase):
    def test_download_policy_changes_image_identity(self):
        task = moss_task()
        task.model_download_policy = "auto"
        original = build_fingerprint(task)
        task.model_download_policy = "full"
        self.assertNotEqual(original, build_fingerprint(task))

    def test_moss_architecture_reuses_adapter_for_another_checkpoint(self):
        task = dataclasses.replace(moss_task(), model_id="Example/Moss", model_config={"model_type": "moss_transcribe_diarize"})
        self.assertEqual(select_runtime_profile(task).adapter, "moss-transcribe-diarize")

    def test_unknown_audio_architecture_is_rejected_before_build(self):
        task = dataclasses.replace(moss_task(), model_id="Example/New", model_config={"model_type": "new_arch"})
        with self.assertRaisesRegex(ValueError, "No registered"):
            select_runtime_profile(task)

    def test_native_metadata_can_have_null_auto_map(self):
        task = dataclasses.replace(
            moss_task(), model_id="Example/Native", pipeline_tag="image-text-to-text",
            model_config={"model_type": "qwen2_vl", "auto_map": None},
        )
        self.assertEqual(select_runtime_profile(task).profile_id, "multimodal-transformers4576")

    def test_another_model_adapter_can_select_its_own_runtime(self):
        profile = RuntimeProfile("example-runtime", "multimodal", "example-adapter")
        task = dataclasses.replace(
            moss_task(), model_id="Example/Custom", pipeline_tag="image-text-to-text",
            model_config={"model_type": "example_arch"},
        )
        with patch.dict(PROFILES, {profile.profile_id: profile}), patch.dict(
            ARCHITECTURE_PROFILES, {"example_arch": profile.profile_id},
        ):
            self.assertIs(select_runtime_profile(task), profile)

    def test_legacy_torch_override_invalidates_cached_image(self):
        task = dataclasses.replace(moss_task(), model_id="Example/NLP", task_family="nlp", pipeline_tag="text-generation")
        with patch.dict(os.environ, {"ACPROF_NLP_TORCH_SPEC": "torch==2.6.0"}):
            original = build_fingerprint(task)
        with patch.dict(os.environ, {"ACPROF_NLP_TORCH_SPEC": "torch==2.7.0"}):
            self.assertNotEqual(original, build_fingerprint(task))

    def test_source_and_dependency_changes_invalidate_cached_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acprof").mkdir()
            (root / "dockerfiles/locks").mkdir(parents=True)
            source = root / "acprof/handler.py"
            lock = root / "dockerfiles/locks/moss-transformers560.txt"
            source.write_text("adapter = 1\n")
            lock.write_text("transformers==5.6.0\n")
            (root / "dockerfiles/locks/common-cu128.txt").write_text("torch==2.11.0+cu128\n")
            original = build_fingerprint(moss_task(), root)
            source.write_text("adapter = 2\n")
            changed = build_fingerprint(moss_task(), root)
            self.assertNotEqual(original, changed)
            lock.write_text("transformers==5.6.1\n")
            self.assertNotEqual(changed, build_fingerprint(moss_task(), root))

    def test_posthoc_rejects_image_from_another_build(self):
        from acprof.host.runtime_images import FINGERPRINT_LABEL

        image = "sha256:" + "a" * 64
        with patch("acprof.host.runtime_images.inspect_identity", return_value={
            "image_id": image, "labels": {FINGERPRINT_LABEL: "other-build"},
        }), self.assertRaisesRegex(RuntimeError, "运行环境"):
            docker_runtime.require_image_identity(image, {"build_fingerprint": "original-build"})

    def test_startup_error_keeps_python_stderr(self):
        import subprocess

        def fake_run(command, **kwargs):
            stderr = "ValueError: custom code is required" if command[:2] == ["docker", "logs"] else ""
            return subprocess.CompletedProcess(command, 0, stdout="", stderr=stderr)

        with patch.object(docker_runtime, "_run", side_effect=fake_run), patch.object(
            docker_runtime, "_inspect_container_state", return_value={
                "Status": "exited", "Running": False, "ExitCode": 1,
            },
        ), patch("requests.get", side_effect=ConnectionError), self.assertRaisesRegex(RuntimeError, "custom code is required"):
            docker_runtime._start_container_session(
                moss_task(), 1, 2, "off", docker_runtime.ImageInfo(tag="unit-image"), "unit-startup", "[test]",
            )

    def test_model_revision_changes_image_identity(self):
        task = moss_task()
        other = dataclasses.replace(task, model_revision="a" * 40)
        self.assertNotEqual(
            docker_runtime._model_image_tag(task),
            docker_runtime._model_image_tag(other),
        )

    def test_explicit_adapter_is_used_by_all_container_entrypoints(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_ADAPTER": "moss-transcribe-diarize"}):
            handler = HandlerRegistry.get("multimodal", "transformers_model")
        self.assertEqual(type(handler).__name__, "MossTranscribeDiarizeHandler")

    def test_unknown_explicit_adapter_cannot_fall_back(self):
        with patch.dict(os.environ, {"ACPROF_MODEL_ADAPTER": "unknown-adapter"}):
            with self.assertRaisesRegex(ValueError, "adapter"):
                HandlerRegistry.get("multimodal", "transformers_model")

    def test_new_adapter_registers_without_changing_family_routing(self):
        class ExampleAdapter:
            pass

        with patch.dict(HandlerRegistry._adapters, clear=True), patch.dict(
            os.environ, {"ACPROF_MODEL_ADAPTER": "example"},
        ):
            HandlerRegistry.register_adapter("example", "multimodal", "transformers_model", ExampleAdapter)
            self.assertIsInstance(HandlerRegistry.get("multimodal", "transformers_model"), ExampleAdapter)
            with self.assertRaisesRegex(ValueError, "adapter"):
                HandlerRegistry.get("nlp", "transformers_model")

    def test_moss_default_workload_requests_timestamped_transcription(self):
        generator = get_generator(
            "multimodal", moss_task().model_id, "audio-text-to-text", 1,
        )
        payload = generator.generate(1)
        self.assertIn("说话人", payload["samples"][0]["text"])
        self.assertGreaterEqual(payload["params"]["max_new_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
