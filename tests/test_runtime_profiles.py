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
from acprof.runtime_profiles import (
    PROFILES,
    RuntimeProfile,
    select_runtime_profile,
)
from acprof.host.runtime_images import request_fingerprint
from runtime_fixture import copy_dependency_tree


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
        original = request_fingerprint(task)
        task.model_download_policy = "full"
        self.assertNotEqual(original, request_fingerprint(task))

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
        from acprof.extensions import CATALOG, ExtensionCatalog
        profile = RuntimeProfile("example-runtime", "multimodal", PROFILES["multimodal-transformers4576"].environment, "example-adapter")
        task = dataclasses.replace(
            moss_task(), model_id="Example/Custom", pipeline_tag="image-text-to-text",
            model_config={"model_type": "example_arch"},
        )
        catalog = ExtensionCatalog()
        declaration = dataclasses.replace(CATALOG.get_extension("multimodal", "transformers_model"),
            extension_id="example", adapter="example-adapter", model_types=("example_arch",),
            profile=profile.profile_id)
        catalog.add(declaration)
        with patch.dict(PROFILES, {profile.profile_id: profile}), patch(
            "acprof.runtime_profiles.select_extension", catalog.select_extension,
        ):
            self.assertIs(select_runtime_profile(task), profile)

    def test_architecture_profile_selection_uses_backend_not_lossy_compatibility_view(self):
        from acprof.extensions import CATALOG, ExtensionCatalog
        catalog = ExtensionCatalog()
        profiles = {}
        for backend in ("first", "second"):
            declaration = dataclasses.replace(CATALOG.get_extension("nlp", "transformers_model"),
                extension_id=backend, backends=(backend,), backend_tasks={}, adapter=backend, model_types=("shared_arch",),
                profile=backend + "-runtime")
            catalog.add(declaration)
            profiles[declaration.profile] = RuntimeProfile(declaration.profile, "nlp",
                PROFILES["nlp-cpu"].environment, backend, task_types=("text-generation",),
                model_types=("shared_arch",), backends=(backend,))
        task = TaskInfo("owner/checkpoint", "text-generation", "nlp", "first", "custom", "fixed", "manual",
                        model_config={"model_type": "shared_arch"})
        with patch.dict(PROFILES, profiles), patch(
            "acprof.runtime_profiles.select_extension", catalog.select_extension,
        ):
            self.assertIs(select_runtime_profile(task), profiles["first-runtime"])

    def test_legacy_torch_override_invalidates_cached_image(self):
        task = dataclasses.replace(moss_task(), model_id="Example/NLP", task_family="nlp", pipeline_tag="text-generation")
        with patch.dict(os.environ, {"ACPROF_NLP_TORCH_SPEC": "torch==2.6.0"}):
            original = request_fingerprint(task)
        with patch.dict(os.environ, {"ACPROF_NLP_TORCH_SPEC": "torch==2.7.0"}):
            self.assertNotEqual(original, request_fingerprint(task))

    def test_source_and_dependency_changes_invalidate_cached_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            source = root / "acprof/handler.py"
            lock = root / "dockerfiles/locks/moss-transformers560.txt"
            source.write_text("adapter = 1\n")
            original = request_fingerprint(moss_task(), root)
            source.write_text("adapter = 2\n")
            changed = request_fingerprint(moss_task(), root)
            self.assertNotEqual(original, changed)
            lock.write_text(lock.read_text().replace("transformers-5.6.0-", "transformers-5.6.1-"))
            self.assertNotEqual(changed, request_fingerprint(moss_task(), root))

    def test_manifest_only_change_invalidates_service_but_not_environment_identity(self):
        from acprof.runtime_profiles import environment_id
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_dependency_tree(root)
            path = root / 'acprof/extensions/test/manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"schema_version": 1, "extensions": []}\n')
            environment = select_runtime_profile(moss_task()).environment
            locked = environment_id(environment, root)
            original = request_fingerprint(moss_task(), root)
            path.write_text('{"schema_version": 1, "extensions": [], "reviewed": true}\n')
            self.assertNotEqual(original, request_fingerprint(moss_task(), root))
            self.assertEqual(locked, environment_id(environment, root))

    def test_posthoc_rejects_image_from_another_build(self):
        from acprof.host.runtime_images import FINGERPRINT_LABEL

        image = "sha256:" + "a" * 64
        with patch("acprof.host.runtime_images.inspect_identity", return_value={
            "image_id": image, "labels": {FINGERPRINT_LABEL: "other-build"},
        }), self.assertRaisesRegex(RuntimeError, "运行环境"):
            docker_runtime.require_image_identity(image, {"build_fingerprint": "original-build"})

    def test_historical_image_identity_does_not_require_new_environment_fields(self):
        from acprof.host.runtime_images import FINGERPRINT_LABEL
        image = "sha256:" + "a" * 64
        original = {"schema_version": 1, "profile_id": "legacy-nlp", "build_fingerprint": "original-build"}
        with patch("acprof.host.runtime_images.inspect_identity", return_value={
            "image_id": image, "labels": {FINGERPRINT_LABEL: "original-build"},
        }), patch("acprof.host.runtime_images.prepare_environment_image", side_effect=AssertionError("rebuilt history")):
            docker_runtime.require_image_identity(image, original)
        self.assertNotIn("environment_id", original)
        self.assertNotIn("platform_id", original)

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
