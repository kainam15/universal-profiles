import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import execution_profile as profile
from acprof.host.profilers import execution_environment as environment


BASE_ID = "sha256:" + "a" * 64
OLD_BASE_ID = "sha256:" + "b" * 64
PROFILE_ID = "sha256:" + "c" * 64
MODEL_TAG = "acprof-test:latest"


def image_result(image_id=BASE_ID, labels=None, references=None):
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "Id": image_id,
            "Config": {"Labels": labels},
            "RepoTags": [MODEL_TAG] if references is None else references,
        }),
        stderr="",
    )


def missing_image():
    return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such image: test")


def build_success():
    return SimpleNamespace(returncode=0, stdout="", stderr="")


class ExecutionProfileImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        dockerfiles = self.project / "dockerfiles"
        dockerfiles.mkdir()
        self.recipes = {}
        for tool in ("massif", "nsys"):
            recipe = (
                'ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n'
                f'LABEL org.acprof.execution-profile.{tool}="1"\n'
            )
            (dockerfiles / f"{tool}.Dockerfile").write_text(recipe)
            self.recipes[tool] = hashlib.sha256(recipe.encode()).hexdigest()

    def labels(self, tool, base_id=BASE_ID):
        return {
            profile.EXECUTION_BASE_IMAGE_LABEL: base_id,
            profile.EXECUTION_DOCKERFILE_LABEL: self.recipes[tool],
            profile.EXECUTION_RUNTIME_LABEL_PREFIX + tool: profile.EXECUTION_RUNTIME_VERSION,
        }

    def prepare(self, tool):
        prepare = getattr(profile, f"_build_{tool}_image")
        return prepare(MODEL_TAG, str(self.project))

    @staticmethod
    def builds(run):
        return [
            call.args[0] for call in run.call_args_list
            if call.args[0][:2] == ["docker", "build"]
        ]

    def test_shared_base_runs_both_tools_on_same_model_without_build(self):
        labels = {
            profile.EXECUTION_RUNTIME_LABEL_PREFIX + tool: profile.EXECUTION_RUNTIME_VERSION
            for tool in ("massif", "nsys")
        }
        # Ready model images need no legacy recipe or separate profiler image.
        for recipe in (self.project / "dockerfiles").iterdir():
            recipe.unlink()
        with patch.object(environment, "_run", return_value=image_result(labels=labels)) as run:
            for tool in ("massif", "nsys"):
                self.assertEqual(self.prepare(tool), BASE_ID)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(self.builds(run), [])

    def test_legacy_image_builds_once_and_next_analysis_skips_build(self):
        for tool in ("massif", "nsys"):
            with self.subTest(tool=tool), patch.object(environment, "_run", side_effect=[
                image_result(), missing_image(), build_success(), build_success(),
                image_result(), build_success(),
                image_result(PROFILE_ID, self.labels(tool)),
                image_result(), image_result(PROFILE_ID, self.labels(tool)),
            ]) as run:
                self.assertEqual(self.prepare(tool), PROFILE_ID)
                self.assertEqual(self.prepare(tool), PROFILE_ID)
            builds = self.builds(run)
            self.assertEqual(len(builds), 1)
            tag = next(
                call.args[0] for call in run.call_args_list
                if call.args[0][:3] == ["docker", "image", "tag"]
            )
            self.assertEqual(tag[3], BASE_ID)
            self.assertIn(f"BASE_IMAGE={tag[4]}", builds[0])
            self.assertTrue(any(
                call.args[0] == ["docker", "image", "rm", "--no-prune", tag[4]]
                for call in run.call_args_list
            ))
            self.assertNotIn(f"BASE_IMAGE={MODEL_TAG}", builds[0])
            self.assertIn(f"{profile.EXECUTION_BASE_IMAGE_LABEL}={BASE_ID}", builds[0])
            self.assertIn(
                f"{profile.EXECUTION_DOCKERFILE_LABEL}={self.recipes[tool]}", builds[0]
            )

    def test_existing_compatible_image_needs_no_build_cache(self):
        for tool in ("massif", "nsys"):
            with self.subTest(tool=tool), patch.object(environment, "_run", side_effect=[
                image_result(), image_result(PROFILE_ID, self.labels(tool)),
            ]) as run:
                self.assertEqual(self.prepare(tool), PROFILE_ID)
            self.assertEqual(self.builds(run), [])

    def test_changed_base_recipe_or_missing_metadata_rebuilds_same_tag(self):
        for tool in ("massif", "nsys"):
            changed_recipe = self.labels(tool)
            changed_recipe[profile.EXECUTION_DOCKERFILE_LABEL] = "outdated"
            missing_runtime = self.labels(tool)
            missing_runtime.pop(profile.EXECUTION_RUNTIME_LABEL_PREFIX + tool)
            for reason, labels in (
                ("base changed under same model tag", self.labels(tool, OLD_BASE_ID)),
                ("recipe changed", changed_recipe),
                ("runtime missing", missing_runtime),
                ("old unlabeled profiler image", {}),
            ):
                with self.subTest(tool=tool, reason=reason), patch.object(
                    environment, "_run", side_effect=[
                        image_result(), image_result(PROFILE_ID, labels),
                        build_success(), build_success(), image_result(), build_success(),
                        image_result(PROFILE_ID, self.labels(tool)),
                    ]
                ) as run:
                    self.assertEqual(self.prepare(tool), PROFILE_ID)
                builds = self.builds(run)
                self.assertEqual(len(builds), 1)
                digest = hashlib.sha256(MODEL_TAG.encode()).hexdigest()[:12]
                self.assertEqual(
                    builds[0][builds[0].index("--tag") + 1],
                    f"acprof-{tool}-{digest}:latest",
                )

    def test_capability_is_checked_for_the_requested_tool(self):
        labels = {profile.EXECUTION_RUNTIME_LABEL_PREFIX + "nsys": "1"}
        with patch.object(environment, "_run", side_effect=[
            image_result(labels=labels), missing_image(),
            build_success(), build_success(), image_result(), build_success(),
            image_result(PROFILE_ID, self.labels("massif")),
        ]) as run:
            self.assertEqual(self.prepare("massif"), PROFILE_ID)
        self.assertEqual(len(self.builds(run)), 1)

    def test_failed_legacy_build_is_reported_for_the_requested_tool(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="apt failed")
        for tool in ("massif", "nsys"):
            with self.subTest(tool=tool), patch.object(environment, "_run", side_effect=[
                image_result(), missing_image(), build_success(), failed,
                image_result(), build_success(),
            ]) as run:
                with self.assertRaisesRegex(RuntimeError, f"{tool}_image_build_failed:apt failed"):
                    self.prepare(tool)
            self.assertEqual(run.call_args_list[-1].args[0][:4],
                             ["docker", "image", "rm", "--no-prune"])

    def test_build_must_publish_matching_metadata_before_use(self):
        with patch.object(environment, "_run", side_effect=[
            image_result(), missing_image(), build_success(), build_success(),
            image_result(), build_success(), image_result(PROFILE_ID),
        ]), self.assertRaisesRegex(RuntimeError, "built_image_metadata_mismatch"):
            self.prepare("massif")

    def test_daemon_failure_is_not_treated_as_a_missing_cached_image(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to Docker daemon")
        with patch.object(environment, "_run", side_effect=[image_result(), failed]) as run:
            with self.assertRaisesRegex(RuntimeError, "execution_image_inspect_failed"):
                self.prepare("nsys")
        self.assertEqual(self.builds(run), [])

    def test_source_image_is_preserved_if_original_tag_moves_during_build(self):
        with patch.object(environment, "_run", side_effect=[
            image_result(), missing_image(), build_success(), build_success(),
            image_result(references=[]), image_result(PROFILE_ID, self.labels("massif")),
        ]) as run:
            self.assertEqual(self.prepare("massif"), PROFILE_ID)
        self.assertFalse(any(
            call.args[0][:3] == ["docker", "image", "rm"] for call in run.call_args_list
        ))

    def test_temporary_tag_failure_does_not_start_build(self):
        failed = SimpleNamespace(returncode=1, stdout="", stderr="tag failed")
        with patch.object(environment, "_run", side_effect=[
            image_result(), missing_image(), failed,
        ]) as run:
            with self.assertRaisesRegex(RuntimeError, "massif_image_build_failed:tag failed"):
                self.prepare("massif")
        self.assertEqual(self.builds(run), [])

    def test_missing_model_does_not_trigger_a_pull_or_build(self):
        with patch.object(environment, "_run", return_value=missing_image()) as run:
            with self.assertRaisesRegex(RuntimeError, "base_image_not_found"):
                self.prepare("massif")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.builds(run), [])

    def test_invalid_image_metadata_does_not_trigger_a_build(self):
        for payload in (
            "not json", "[]", '{"Id":"latest"}',
            json.dumps({"Id": BASE_ID, "Config": {"Labels": ["invalid"]}}),
        ):
            result = SimpleNamespace(returncode=0, stdout=payload, stderr="")
            with self.subTest(payload=payload), patch.object(environment, "_run", return_value=result) as run:
                with self.assertRaisesRegex(RuntimeError, "invalid_metadata"):
                    self.prepare("nsys")
            self.assertEqual(self.builds(run), [])


if __name__ == "__main__":
    unittest.main()
