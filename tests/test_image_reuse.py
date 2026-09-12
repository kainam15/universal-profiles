import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from acprof.host import docker_runtime
from acprof.host.detect import TaskInfo
from acprof.host.runtime_images import FINGERPRINT_LABEL, build_fingerprint


class PrepareImageTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo(
            model_id='Org/Model.v1', pipeline_tag='fill-mask', task_family='nlp',
            runtime_backend='transformers_pipeline', library_name='transformers',
            model_revision='1' * 40, detection_method='unit',
        )
        self.tag = docker_runtime._model_image_tag(self.task, '/project')
        self.image_id = 'sha256:' + 'a' * 64
        self.fingerprint = build_fingerprint(self.task, '/project')
        self.manifest = {
            'build_fingerprint': self.fingerprint, 'profile_id': 'legacy-nlp',
            'adapter': 'family-default', 'model_id': self.task.model_id,
            'model_revision': self.task.model_revision,
            'model_snapshot_revision': self.task.model_revision,
        }

    def existing_image(self, command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            payload = {'image_id': self.image_id, 'labels': {FINGERPRINT_LABEL: self.fingerprint}}
        else:
            self.assertEqual(command[-2:], [self.image_id, '/app/runtime_environment.json'])
            payload = self.manifest
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr='')

    def test_existing_image_is_verified_and_pinned_without_building(self):
        stdout = io.StringIO()
        with patch.object(docker_runtime, '_run', side_effect=self.existing_image), patch.object(
            docker_runtime, 'build_image',
        ) as build, redirect_stdout(stdout):
            image = docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)
        self.assertEqual(image.tag, self.image_id)
        self.assertEqual(image.name, self.tag)
        self.assertEqual(image.runtime_environment, self.manifest)
        build.assert_not_called()
        self.assertIn('跳过构建并复用', stdout.getvalue())

    def test_missing_image_is_announced_before_building(self):
        query = subprocess.CompletedProcess([], 1, stdout='', stderr='Error: No such image: expected')
        stdout = io.StringIO()
        built = docker_runtime.ImageInfo(tag=self.image_id)

        def build_image(task, project_dir):
            self.assertIs(task, self.task)
            self.assertEqual(project_dir, '/project')
            self.assertIn(self.tag, stdout.getvalue())
            self.assertIn('自动构建', stdout.getvalue())
            return built

        with patch.object(docker_runtime, '_run', return_value=query), patch.object(
            docker_runtime, 'build_image', side_effect=build_image,
        ), redirect_stdout(stdout):
            image = docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)
        self.assertIs(image, built)

    def test_docker_query_errors_stop_without_building(self):
        for error in ('Cannot connect to the Docker daemon', 'permission denied'):
            with self.subTest(error=error):
                query = subprocess.CompletedProcess([], 1, stdout='', stderr=error)
                with patch.object(docker_runtime, '_run', return_value=query), patch.object(
                    docker_runtime, 'build_image',
                ) as build, self.assertRaisesRegex(RuntimeError, error):
                    docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)
                build.assert_not_called()

    def test_retagged_or_wrong_revision_image_is_rejected(self):
        self.manifest['model_revision'] = 'another-revision'
        with patch.object(docker_runtime, '_run', side_effect=self.existing_image), patch.object(
            docker_runtime, 'build_image',
        ) as build, self.assertRaisesRegex(RuntimeError, 'revision'):
            docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)
        build.assert_not_called()

    def test_actual_snapshot_must_match_declared_revision(self):
        self.manifest['model_snapshot_revision'] = '2' * 40
        with patch.object(docker_runtime, '_run', side_effect=self.existing_image), self.assertRaisesRegex(RuntimeError, 'revision'):
            docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)

    def test_mutable_revision_is_resolved_before_choosing_image(self):
        self.task.model_revision = 'main'
        from types import SimpleNamespace

        with patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='3' * 40)) as lookup, patch.object(
            docker_runtime, 'build_image', return_value=docker_runtime.ImageInfo(tag=self.image_id),
        ):
            docker_runtime.prepare_image(self.task, '/project')
        lookup.assert_called_once_with(self.task.model_id, revision='main')
        self.assertEqual(self.task.model_revision, '3' * 40)

    def test_unlabelled_legacy_image_cannot_be_reused_as_managed_image(self):
        query = subprocess.CompletedProcess([], 0, stdout=json.dumps({'image_id': self.image_id, 'labels': {}}), stderr='')
        with patch.object(docker_runtime, '_run', return_value=query), self.assertRaisesRegex(RuntimeError, '指纹'):
            docker_runtime.prepare_image(self.task, '/project', reuse_existing=True)

    def test_without_reuse_builds_without_querying_image_store(self):
        built = docker_runtime.ImageInfo(tag=self.image_id)
        with patch.object(docker_runtime, '_run') as docker, patch.object(
            docker_runtime, 'build_image', return_value=built,
        ) as build:
            image = docker_runtime.prepare_image(self.task, '/project')
        self.assertIs(image, built)
        docker.assert_not_called()
        build.assert_called_once_with(self.task, '/project')


if __name__ == '__main__':
    unittest.main()
