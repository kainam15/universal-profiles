"""按 image ID 合并别名、检查容器引用，并只删除确认过的标签。"""

import json
import os
import subprocess
import unittest
from unittest.mock import patch

from acprof.host.image_management import ImageManagementError, delete_images, list_images


RUNTIME = "sha256:" + "a" * 64
WEIGHTS = "sha256:" + "b" * 64
FINAL = "sha256:" + "c" * 64


def image(image_id, tags, size, layers, model="", labels=None):
    return dict(Id=image_id, RepoTags=tags, Size=size, Created="2026-09-13T00:00:00Z",
                Config=dict(Env=[f"MODEL_ID={model}", "HF_TOKEN=must-not-be-displayed"],
                            Labels=labels or {}), RootFS=dict(Layers=layers))


class DockerFixture:
    def __init__(self):
        self.images = {
            RUNTIME: image(RUNTIME, ["acprof-runtime-audio:env"], 100, ["os", "deps"]),
            WEIGHTS: image(WEIGHTS, ["acprof-weights-audio-demo--model:weights",
                                    "acprof-build-source:" + "b" * 64],
                           400, ["os", "deps", "weights"], "demo/model"),
            FINAL: image(FINAL, ["acprof-audio-demo--model:code"],
                         410, ["os", "deps", "weights", "code"], "demo/model"),
        }
        self.containers = {}
        self.daemon_id = "daemon-original"
        self.commands = []
        self.fail_remove = set()

    def run(self, command, **kwargs):
        self.commands.append(command)
        self.assertions(kwargs)
        args = command[1:]
        if args[:2] == ["context", "show"]:
            return subprocess.CompletedProcess(command, 0, "local-test\n", "")
        if args[:2] != ["--context", "local-test"]:
            raise AssertionError(f"Docker 连接必须固定: {command}")
        args = args[2:]
        output = ""
        if args[0] == "info":
            output = self.daemon_id
        elif args[:2] == ["image", "ls"]:
            # 同一 ID 的多个标签会在 image ls 中重复出现。
            output = "\n".join(key for key, data in self.images.items() for _ in data["RepoTags"] or [None])
        elif args[:2] == ["image", "inspect"]:
            rows = []
            for ref in args[2:]:
                found = self.images.get(ref) or next(
                    (data for data in self.images.values() if ref in data["RepoTags"]), None)
                if found is None:
                    return subprocess.CompletedProcess(command, 1, "", "No such image")
                rows.append(found)
            output = json.dumps(rows)
        elif args[:2] == ["container", "ls"]:
            output = "\n".join(self.containers)
        elif args[:2] == ["container", "inspect"]:
            output = json.dumps([self.containers[key] for key in args[2:]])
        elif args[:2] == ["image", "rm"]:
            if "--no-prune" not in args or "--force" in args or "-f" in args:
                raise AssertionError("必须保留未选择的父镜像，且不强制删除")
            refs = args[3:]
            if any(ref in self.fail_remove for ref in refs):
                return subprocess.CompletedProcess(command, 1, "", "conflict: dependent image")
            for ref in refs:
                for key, data in list(self.images.items()):
                    if ref == key or ref in data["RepoTags"]:
                        if ref in data["RepoTags"]:
                            data["RepoTags"].remove(ref)
                        if ref == key or not data["RepoTags"]:
                            del self.images[key]
                        break
            output = "Untagged: " + "\nUntagged: ".join(refs)
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, output, "")

    @staticmethod
    def assertions(kwargs):
        if kwargs.get("shell"):
            raise AssertionError("Docker 命令不能经过 shell")
        if not kwargs.get("timeout"):
            raise AssertionError("Docker 查询必须有超时")

    @property
    def removals(self):
        return [command for command in self.commands if "rm" in command]


class ImageManagementTests(unittest.TestCase):
    def setUp(self):
        self.docker = DockerFixture()
        patcher = patch("acprof.host.image_management.subprocess.run", side_effect=self.docker.run)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_inventory_groups_aliases_keeps_full_size_and_records_stopped_containers(self):
        self.docker.containers["container-id"] = dict(Image=FINAL, Name="/old-run", State=dict(Status="exited"))
        inventory = list_images()
        self.assertEqual(len(inventory.images), 3)
        indexed = {item.image_id: item for item in inventory.images}
        self.assertEqual(indexed[WEIGHTS].size_bytes, 400)
        self.assertEqual(len(indexed[WEIGHTS].tags), 2)
        self.assertEqual(indexed[WEIGHTS].kind, "weights")
        self.assertEqual(indexed[RUNTIME].kind, "runtime")
        self.assertEqual(indexed[FINAL].kind, "model")
        self.assertIn("old-run", indexed[FINAL].containers[0])
        self.assertEqual(indexed[FINAL].model_id, "demo/model")
        self.assertNotIn("must-not-be-displayed", repr(inventory))
        self.assertFalse(self.docker.removals)

    def test_delete_removes_all_selected_aliases_children_first_and_preserves_runtime(self):
        inventory = list_images()
        outcome = delete_images(inventory, (WEIGHTS, FINAL, WEIGHTS))
        self.assertEqual(len(outcome), 2)
        self.assertTrue(all(item.success for item in outcome))
        self.assertEqual(list(self.docker.images), [RUNTIME])
        commands = self.docker.removals
        self.assertIn("acprof-audio-demo--model:code", commands[0])
        self.assertEqual(set(commands[1][-2:]), {
            "acprof-weights-audio-demo--model:weights", "acprof-build-source:" + "b" * 64})

    def test_retagged_image_aborts_whole_batch_before_any_delete(self):
        inventory = list_images()
        self.docker.images[WEIGHTS]["RepoTags"].append("retained:tag")
        with self.assertRaises(ImageManagementError):
            delete_images(inventory, (FINAL, WEIGHTS))
        self.assertFalse(self.docker.removals)

    def test_new_container_reference_blocks_deletion_including_stopped_container(self):
        inventory = list_images()
        self.docker.containers["new"] = dict(Image=WEIGHTS, Name="/protected", State=dict(Status="exited"))
        with self.assertRaises(ImageManagementError):
            delete_images(inventory, (FINAL, WEIGHTS))
        self.assertFalse(self.docker.removals)

    def test_changed_daemon_and_unknown_selection_never_delete(self):
        inventory = list_images()
        self.docker.daemon_id = "different-daemon"
        with self.assertRaises(ImageManagementError):
            delete_images(inventory, (FINAL,))
        self.docker.daemon_id = inventory.daemon_id
        with self.assertRaises(ImageManagementError):
            delete_images(inventory, ("sha256:" + "d" * 64,))
        self.assertFalse(self.docker.removals)

    def test_partial_failure_is_returned_per_image_and_does_not_force(self):
        inventory = list_images()
        self.docker.fail_remove.add("acprof-audio-demo--model:code")
        outcome = delete_images(inventory, (FINAL, WEIGHTS))
        self.assertEqual([(item.image_id, item.success) for item in outcome], [(FINAL, False), (WEIGHTS, True)])
        self.assertIn("conflict", outcome[0].detail)
        self.assertIn(FINAL, self.docker.images)

    def test_timeout_and_bad_docker_output_are_errors_not_empty_inventory(self):
        for effect in (subprocess.TimeoutExpired("docker", 30), FileNotFoundError("docker")):
            with self.subTest(effect=effect), patch("acprof.host.image_management.subprocess.run", side_effect=effect):
                with self.assertRaises(ImageManagementError):
                    list_images()
        with patch("acprof.host.image_management.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "bad-json", "")):
            with self.assertRaises(ImageManagementError):
                list_images()


if __name__ == "__main__":
    unittest.main()
