"""按 image ID 合并别名、检查容器引用，并只删除确认过的标签。"""

import json
import os
import subprocess
import unittest
from dataclasses import replace
from unittest.mock import patch

from acprof.host.image_management import ImageManagementError, delete_images, list_images
from acprof.host.image_graph import reclaimable_image_bytes


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
        self.layer_sizes = {"os": 20, "deps": 80, "weights": 300, "code": 10}
        self.history_overrides = {}
        self.fail_space_query = False
        self.space_rows = []

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
        elif args[:2] == ["image", "history"]:
            data = self.images[args[-1]]
            sizes = self.history_overrides.get(args[-1], [0, *reversed([
                self.layer_sizes.get(layer, 1) for layer in data["RootFS"]["Layers"]])])
            output = "\n".join(json.dumps(size if isinstance(size, dict) else {
                "size": size, "command": "COPY /app" if size else "ENV A=1"}) for size in sizes)
        elif args[:2] == ["system", "df"]:
            if self.fail_space_query:
                return subprocess.CompletedProcess(command, 1, "", "space query unavailable")
            output = json.dumps(self.space_rows)
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


def dependency_images(docker, profile="moss-transformers560"):
    from pathlib import Path
    from acprof.host.dependency_images import platform_fingerprint
    from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS, environment_id

    root = Path(__file__).resolve().parents[1]
    labels = {"org.acprof.platform": "cu128",
              "org.acprof.platform-build-fingerprint": platform_fingerprint(PLATFORMS["cu128"])}
    docker.images[RUNTIME] = image(RUNTIME, ["acprof-platform-cu128:test"], 100, ["os", "deps"],
                                        labels={**labels, "org.acprof.image-kind": "platform"})
    env = {**labels, "org.acprof.image-kind": "environment",
           "org.acprof.environment": environment_id(ENVIRONMENTS[profile], root),
           "org.acprof.environment-build-fingerprint": "environment-key"}
    docker.images[WEIGHTS] = image(WEIGHTS, ["acprof-runtime-env:test"], 400,
                                        ["os", "deps", "weights"], labels=env)
    docker.images[FINAL] = image(FINAL, ["acprof-weights-multimodal-demo--model:test"], 410,
                                      ["os", "deps", "weights", "code"], "demo/model",
                                      {**env, "org.acprof.image-kind": "weights"})
    docker.history_overrides[RUNTIME] = [
        {"size": 80, "command": "RUN /bin/sh -c python /build/environment_tools.py platform # buildkit"},
        {"size": 0, "command": "COPY requirements.lock expectation.json /opt/acprof/ # buildkit"},
        {"size": 20, "command": "RUN /bin/sh -c python /build/environment_tools.py system /opt/acprof/system.lock # buildkit"},
        {"size": 0, "command": "COPY system.lock /opt/acprof/system.lock # buildkit"},
    ]
    docker.history_overrides[WEIGHTS] = [
        {"size": 300, "command": "RUN |2 ENVIRONMENT_ID=id ENVIRONMENT_BUILD_FINGERPRINT=key /bin/sh -c python /build/environment_tools.py environment # buildkit"},
        {"size": 0, "command": "COPY requirements.lock expectation.json /opt/acprof/ # buildkit"},
        *docker.history_overrides[RUNTIME],
    ]
    docker.history_overrides[FINAL] = [
        {"size": 10, "command": 'RUN /bin/sh -c if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; python /opt/acprof/download_model.py # buildkit'},
        {"size": 0, "command": "COPY acprof/container/download_model.py acprof/container/model_files.py /opt/acprof/ # buildkit"},
        *docker.history_overrides[WEIGHTS],
    ]


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

    def test_content_labels_identify_all_four_layers_without_profile_names(self):
        self.docker.images = {
            "sha256:" + str(index) * 64: image("sha256:" + str(index) * 64, [f"custom-cache:{index}"], 100, [],
                                              labels={"org.acprof.image-kind": kind})
            for index, kind in enumerate(("platform", "environment", "weights", "model"))
        }
        kinds = {item.image_id: item.kind for item in list_images().images}
        self.assertEqual(kinds, {"sha256:" + str(index) * 64: kind
                                 for index, kind in enumerate(("base", "runtime", "weights", "model"))})

    def test_tree_distinguishes_inherited_added_shared_and_unique_bytes(self):
        items = {item.image_id: item for item in list_images().images}
        final, weights = items[FINAL], items[WEIGHTS]
        self.assertEqual(getattr(final, "parent_id", None), WEIGHTS)
        self.assertEqual(final.parent_source, "layer-prefix")
        self.assertEqual((final.inherited_bytes, final.added_bytes), (400, 10))
        self.assertEqual((weights.inherited_bytes, weights.added_bytes), (100, 300))
        self.assertEqual((weights.shared_bytes, weights.unique_bytes), (400, 0))
        self.assertEqual((final.shared_bytes, final.unique_bytes), (400, 10))
        self.assertEqual(final.ancestor_ids, (RUNTIME, WEIGHTS))
        self.assertEqual(items[RUNTIME].descendant_ids, (WEIGHTS, FINAL))

    def test_explicit_missing_parent_and_ambiguous_prefix_are_not_invented(self):
        missing = "sha256:" + "d" * 64
        self.docker.images[FINAL]["Config"]["Env"].append("ACPROF_MODEL_IMAGE_ID=" + missing)
        self.docker.images[FINAL]["Config"]["Labels"]["org.acprof.image-kind"] = "model"
        items = {item.image_id: item for item in list_images().images}
        self.assertEqual(getattr(items[FINAL], "parent_id", None), missing)
        self.assertIsNone(items[FINAL].added_bytes)
        self.assertEqual(items[FINAL].parent_source, "missing")
        del self.docker.images[FINAL]["Config"]["Env"][-1]
        self.docker.images[missing] = image(missing, ["duplicate:weights"], 400,
                                            ["os", "deps", "weights"])
        final = next(item for item in list_images().images if item.image_id == FINAL)
        self.assertEqual(final.parent_id, "")
        self.assertEqual(final.parent_source, "ambiguous")

    def test_layer_references_count_image_ids_and_preserve_chain_identity(self):
        other = "sha256:" + "d" * 64
        self.docker.images[other] = image(other, ["unrelated:one", "unrelated:alias"], 310,
                                           ["weights", "code"])
        inventory = list_images()
        self.assertTrue(getattr(inventory, "layers", ()), "应提供每层及其引用镜像")
        code_layers = [layer for layer in inventory.layers if layer.diff_id == "code"]
        self.assertEqual(len(code_layers), 2, "同 diff ID 在不同前缀上不能冒充相同的实体层链")
        self.assertEqual({layer.image_ids for layer in code_layers}, {(FINAL,), (other,)})
        os_layer = next(layer for layer in inventory.layers if layer.diff_id == "os")
        self.assertEqual(set(os_layer.image_ids), {RUNTIME, WEIGHTS, FINAL})

    def test_incomplete_history_is_unknown_and_does_not_prevent_inventory(self):
        self.docker.history_overrides[FINAL] = [0, 0, 300, 80, 20]
        self.docker.fail_space_query = True
        final = next(item for item in list_images().images if item.image_id == FINAL)
        self.assertEqual(getattr(final, "parent_id", None), WEIGHTS)
        self.assertIsNone(final.unique_bytes)
        self.assertEqual(final.added_bytes, 10)

    def test_zero_byte_filesystem_layer_is_distinct_from_metadata_history(self):
        self.docker.images[FINAL]["RootFS"]["Layers"].append("empty-workdir")
        self.docker.history_overrides[FINAL] = [
            {"size": 0, "command": "WORKDIR /app"}, 0, 10, 300, 80, 20]
        inventory = list_images()
        layer = next(layer for layer in inventory.layers if layer.diff_id == "empty-workdir")
        self.assertEqual(layer.size_bytes, 0)
        self.assertEqual(reclaimable_image_bytes(inventory, (FINAL,)), 10)

    def test_batch_reclaim_deduplicates_layers_and_excludes_retained_references(self):
        inventory = list_images()
        self.assertEqual(reclaimable_image_bytes(inventory, (WEIGHTS,)), 0)
        self.assertEqual(reclaimable_image_bytes(inventory, (FINAL,)), 10)
        self.assertEqual(reclaimable_image_bytes(inventory, (FINAL, WEIGHTS)), 310)
        self.assertEqual(reclaimable_image_bytes(inventory, (FINAL, WEIGHTS, RUNTIME)), 410)
        used = replace(inventory, images=tuple(replace(item, containers=("kept (exited)",))
                                                if item.image_id == FINAL else item for item in inventory.images))
        self.assertEqual(reclaimable_image_bytes(used, (FINAL,)), 0)

    def test_unknown_history_can_use_docker_df_approximation_without_inventing_layer_sizes(self):
        self.docker.history_overrides[FINAL] = [0, 10]
        self.docker.space_rows = [{"ID": FINAL, "SharedSize": "400B", "UniqueSize": "10B"}]
        inventory = list_images()
        final = next(item for item in inventory.images if item.image_id == FINAL)
        self.assertEqual((final.shared_bytes, final.unique_bytes, final.space_source), (400, 10, "docker-df"))
        self.assertIsNone(next(layer for layer in inventory.layers if layer.diff_id == "code").size_bytes)
        self.assertEqual(reclaimable_image_bytes(inventory, (FINAL,)), 10)
        self.assertIsNone(reclaimable_image_bytes(inventory, (FINAL, WEIGHTS)))

    def test_runtime_display_names_require_matching_environment_identity(self):
        from acprof.runtime_profiles import ENVIRONMENTS, environment_id
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        self.docker.images[RUNTIME]["RepoTags"] = ["acprof-runtime-env:opaque"]
        self.docker.images[RUNTIME]["Config"]["Labels"] = {
            "org.acprof.image-kind": "environment", "org.acprof.platform": "cpu",
            "org.acprof.environment": environment_id(ENVIRONMENTS["nlp-cpu"], root),
        }
        runtime = next(item for item in list_images().images if item.image_id == RUNTIME)
        self.assertIn("nlp-cpu", runtime.display_name)
        self.docker.images[RUNTIME]["Config"]["Labels"]["org.acprof.environment"] = "unknown-environment"
        runtime = next(item for item in list_images().images if item.image_id == RUNTIME)
        self.assertNotIn("nlp", runtime.display_name)
        self.assertIn("env-unknown", runtime.display_name)

    def test_dependencies_show_locked_versions_and_exclude_inherited_packages(self):
        dependency_images(self.docker)
        indexed = {item.image_id: item for item in list_images().images}
        platform, runtime, weights = (indexed[key] for key in (RUNTIME, WEIGHTS, FINAL))
        self.assertEqual(getattr(platform, "dependency_source", ""), "platform-lock")
        self.assertIn(("torch", "2.11.0+cu128"), platform.python_dependencies)
        self.assertIn(("build-essential:amd64", "12.12"), platform.system_dependencies)
        self.assertNotIn("base-files:amd64", dict(platform.system_dependencies), "不能把系统基础镜像已有包算成本层安装制品")
        self.assertEqual(runtime.dependency_source, "environment-lock")
        self.assertIn(("transformers", "5.6.0"), runtime.python_dependencies)
        self.assertIn(("torchaudio", "2.11.0+cu128"), runtime.python_dependencies)
        self.assertNotIn("torch", dict(runtime.python_dependencies))
        self.assertEqual(runtime.system_dependencies, ())
        self.assertEqual(weights.dependency_source, "inherited")
        self.assertEqual(weights.python_dependencies, ())
        self.assertFalse(any("run" in cmd or "create" in cmd for cmd in self.docker.commands))

    def test_dependencies_never_use_current_profile_for_stale_or_unverified_images(self):
        dependency_images(self.docker)
        self.docker.images[RUNTIME]["Config"]["Labels"]["org.acprof.platform-build-fingerprint"] = "old-platform"
        self.docker.images[WEIGHTS]["Config"]["Labels"]["org.acprof.environment"] = "old-environment"
        for item in list_images().images:
            self.assertEqual(getattr(item, "dependency_source", ""), "unknown")
            self.assertEqual(item.python_dependencies, ())
        dependency_images(self.docker)
        # 继承已知标签后另装包的自定义镜像不能显示旧锁或“无新增”。
        self.docker.history_overrides[FINAL].insert(0, {"size": 0, "command": "RUN pip install private-package"})
        final = next(item for item in list_images().images if item.image_id == FINAL)
        self.assertEqual(final.dependency_source, "unknown")

    def test_dependencies_handle_missing_lock_and_failed_history_without_losing_inventory(self):
        dependency_images(self.docker)
        with patch("acprof.runtime_profiles.platform_identity", side_effect=OSError("lock missing")):
            inventory = list_images()
        self.assertEqual(len(inventory.images), 3)
        self.assertTrue(all(getattr(item, "dependency_source", "") == "unknown" for item in inventory.images))
        self.docker.history_overrides[WEIGHTS] = [{"size": 0, "command": None}]
        indexed = {item.image_id: item for item in list_images().images}
        self.assertEqual(indexed[WEIGHTS].dependency_source, "unknown")
        self.assertEqual(indexed[FINAL].dependency_source, "unknown")

    def test_build_identity_beats_similar_prefix_and_conflicting_parent_is_rejected(self):
        self.docker.images[RUNTIME]["Config"]["Labels"] = {"org.acprof.environment-build-fingerprint": "env-fingerprint"}
        self.docker.images[WEIGHTS]["Config"]["Labels"] = {"org.acprof.environment-build-fingerprint": "env-fingerprint"}
        inventory = list_images()
        weights = next(item for item in inventory.images if item.image_id == WEIGHTS)
        self.assertEqual((weights.parent_id, weights.parent_source), (RUNTIME, "metadata"))
        self.docker.images[WEIGHTS]["Parent"] = FINAL
        weights = next(item for item in list_images().images if item.image_id == WEIGHTS)
        self.assertEqual((weights.parent_id, weights.parent_source), ("", "conflict"))

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
