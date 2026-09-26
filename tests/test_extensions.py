"""Declaration consumers share routing without loading runtime packages."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from acprof.host.detect import TaskInfo
from acprof.host.task_support import require_task_support


class ExtensionDeclarationTests(unittest.TestCase):
    def test_workload_manifest_is_shared_by_host_and_container_consumers(self):
        from acprof.extensions import CATALOG
        host = CATALOG.select_extension(TaskInfo(
            "local/tiny", "tabular-regression", "structured", "onnxruntime", "onnx", "fixed", "manual",
        ))
        torch = CATALOG.get_extension("structured", "torchscript", task="tabular-regression")
        self.assertEqual(CATALOG.workloads[host.family], torch.workload_entrypoint)
        # Both process roles parse the same declaration without importing its workload.
        for role in ("acprof.host.task_support", "acprof.container.handlers"):
            script = (
                f"import {role}; from acprof.extensions import CATALOG; "
                "print(CATALOG.workloads['structured'])"
            )
            result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                    text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), torch.workload_entrypoint)

    def test_workload_conflict_reports_family_sources_and_is_atomic(self):
        from acprof.extensions import CATALOG, ExtensionCatalog
        catalog = ExtensionCatalog()
        original = CATALOG.get_extension("structured", "torchscript")
        catalog.add(original)
        conflicting = replace(original, extension_id="conflicting", backends=("other",),
                              workload_entrypoint="other_plugin:Generator")
        with self.assertRaisesRegex(ValueError, "structured.*StructuredWorkloadGenerator.*other_plugin"):
            catalog.add(conflicting)
        self.assertNotIn("conflicting", catalog.extensions)
        self.assertEqual(catalog.workloads["structured"], original.workload_entrypoint)

    def test_repeated_identical_manifest_is_idempotent(self):
        from acprof.extensions import load_catalog
        import acprof.extensions
        manifest = Path(acprof.extensions.__file__).parent / "builtin" / "manifest.json"
        once = load_catalog([manifest])
        repeated = load_catalog([manifest, manifest])
        self.assertEqual(once.extensions, repeated.extensions)
        self.assertEqual(once.workloads, repeated.workloads)
        self.assertEqual(once.backend_rules, repeated.backend_rules)

    def test_workload_entrypoint_is_validated_without_import(self):
        from acprof.extensions import CATALOG, ExtensionCatalog
        original = CATALOG.get_extension("structured", "torchscript")
        for value in ("missing-colon", ":Generator", "module:", "module:bad-name", None, 4):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "workload.*entrypoint"):
                ExtensionCatalog().add(replace(original, workload_entrypoint=value))

    def test_metadata_import_is_framework_free(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from acprof.extensions import get_extension; "
             "e=get_extension('structured','onnxruntime',task='tabular-regression'); "
             "assert e.runtime == 'onnxruntime'; "
             "assert not any(x in sys.modules for x in ('torch','onnxruntime','acprof.container.handlers'))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_onnx_declaration_routes_preflight_without_backend_branch(self):
        task = TaskInfo("local/tiny", "tabular-regression", "structured", "onnxruntime", "onnx", "fixed", "manual")
        require_task_support(task)
        self.assertEqual(task.runtime_profile_id, "onnxruntime-cpu")

    def test_backend_task_and_batch_constraints_remain_specific(self):
        from acprof.extensions import UnsupportedExtensionError, get_extension
        with self.assertRaises(UnsupportedExtensionError):
            get_extension("nlp", "sentence_transformers", task="text-generation")
        self.assertEqual(get_extension("cv", "transformers_model", task="image-classification").execution["batch"], "unsupported")
        self.assertEqual(get_extension("diffusion", "diffusers", task="text-to-image").execution["batch"], "available")
        self.assertEqual(get_extension("diffusion", "diffusers", task="image-to-image").execution["batch"], "unsupported")

    def test_model_architecture_selects_adapter_for_another_checkpoint(self):
        from acprof.extensions import select_extension
        task = TaskInfo("someone/checkpoint", "audio-text-to-text", "multimodal", "transformers_model", "transformers", "fixed", "manual", model_config={"model_type": "moss_transcribe_diarize"})
        self.assertEqual(select_extension(task).adapter, "moss-transcribe-diarize")
        self.assertEqual(select_extension(replace(task, model_id="another/checkpoint")).profile, "moss-transformers560")

    def test_new_manifest_adds_backend_without_importing_handler(self):
        from acprof.extensions import load_catalog
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps({"schema_version": 2, "extensions": [{
                "extension_id": "new-runtime", "family": "structured", "runtime": "new-runtime",
                "backends": ["new-runtime"], "tasks": ["tabular-regression"],
                "handler_entrypoint": "not_installed_plugin:Handler", "validation_entrypoint": "not_installed_plugin:validate",
                "environment": "new-cpu", "profile": "new-cpu", "execution": {"cpu": "available", "cuda": "unsupported", "batch": "available", "streaming": "unsupported"},
                "measurement": {"latency": "available"}, "dtypes": ["FP32"], "input_modalities": ["tabular"],
            }]}))
            catalog = load_catalog([path])
            extension = catalog.get_extension("structured", "new-runtime", task="tabular-regression")
            self.assertEqual(extension.environment, "new-cpu")
            self.assertNotIn("not_installed_plugin", sys.modules)
            self.assertEqual(catalog.task_families["tabular-regression"], "structured")

    def test_duplicate_route_or_invalid_status_rejected_at_declaration(self):
        from acprof.extensions import load_catalog
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            entry = {
                "extension_id": "first", "family": "structured", "runtime": "custom",
                "backends": ["custom"], "tasks": ["tabular-regression"],
                "handler_entrypoint": "missing:Handler", "validation_entrypoint": "missing:validate",
                "execution": {"cpu": "available"}, "dtypes": ["FP32"], "input_modalities": ["tabular"],
            }
            duplicate = dict(entry, extension_id="second", handler_entrypoint="other:Handler")
            path.write_text(json.dumps({"schema_version": 2, "extensions": [entry, duplicate]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_catalog([path])
            entry["execution"]["cpu"] = True
            path.write_text(json.dumps({"schema_version": 2, "extensions": [entry]}))
            with self.assertRaisesRegex(ValueError, "status|execution.cpu"):
                load_catalog([path])

    def test_existing_architecture_checkpoint_changes_only_configuration(self):
        from acprof.extensions import select_extension
        task = TaskInfo("owner/checkpoint-a", "text-generation", "nlp", "transformers_model", "transformers", "fixed", "manual", model_config={"model_type": "gpt2"})
        original = select_extension(task)
        second = select_extension(replace(task, model_id="other/checkpoint-b"))
        self.assertEqual(original.extension_id, second.extension_id)
        self.assertEqual(second.handler_entrypoint, "acprof.container.handlers.nlp:NLPHandler")

    def test_new_architecture_manifest_routes_existing_protocol_without_core_edit(self):
        from acprof.extensions import load_catalog
        import acprof.extensions

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps({"schema_version": 2, "architecture_tasks": {
                "NewForCausalLM": "text-classification",
            }, "extensions": [{
                "extension_id": "custom-architecture", "family": "nlp", "runtime": "torch",
                "backends": ["transformers_model"], "tasks": ["text-classification"],
                "adapter": "custom-architecture", "model_types": ["new_architecture"],
                "handler_entrypoint": "external_architecture:Handler", "validation_entrypoint": "external_architecture:validate",
                "execution": {"cpu": "available"}, "dtypes": ["FP32"], "input_modalities": ["text"],
            }]}))
            catalog = load_catalog([*Path(acprof.extensions.__file__).parent.glob("*/manifest.json"), path])
            inferred = catalog.infer_task(["NewForCausalLM"])
            self.assertEqual(inferred, "text-classification")
            task = TaskInfo("other/checkpoint", inferred, "nlp", "transformers_model", "transformers", "fixed", "manual", model_config={"model_type": "new_architecture"})
            selected = catalog.select_extension(task)
            self.assertEqual(selected.handler_entrypoint, "external_architecture:Handler")
            self.assertNotIn("external_architecture", sys.modules)

    def test_same_architecture_selects_declared_backend_and_rejects_unknown_route(self):
        from acprof.extensions import CATALOG, ExtensionCatalog, UnsupportedExtensionError
        catalog = ExtensionCatalog()
        template = CATALOG.get_extension("nlp", "transformers_model")
        for backend in ("first", "second"):
            catalog.add(replace(template, extension_id=backend, backends=(backend,), backend_tasks={},
                                adapter=backend, model_types=("shared_architecture",),
                                model_ids=("owner/shared",), profile=backend + "-profile"))
        task = TaskInfo("owner/shared", "text-generation", "nlp", "second", "custom", "fixed", "manual",
                        model_config={"model_type": "shared_architecture"})
        self.assertEqual(catalog.select_extension(task).extension_id, "second")
        with self.assertRaises(UnsupportedExtensionError):
            catalog.select_extension(replace(task, runtime_backend="undeclared"))


if __name__ == "__main__":
    unittest.main()
