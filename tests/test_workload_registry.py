"""Lazy workload discovery, failure diagnostics and legacy factory contracts."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import acprof.workloads as workloads


class LegacyGenerator(workloads.WorkloadGenerator):
    def generate(self, scale_value):
        return {"scale": scale_value, "batch_size": self.batch_size}

    def scale_label(self, scale_value):
        return str(scale_value)


class OtherGenerator(LegacyGenerator):
    pass


class ConfiguredGenerator(LegacyGenerator):
    def __init__(self, model_id, task_type, batch_size, *, factor=1):
        super().__init__(model_id, task_type, batch_size)
        self.factor = factor

    def generate(self, scale_value):
        return {"scale": scale_value * self.factor}


class WorkloadRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = patch.dict(workloads._generators, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    @contextmanager
    def plugin(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            name = "acprof_test_workload_plugin"
            Path(tmp, name + ".py").write_text(textwrap.dedent(source))
            sys.path.insert(0, tmp)
            try:
                yield name
            finally:
                sys.path.remove(tmp)
                sys.modules.pop(name, None)

    def test_discovery_and_listing_do_not_import_implementations_or_frameworks(self):
        script = """
            import sys
            from acprof.extensions import CATALOG
            import acprof.workloads
            forbidden = ('torch', 'transformers', 'onnxruntime', 'numpy', 'PIL',
                         'acprof.workloads.nlp', 'acprof.workloads.cv',
                         'acprof.workloads.audio', 'acprof.workloads.multimodal')
            assert not any(name in sys.modules for name in forbidden), sorted(sys.modules)
            assert CATALOG.workloads['structured'].endswith(':StructuredWorkloadGenerator')
        """
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unselected_missing_dependency_does_not_affect_registered_task(self):
        with self.plugin("import acprof_missing_workload_dependency") as name:
            workloads.register_generator_lazy("broken", name + ":Generator")
            workloads.register_generator("working", LegacyGenerator)
            self.assertEqual(workloads.get_generator("working", "model", "task", 2).generate(3),
                             {"scale": 3, "batch_size": 2})
            self.assertNotIn(name, sys.modules)

    def test_dependency_failure_preserves_cause_and_selected_source(self):
        with self.plugin("import acprof_missing_workload_dependency") as name:
            workloads.register_generator_lazy("broken", name + ":Generator")
            with self.assertRaises(workloads.WorkloadDependencyMissingError) as caught:
                workloads.get_generator("broken", "model", "task", 1)
            self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)
            self.assertEqual(caught.exception.__cause__.name, "acprof_missing_workload_dependency")
            self.assertIn("broken", str(caught.exception))
            self.assertIn(name, str(caught.exception))

    def test_missing_declared_module_is_import_failure_not_dependency_failure(self):
        workloads.register_generator_lazy("broken", "acprof_missing_declared_module:Generator")
        with self.assertRaises(workloads.WorkloadModuleImportError) as caught:
            workloads.get_generator("broken", "model", "task", 1)
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_import_error_keeps_original_exception_chain(self):
        with self.plugin("raise ImportError('incompatible plugin API')") as name:
            workloads.register_generator_lazy("broken", name + ":Generator")
            with self.assertRaises(workloads.WorkloadModuleImportError) as caught:
                workloads.get_generator("broken", "model", "task", 1)
            self.assertIsInstance(caught.exception.__cause__, ImportError)
            self.assertIn("incompatible plugin API", str(caught.exception.__cause__))

    def test_partial_self_registration_cannot_cache_a_failed_module_as_success(self):
        with self.plugin("""
            from acprof.workloads import WorkloadGenerator, register_generator
            class Generator(WorkloadGenerator):
                def generate(self, scale): return {'scale': scale}
                def scale_label(self, scale): return str(scale)
            register_generator('broken', Generator)
            raise RuntimeError('module initialization failed')
        """) as name:
            workloads.register_generator_lazy("broken", name + ":Generator")
            for _ in range(2):
                with self.assertRaises(workloads.WorkloadModuleImportError) as caught:
                    workloads.get_generator("broken", "model", "task", 1)
                self.assertIsInstance(caught.exception.__cause__, RuntimeError)
                self.assertIn("module initialization failed", str(caught.exception))

    def test_conflict_does_not_overwrite_original_and_identifies_both_sources(self):
        workloads.register_generator("example", LegacyGenerator)
        with self.assertRaisesRegex(ValueError, "example.*LegacyGenerator.*OtherGenerator"):
            workloads.register_generator("example", OtherGenerator)
        self.assertIsInstance(workloads.get_generator("example", "model", "task", 1), LegacyGenerator)

    def test_same_registration_and_same_lazy_declaration_are_idempotent(self):
        workloads.register_generator("example", LegacyGenerator)
        workloads.register_generator("example", LegacyGenerator)
        source = LegacyGenerator.__module__ + ":LegacyGenerator"
        workloads.register_generator_lazy("example", source)
        workloads.register_generator_lazy("other", source)
        workloads.register_generator_lazy("other", source)
        self.assertIsInstance(workloads.get_generator("other", "model", "task", 1), LegacyGenerator)

    def test_entrypoint_can_export_an_existing_class_under_a_local_alias(self):
        with self.plugin(f"from {__name__} import LegacyGenerator as Generator") as name:
            workloads.register_generator_lazy("alias", name + ":Generator")
            self.assertIsInstance(workloads.get_generator("alias", "model", "task", 1), LegacyGenerator)
            workloads.register_generator_lazy("alias", name + ":Generator")
            workloads.register_generator("alias", LegacyGenerator)

    def test_loaded_class_identity_cannot_be_replaced_by_same_named_class(self):
        source = LegacyGenerator.__module__ + ':LegacyGenerator'
        workloads.register_generator_lazy('example', source)
        workloads.get_generator('example', 'model', 'task', 1)
        replacement = type('LegacyGenerator', (LegacyGenerator,), {'__module__': LegacyGenerator.__module__})
        with self.assertRaises(workloads.DuplicateWorkloadRegistrationError):
            workloads.register_generator('example', replacement)

    def test_legacy_constructor_and_adapter_default_remain_compatible(self):
        workloads.register_generator("legacy", LegacyGenerator)
        generator = workloads.get_generator("legacy", "model", "task", 3,
                                            model_adapter="family-default")
        self.assertEqual(generator.generate(2), {"scale": 2, "batch_size": 3})

    def test_new_task_can_consume_options_without_central_family_branch(self):
        workloads.register_generator("new-task", ConfiguredGenerator)
        generator = workloads.get_generator("new-task", "model", "task", 1, factor=4)
        self.assertEqual(generator.generate(3), {"scale": 12})

    def test_bad_options_and_unknown_registration_have_distinct_errors(self):
        workloads.register_generator("example", ConfiguredGenerator)
        with self.assertRaises(workloads.WorkloadConfigurationError) as caught:
            workloads.get_generator("example", "model", "task", 1, misspelled=3)
        self.assertIsInstance(caught.exception.__cause__, TypeError)
        with self.assertRaises(workloads.WorkloadNotRegisteredError):
            workloads.get_generator("missing", "model", "task", 1)

    def test_manifest_extension_loads_new_task_without_module_self_registration(self):
        from acprof.extensions import load_catalog
        with self.plugin("""
            from acprof.workloads import WorkloadGenerator
            class Generator(WorkloadGenerator):
                def generate(self, scale):
                    return {'custom_values': [scale] * self.batch_size}
                def scale_label(self, scale):
                    return 'custom' + str(scale)
        """) as name, tempfile.TemporaryDirectory() as tmp:
            entry = {
                "extension_id": "new-extension", "family": "new-task", "runtime": "custom",
                "backends": ["custom"], "tasks": ["custom-task"],
                "handler_entrypoint": name + ":Handler", "validation_entrypoint": name + ":validate",
                "workload_entrypoint": name + ":Generator", "execution": {"cpu": "available"},
                "dtypes": ["FP32"], "input_modalities": ["custom"],
            }
            manifest = Path(tmp, "manifest.json")
            manifest.write_text(json.dumps({"schema_version": 1, "extensions": [entry]}))
            catalog = load_catalog([manifest])
            self.assertNotIn(name, sys.modules)
            workloads.register_catalog(catalog)
            workloads.register_catalog(catalog)
            generator = workloads.get_generator("new-task", "model", "custom-task", 2)
            self.assertEqual(generator.generate(5), {"custom_values": [5, 5]})


if __name__ == "__main__":
    unittest.main()
