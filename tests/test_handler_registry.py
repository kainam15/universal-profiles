"""Registry isolation and actionable selected-backend failures."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from acprof.container import handlers


class HandlerRegistryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(handlers.HandlerRegistry._handlers, clear=True))
        self.stack.enter_context(patch.dict(handlers.HandlerRegistry._adapters, clear=True))
        self.stack.enter_context(patch.dict(os.environ, {"ACPROF_MODEL_ADAPTER": "family-default"}))

    def test_normal_registration_and_duplicate_rejected_before_construction(self):
        class First:
            pass

        class Replacement:
            def __init__(self):
                self.fail = "must not be constructed on duplicate"

        handlers.HandlerRegistry.register("test", "runtime", First)
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.register("test", "runtime", Replacement)
        self.assertEqual(type(caught.exception).__name__, "DuplicateHandlerRegistrationError")
        for detail in ("test:runtime", "First", "Replacement", __name__):
            self.assertIn(detail, str(caught.exception))
        self.assertIsInstance(handlers.HandlerRegistry.get("test", "runtime"), First)

    def test_explicit_override_and_adapter_duplicates(self):
        class First:
            pass

        class Second:
            pass

        handlers.HandlerRegistry.register("test", "runtime", First)
        handlers.HandlerRegistry.register("test", "runtime", Second, override=True)
        self.assertIsInstance(handlers.HandlerRegistry.get("test", "runtime"), Second)
        handlers.HandlerRegistry.register_adapter("a", "test", "runtime", First)
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.register_adapter("a", "test", "runtime", Second)
        self.assertEqual(type(caught.exception).__name__, "DuplicateHandlerRegistrationError")

    def test_package_import_does_not_import_optional_handlers(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import acprof.container.handlers; "
             "assert 'acprof.container.handlers.nlp' not in sys.modules; "
             "assert 'acprof.container.handlers.audio' not in sys.modules; "
             "assert 'torch' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def _module(self, source):
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        name = "acprof_registry_fixture"
        Path(directory, name + ".py").write_text(source, encoding="utf-8")
        self.stack.enter_context(patch.object(sys, "path", [directory, *sys.path]))
        self.addCleanup(sys.modules.pop, name, None)
        handlers.HandlerRegistry.register_lazy("test", "optional", name + ":Handler")
        self.assertNotIn(name, sys.modules)
        return name

    def test_lazy_load_constructs_once(self):
        self._module("class Handler: pass\n")
        first = handlers.HandlerRegistry.get("test", "optional")
        self.assertIs(first, handlers.HandlerRegistry.get("test", "optional"))

    def test_unselected_missing_dependency_does_not_block_registration(self):
        self._module("import acprof_missing_optional_dependency\nclass Handler: pass\n")
        handlers.HandlerRegistry.register("other", "available", object)
        self.assertIsInstance(handlers.HandlerRegistry.get("other", "available"), object)

    def test_selected_missing_dependency_preserves_original_exception(self):
        name = self._module("import acprof_missing_optional_dependency\nclass Handler: pass\n")
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.get("test", "optional")
        self.assertEqual(type(caught.exception).__name__, "HandlerDependencyMissingError")
        for detail in ("optional", name, "acprof_missing_optional_dependency", "ModuleNotFoundError"):
            self.assertIn(detail, str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_internal_module_failure_is_distinct_from_missing_dependency(self):
        self._module("raise RuntimeError('module defect')\n")
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.get("test", "optional")
        self.assertEqual(type(caught.exception).__name__, "HandlerModuleImportError")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_handler_constructor_failure_is_distinct(self):
        self._module("class Handler:\n def __init__(self): raise RuntimeError('constructor defect')\n")
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.get("test", "optional")
        self.assertEqual(type(caught.exception).__name__, "HandlerInitializationError")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_unknown_handler_is_distinct(self):
        with self.assertRaises(ValueError) as caught:
            handlers.HandlerRegistry.get("test", "unknown")
        self.assertEqual(type(caught.exception).__name__, "HandlerNotRegisteredError")

    def test_selected_handler_load_missing_dependency_is_actionable(self):
        class Handler:
            def load(self, *_args):
                import acprof_missing_runtime_dependency

        with self.assertRaises(ValueError) as caught:
            handlers.load_handler(Handler(), "/model", "task", "optional", "cpu")
        self.assertEqual(type(caught.exception).__name__, "HandlerDependencyMissingError")
        self.assertIn("acprof_missing_runtime_dependency", str(caught.exception))
        self.assertIn("optional", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_selected_handler_load_failure_preserves_exception_chain(self):
        class Handler:
            def load(self, *_args):
                raise RuntimeError("corrupt model")

        with self.assertRaises(ValueError) as caught:
            handlers.load_handler(Handler(), "/model", "task", "optional", "cpu")
        self.assertEqual(type(caught.exception).__name__, "HandlerInitializationError")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_manifest_validator_is_invoked_and_original_task_error_preserved(self):
        self._module("def validate(*args):\n return {'sentinel': args[1]['value']}\n"
                     "def fail(*args):\n raise ValueError('task sanity failed')\n"
                     "not_callable = 7\n")
        context = {"_validation_entrypoint": "acprof_registry_fixture:validate"}
        result = handlers.BaseHandler.validate_output(None, context, {"value": 42}, {}, {}, {})
        self.assertEqual(result, {"sentinel": 42})
        context["_validation_entrypoint"] = "acprof_registry_fixture:fail"
        with self.assertRaisesRegex(ValueError, "task sanity failed"):
            handlers.BaseHandler.validate_output(None, context, {}, {}, {}, {})
        context["_validation_entrypoint"] = "acprof_registry_fixture:not_callable"
        with self.assertRaisesRegex(TypeError, "callable"):
            handlers.BaseHandler.validate_output(None, context, {}, {}, {}, {})


if __name__ == "__main__":
    unittest.main()
