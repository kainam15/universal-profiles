import importlib
import sys
import types
import unittest
from unittest.mock import patch


class ComputeProfileRunnerITTTests(unittest.TestCase):
    def _import_runner(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                synchronize=lambda: None,
                nvtx=types.SimpleNamespace(
                    range_push=lambda *_args, **_kwargs: None,
                    range_pop=lambda *_args, **_kwargs: None,
                ),
            ),
            profiler=types.SimpleNamespace(
                ProfilerActivity=types.SimpleNamespace(CPU=object()),
                profile=lambda *_args, **_kwargs: None,
            ),
            inference_mode=lambda: None,
            set_num_threads=lambda *_args, **_kwargs: None,
        )
        fake_handlers = types.ModuleType("acprof.container.handlers")
        fake_handlers.HandlerRegistry = types.SimpleNamespace(
            get=lambda *_args, **_kwargs: None
        )
        sys.modules.pop("acprof.container.compute_profile_runner", None)
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "acprof.container.handlers": fake_handlers},
        ):
            return importlib.import_module("acprof.container.compute_profile_runner")

    def test_itt_control_prefers_advisor_injected_collector_environment(self):
        runner = self._import_runner()
        loaded = []
        collector = "/opt/intel/oneapi/advisor/2025.5/lib64/runtime/libittnotify_collector.so"

        class FakeLib:
            def __init__(self):
                setattr(self, "__itt_resume", lambda: None)
                setattr(self, "__itt_pause", lambda: None)

        def fake_cdll(name):
            loaded.append(name)
            if name == collector:
                return FakeLib()
            raise OSError("unexpected ITT library")

        with patch.dict(runner.os.environ, {"INTEL_LIBITTNOTIFY64": collector}, clear=False), \
             patch.object(runner.os.path, "exists", side_effect=lambda path: path == collector), \
             patch.object(runner.glob, "glob", return_value=[]), \
             patch.object(runner.ctypes, "CDLL", side_effect=fake_cdll):
            control = runner._ITTControl()

        self.assertIsNotNone(control._lib)
        self.assertIn(collector, loaded)
        self.assertEqual(loaded[0], collector)

    def test_itt_control_invokes_literal_itt_symbols(self):
        runner = self._import_runner()
        collector = "/tmp/libittnotify_collector.so"
        calls = []

        class FakeLib:
            def __init__(self):
                setattr(self, "__itt_resume", lambda: calls.append("resume"))
                setattr(self, "__itt_pause", lambda: calls.append("pause"))

        with patch.dict(runner.os.environ, {"ADVISOR_ITT_LIB": collector}, clear=False), \
             patch.object(runner.os.path, "exists", side_effect=lambda path: path == collector), \
             patch.object(runner.glob, "glob", return_value=[]), \
             patch.object(runner.ctypes, "CDLL", return_value=FakeLib()):
            control = runner._ITTControl()
            control.resume()
            control.pause()

        self.assertEqual(calls, ["resume", "pause"])


if __name__ == "__main__":
    unittest.main()
