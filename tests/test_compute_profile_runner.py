import importlib
import io
import json
import sys
import types
import unittest
import weakref
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stdout
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
        fake_handlers.resolve_model_source = (
            lambda model_id, model_path=None: model_path or model_id
        )
        fake_handlers.load_handler = lambda handler, *args, **kwargs: handler.load(*args, **kwargs)
        sys.modules.pop("acprof.container.compute_profile_runner", None)
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "acprof.container.handlers": fake_handlers},
        ):
            runner = importlib.import_module("acprof.container.compute_profile_runner")
        runner.torch = fake_torch
        return runner

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

    def test_eager_load_option_is_isolated_from_vendor_modes(self):
        runner = self._import_runner()

        self.assertEqual(
            runner._load_options_for_profile_mode("torch_eager_cpu"),
            {"attention_implementation": "eager"},
        )
        self.assertEqual(
            runner._load_options_for_profile_mode("torch_eager_gpu"),
            {"attention_implementation": "eager"},
        )
        self.assertIsNone(runner._load_options_for_profile_mode("cpu"))
        self.assertIsNone(runner._load_options_for_profile_mode("gpu"))

    def test_eager_attention_verification_reads_loaded_model_config(self):
        runner = self._import_runner()
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager")
        )
        model_ctx = {
            "pipeline": types.SimpleNamespace(model=model),
        }

        self.assertEqual(runner._verify_eager_attention(model_ctx), "eager")

    def test_eager_attention_verification_rejects_non_eager_model(self):
        runner = self._import_runner()
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="sdpa")
        )
        model_ctx = {
            "pipeline": types.SimpleNamespace(model=model),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "expected=eager,actual=sdpa",
        ):
            runner._verify_eager_attention(model_ctx)

    @contextmanager
    def _main_context(self, mode, events, *, invalid_output=False):
        runner = self._import_runner()
        output = object()
        model_ctx = {"model": types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager"),
        )}

        class Handler:
            def load(self, *_args, **_kwargs):
                events.append("load")
                return model_ctx

            def preprocess(self, *_args):
                events.append("preprocess")
                return {"input": "prepared"}

            def predict(self, *_args):
                events.append("predict")
                return output

            def postprocess(self, _ctx, generated):
                if generated is not output:
                    raise AssertionError("postprocess must validate the actual warmup output")
                events.append("postprocess")
                if invalid_output:
                    raise ValueError("pipeline video frame count differs from requested num_frames")
                return {"output_type": "video", "video_frame_count": 17}

            def validate_output(self, *_args):
                raise AssertionError('full validation must run in the separate preflight process')

        @contextmanager
        def profiler(*_args, **_kwargs):
            events.append("capture_start")
            try:
                yield types.SimpleNamespace(
                    key_averages=lambda: [types.SimpleNamespace(flops=32)],
                )
            finally:
                events.append("capture_end")

        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"torch": runner.torch}))
            execution = types.SimpleNamespace(inference_context=nullcontext, synchronize=lambda: None)
            stack.enter_context(patch.object(runner, "configured_execution", return_value=(execution, "cpu")))
            stack.enter_context(patch.dict(runner.os.environ, {"MODEL_ID": "local/tiny"}, clear=True))
            stack.enter_context(patch.object(sys, "argv", [
                "compute_profile_runner", "--payload-file", "unused.json",
                "--input-scale", "64", "--repeat", "2", "--profile-mode", mode,
            ]))
            stack.enter_context(patch.object(runner.HandlerRegistry, "get", return_value=Handler()))
            stack.enter_context(patch.object(runner, "_find_payload", return_value={"resolution": 64}))
            stack.enter_context(patch.object(runner.torch, "inference_mode", side_effect=nullcontext))
            stack.enter_context(patch.object(runner.torch.profiler, "profile", side_effect=profiler))
            stack.enter_context(patch.object(runner, "_ITTControl", return_value=types.SimpleNamespace(
                resume=lambda: events.append("capture_start"),
                pause=lambda: events.append("capture_end"),
            )))
            stack.enter_context(patch.object(runner.torch.cuda.nvtx, "range_push", side_effect=lambda *_args: events.append("capture_start")))
            stack.enter_context(patch.object(runner.torch.cuda.nvtx, "range_pop", side_effect=lambda: events.append("capture_end")))
            stack.enter_context(redirect_stdout(stdout))
            yield runner, stdout

    def test_invalid_warmup_output_prevents_all_profiler_capture_and_success(self):
        for mode in ("cpu", "gpu", "torch_eager_cpu"):
            with self.subTest(mode=mode):
                events = []
                with self._main_context(mode, events, invalid_output=True) as (runner, stdout):
                    with self.assertRaisesRegex(ValueError, "frame count differs"):
                        runner.main()
                self.assertEqual(events, ["load", "preprocess", "predict", "postprocess"])
                self.assertEqual(stdout.getvalue(), "")

    def test_warmup_postprocess_does_not_repeat_full_validation_under_profiler(self):
        for mode in ("cpu", "gpu", "torch_eager_cpu"):
            with self.subTest(mode=mode):
                events = []
                with self._main_context(mode, events) as (runner, stdout):
                    runner.main()
                self.assertEqual(events, [
                    "load", "preprocess", "predict", "postprocess",
                    "capture_start", "predict", "predict", "capture_end",
                ])
                result = json.loads(stdout.getvalue())
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["repeat"], 2)

    def test_completed_outputs_are_released_before_the_next_profiled_request(self):
        class Output:
            pass

        for mode in ("cpu", "gpu", "torch_eager_cpu", "torch_eager_gpu"):
            with self.subTest(mode=mode):
                events, references = [], []
                with self._main_context(mode, events) as (runner, stdout):
                    handler = runner.HandlerRegistry.get()

                    def predict(*_args):
                        if references:
                            self.assertIsNone(references[-1](), "previous output survives into the next inference")
                        result = Output()
                        references.append(weakref.ref(result))
                        events.append("predict")
                        return result

                    def complete(_context, output, *, timeout_s):
                        self.assertIs(output, references[-1]())
                        events.append("complete")
                        return output

                    def postprocess(_context, output):
                        self.assertIs(output, references[-1]())
                        self.assertEqual(events[-1], "complete")
                        events.append("postprocess")
                        return {}

                    handler.predict = predict
                    handler.postprocess = postprocess
                    runner.configured_execution.return_value[0].wait_for_completion = complete
                    runner.main()
                self.assertEqual(len(references), 3)
                self.assertTrue(all(reference() is None for reference in references))
                self.assertEqual(events, [
                    "load", "preprocess", "predict", "complete", "postprocess",
                    "capture_start", "predict", "complete", "predict", "complete", "capture_end",
                ])
                self.assertEqual(json.loads(stdout.getvalue())["repeat"], 2)


if __name__ == "__main__":
    unittest.main()
