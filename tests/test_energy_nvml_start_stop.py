import math
import unittest
from unittest.mock import patch

from acprof.monitors import energy_nvml


class FakeThread:
    instances = []

    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined = False
        FakeThread.instances.append(self)

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joined = True

    def is_alive(self):
        return self.started and not self.joined


class GPUEnergyMonitorStartStopTests(unittest.TestCase):

    def test_apply_control_baseline_uses_integrated_average_and_records_method(self) -> None:
        with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit"), patch(
            "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
            return_value="handle",
        ), patch(
            "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetName",
            return_value=b"Test GPU",
        ):
            monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=2.0)

        samples = [(0.0, 10.0), (1.0, 30.0), (3.0, 30.0)]
        result = energy_nvml._result_from_samples(samples, idle_power_w=float("nan"))
        idle_power_w = monitor.apply_control_baseline(result, samples, trace=True)

        self.assertAlmostEqual(idle_power_w, 80.0 / 3.0)
        self.assertEqual(
            monitor.idle_trace["gpu_idle_baseline_method"],
            "matched_control_time_weighted_mean",
        )
        self.assertEqual(
            monitor.idle_trace["gpu_idle_trace_schema"],
            "nvml_gpu_control_v1",
        )


    def test_start_stop_samples_and_calculates_energy(self) -> None:
        FakeThread.instances = []
        with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit"), patch(
            "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
            return_value="handle",
        ), patch(
            "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetName",
            return_value="Test GPU",
        ), patch(
            "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetPowerUsage",
            side_effect=[20000, 40000],
        ), patch(
            "acprof.monitors.energy_nvml.time.perf_counter",
            side_effect=[0.0, 2.0],
        ), patch("acprof.monitors.energy_nvml.threading.Thread", FakeThread):
            monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.0)
            monitor.idle_power_w = 10.0

            monitor.start()
            self.assertTrue(FakeThread.instances[0].started)

            result, gpu_name, err, samples = monitor.stop()

        self.assertTrue(FakeThread.instances[0].joined)
        self.assertEqual(gpu_name, "Test GPU")
        self.assertEqual(err, "")
        self.assertEqual(samples, [(0.0, 20.0), (2.0, 40.0)])
        self.assertEqual(result.energy_iters, 2)
        self.assertEqual(result.idle_power_w, 10.0)
        self.assertEqual(result.avg_power_total_w, 30.0)
        self.assertEqual(result.peak_power_total_w, 40.0)
        self.assertEqual(result.energy_total_j, 60.0)
        self.assertEqual(result.avg_power_eff_w, 20.0)
        self.assertEqual(result.peak_power_eff_w, 30.0)
        self.assertEqual(result.energy_eff_j, 40.0)

    def test_nvml_init_failure_returns_error_result(self) -> None:
        with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit", side_effect=RuntimeError("nvml boom")):
            monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.0)
            monitor.start()
            result, gpu_name, err, samples = monitor.stop()

        self.assertEqual(gpu_name, "unknown")
        self.assertIn("nvml boom", err)
        self.assertEqual(samples, [])
        self.assertEqual(result.energy_iters, 0)
        self.assertTrue(math.isnan(result.energy_total_j))


if __name__ == "__main__":
    unittest.main()
