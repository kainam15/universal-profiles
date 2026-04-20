import math
import unittest
from unittest.mock import patch

import energy_nvml


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
    def test_measure_idle_uses_median_power(self) -> None:
        with patch("energy_nvml.pynvml.nvmlInit"), patch(
            "energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
            return_value="handle",
        ), patch(
            "energy_nvml.pynvml.nvmlDeviceGetName",
            return_value=b"Test GPU",
        ), patch(
            "energy_nvml.pynvml.nvmlDeviceGetPowerUsage",
            side_effect=[30000, 10000],
        ), patch(
            "energy_nvml.time.perf_counter",
            side_effect=[0.0, 0.0, 0.1, 0.2],
        ), patch("energy_nvml.time.sleep"):
            monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.2)
            idle_power_w = monitor.measure_idle()

        self.assertEqual(idle_power_w, 20.0)
        self.assertEqual(monitor.idle_power_w, 20.0)

    def test_start_stop_samples_and_calculates_energy(self) -> None:
        FakeThread.instances = []
        with patch("energy_nvml.pynvml.nvmlInit"), patch(
            "energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
            return_value="handle",
        ), patch(
            "energy_nvml.pynvml.nvmlDeviceGetName",
            return_value="Test GPU",
        ), patch(
            "energy_nvml.pynvml.nvmlDeviceGetPowerUsage",
            side_effect=[20000, 40000],
        ), patch(
            "energy_nvml.time.perf_counter",
            side_effect=[0.0, 2.0],
        ), patch("energy_nvml.threading.Thread", FakeThread):
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
        with patch("energy_nvml.pynvml.nvmlInit", side_effect=RuntimeError("nvml boom")):
            monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.0)
            idle_power_w = monitor.measure_idle()
            monitor.start()
            result, gpu_name, err, samples = monitor.stop()

        self.assertTrue(math.isnan(idle_power_w))
        self.assertEqual(gpu_name, "unknown")
        self.assertIn("nvml boom", err)
        self.assertEqual(samples, [])
        self.assertEqual(result.energy_iters, 0)
        self.assertTrue(math.isnan(result.energy_total_j))

    def test_compat_wrapper_stops_and_closes_when_fn_raises(self) -> None:
        class FakeMonitor:
            def __init__(self):
                self.measured_idle = False
                self.started = False
                self.stopped = False
                self.closed = False

            def measure_idle(self):
                self.measured_idle = True
                return 10.0

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True
                return energy_nvml._nan_result(), "Test GPU", "", []

            def close(self):
                self.closed = True

        fake_monitor = FakeMonitor()

        def failing_fn():
            raise ValueError("workload failed")

        with patch("energy_nvml.GPUEnergyMonitor", return_value=fake_monitor):
            with self.assertRaises(ValueError):
                energy_nvml.measure_energy_threaded(
                    fn=failing_fn,
                    sample_hz=10.0,
                    idle_seconds=0.1,
                    device_index=0,
                )

        self.assertTrue(fake_monitor.measured_idle)
        self.assertTrue(fake_monitor.started)
        self.assertTrue(fake_monitor.stopped)
        self.assertTrue(fake_monitor.closed)


if __name__ == "__main__":
    unittest.main()
