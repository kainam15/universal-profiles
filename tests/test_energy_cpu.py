import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import energy_cpu


class CPUEnergyMonitorTests(unittest.TestCase):
    def test_result_integrates_rapl_energy_and_handles_wrap(self) -> None:
        domain = energy_cpu.RaplDomain("intel-rapl:0", "unused", 100)
        samples = [
            energy_cpu.CPUSample(0.0, [90], 10.0, 1.0),
            energy_cpu.CPUSample(1.0, [10], 12.0, 2.0),
            energy_cpu.CPUSample(2.0, [40], 16.0, 3.0),
        ]

        result = energy_cpu._result_from_samples(samples, idle_power_w=5e-6, domains=[domain])

        self.assertEqual(result.cpu_energy_iters, 3)
        self.assertAlmostEqual(result.cpu_energy_total_j, 50e-6)
        self.assertAlmostEqual(result.cpu_avg_power_total_w, 25e-6)
        self.assertAlmostEqual(result.cpu_peak_power_total_w, 30e-6)
        self.assertAlmostEqual(result.cpu_energy_eff_j, 40e-6)
        self.assertAlmostEqual(result.vcpu_cpu_time_s, 2.0)
        self.assertAlmostEqual(result.vcpu_cpu_share, 2.0 / 6.0)
        self.assertAlmostEqual(result.vcpu_energy_total_j, 17.5e-6)

    def test_no_rapl_returns_nan_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp, idle_seconds=0.0)
            idle_power_w = monitor.measure_idle()
            monitor.start()
            result, err, samples = monitor.stop()

        self.assertTrue(math.isnan(idle_power_w))
        self.assertIn("RAPL", err)
        self.assertEqual(samples, [])
        self.assertEqual(result.cpu_energy_iters, 0)
        self.assertTrue(math.isnan(result.cpu_energy_total_j))

    def test_resolves_cgroup_v2_usage_usec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = os.path.join(tmp, "proc")
            cgroup_root = os.path.join(tmp, "cgroup")
            os.makedirs(os.path.join(proc_root, "123"))
            os.makedirs(os.path.join(cgroup_root, "docker", "abc"))

            with open(os.path.join(proc_root, "123", "cgroup"), "w", encoding="utf-8") as f:
                f.write("0::/docker/abc\n")
            with open(os.path.join(cgroup_root, "docker", "abc", "cpu.stat"), "w", encoding="utf-8") as f:
                f.write("usage_usec 1500000\n")

            fake_completed = SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            with patch("energy_cpu.subprocess.run", return_value=fake_completed):
                reader = energy_cpu._resolve_container_cpu_reader(
                    "case_container",
                    cgroup_root=cgroup_root,
                    proc_root=proc_root,
                )

            self.assertIsNotNone(reader)
            assert reader is not None
            self.assertEqual(reader(), 1.5)

    def test_missing_cgroup_keeps_raw_cpu_and_sets_vcpu_nan(self) -> None:
        domain = energy_cpu.RaplDomain("intel-rapl:0", "unused", 1_000_000)
        samples = [
            energy_cpu.CPUSample(0.0, [0], 10.0, None),
            energy_cpu.CPUSample(1.0, [20_000], 11.0, None),
        ]

        result = energy_cpu._result_from_samples(samples, idle_power_w=0.0, domains=[domain])

        self.assertAlmostEqual(result.cpu_energy_total_j, 0.02)
        self.assertTrue(math.isnan(result.vcpu_energy_total_j))
        self.assertTrue(math.isnan(result.vcpu_cpu_share))


if __name__ == "__main__":
    unittest.main()
