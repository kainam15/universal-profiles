import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import resource_usage


class ResourceUsageMonitorTests(unittest.TestCase):
    def test_result_calculates_container_and_gpu_usage_metrics(self) -> None:
        gib = 1024 ** 3
        samples = [
            resource_usage.ResourceUsageSample(0.0, 0.0, 100, 10.0, 1 * gib, 4 * gib),
            resource_usage.ResourceUsageSample(1.0, 1.0, 200, 30.0, 2 * gib, 4 * gib),
            resource_usage.ResourceUsageSample(2.0, 3.0, 300, 20.0, 3 * gib, 4 * gib),
        ]

        result = resource_usage._result_from_samples(
            samples,
            cpu_cores=2.0,
            mem_limit_bytes=1000,
        )

        self.assertEqual(result.resource_usage_iters, 3)
        self.assertAlmostEqual(result.container_cpu_util_avg_pct, 75.0)
        self.assertAlmostEqual(result.container_cpu_util_peak_pct, 100.0)
        self.assertAlmostEqual(result.container_mem_usage_avg_bytes, 200.0)
        self.assertAlmostEqual(result.container_mem_usage_peak_bytes, 300.0)
        self.assertAlmostEqual(result.container_mem_util_avg_pct, 20.0)
        self.assertAlmostEqual(result.container_mem_util_peak_pct, 30.0)
        self.assertAlmostEqual(result.gpu_util_avg_pct, 20.0)
        self.assertAlmostEqual(result.gpu_util_peak_pct, 30.0)
        self.assertAlmostEqual(result.gpu_mem_used_avg_bytes, 2 * gib)
        self.assertAlmostEqual(result.gpu_mem_used_peak_bytes, 3 * gib)
        self.assertAlmostEqual(result.gpu_mem_util_avg_pct, 50.0)
        self.assertAlmostEqual(result.gpu_mem_util_peak_pct, 75.0)
        self.assertAlmostEqual(result.gpu_mem_total_bytes, 4 * gib)

    def test_result_calculates_cpu_frequency_metrics(self) -> None:
        samples = [
            resource_usage.ResourceUsageSample(
                0.0,
                0.0,
                None,
                None,
                None,
                None,
                cpu_freq_avg_hz=2_000_000_000.0,
                cpu_freq_peak_hz=2_200_000_000.0,
            ),
            resource_usage.ResourceUsageSample(
                1.0,
                1.0,
                None,
                None,
                None,
                None,
                cpu_freq_avg_hz=3_000_000_000.0,
                cpu_freq_peak_hz=3_200_000_000.0,
            ),
        ]

        result = resource_usage._result_from_samples(
            samples,
            cpu_cores=1.0,
            mem_limit_bytes=0.0,
        )

        self.assertAlmostEqual(result.cpu_freq_avg_hz, 2_500_000_000.0)
        self.assertAlmostEqual(result.cpu_freq_peak_hz, 3_200_000_000.0)

    def test_reads_cpu_frequency_from_sysfs_khz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu_root = os.path.join(tmp, "cpu")
            os.makedirs(os.path.join(cpu_root, "cpu0", "cpufreq"))
            os.makedirs(os.path.join(cpu_root, "cpu1", "cpufreq"))
            with open(os.path.join(cpu_root, "online"), "w", encoding="utf-8") as f:
                f.write("0-1\n")
            with open(
                os.path.join(cpu_root, "cpu0", "cpufreq", "scaling_cur_freq"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("2200000\n")
            with open(
                os.path.join(cpu_root, "cpu1", "cpufreq", "scaling_cur_freq"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("1800000\n")

            avg_hz, peak_hz = resource_usage._read_cpu_frequency_hz(
                cpu_sysfs_root=cpu_root,
                proc_cpuinfo_path=os.path.join(tmp, "missing_cpuinfo"),
            )

        self.assertAlmostEqual(avg_hz, 2_000_000_000.0)
        self.assertAlmostEqual(peak_hz, 2_200_000_000.0)

    def test_peak_cpu_util_ignores_too_short_intervals(self) -> None:
        samples = [
            resource_usage.ResourceUsageSample(0.0, 0.0, None, None, None, None),
            resource_usage.ResourceUsageSample(0.001, 0.02, None, None, None, None),
            resource_usage.ResourceUsageSample(0.101, 0.12, None, None, None, None),
            resource_usage.ResourceUsageSample(0.201, 0.22, None, None, None, None),
        ]

        result = resource_usage._result_from_samples(
            samples,
            cpu_cores=1.0,
            mem_limit_bytes=0.0,
            min_cpu_interval_s=0.05,
        )

        self.assertAlmostEqual(result.container_cpu_util_avg_pct, (0.22 / 0.201) * 100.0)
        self.assertAlmostEqual(result.container_cpu_util_peak_pct, 100.0)

    def test_resolves_cgroup_v2_cpu_and_memory_readers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = os.path.join(tmp, "proc")
            cgroup_root = os.path.join(tmp, "cgroup")
            os.makedirs(os.path.join(proc_root, "123"))
            os.makedirs(os.path.join(cgroup_root, "docker", "abc"))

            with open(os.path.join(proc_root, "123", "cgroup"), "w", encoding="utf-8") as f:
                f.write("0::/docker/abc\n")
            with open(os.path.join(cgroup_root, "docker", "abc", "cpu.stat"), "w", encoding="utf-8") as f:
                f.write("usage_usec 1500000\n")
            with open(os.path.join(cgroup_root, "docker", "abc", "memory.current"), "w", encoding="utf-8") as f:
                f.write("12345\n")

            fake_completed = SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            with patch("resource_usage.subprocess.run", return_value=fake_completed):
                cpu_reader, mem_reader = resource_usage._resolve_container_readers(
                    "case_container",
                    cgroup_root=cgroup_root,
                    proc_root=proc_root,
                )

            self.assertIsNotNone(cpu_reader)
            self.assertIsNotNone(mem_reader)
            assert cpu_reader is not None
            assert mem_reader is not None
            self.assertEqual(cpu_reader(), 1.5)
            self.assertEqual(mem_reader(), 12345)

    def test_resolves_cgroup_v1_cpuacct_and_memory_readers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = os.path.join(tmp, "proc")
            cgroup_root = os.path.join(tmp, "cgroup")
            os.makedirs(os.path.join(proc_root, "123"))
            os.makedirs(os.path.join(cgroup_root, "cpu,cpuacct", "docker", "abc"))
            os.makedirs(os.path.join(cgroup_root, "memory", "docker", "abc"))

            with open(os.path.join(proc_root, "123", "cgroup"), "w", encoding="utf-8") as f:
                f.write("8:cpu,cpuacct:/docker/abc\n")
                f.write("9:memory:/docker/abc\n")
            with open(
                os.path.join(cgroup_root, "cpu,cpuacct", "docker", "abc", "cpuacct.usage"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("2500000000\n")
            with open(
                os.path.join(cgroup_root, "memory", "docker", "abc", "memory.usage_in_bytes"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("67890\n")

            fake_completed = SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            with patch("resource_usage.subprocess.run", return_value=fake_completed):
                cpu_reader, mem_reader = resource_usage._resolve_container_readers(
                    "case_container",
                    cgroup_root=cgroup_root,
                    proc_root=proc_root,
                )

            self.assertIsNotNone(cpu_reader)
            self.assertIsNotNone(mem_reader)
            assert cpu_reader is not None
            assert mem_reader is not None
            self.assertEqual(cpu_reader(), 2.5)
            self.assertEqual(mem_reader(), 67890)

    def test_reads_gpu_utilization_and_memory_from_nvml(self) -> None:
        fake_nvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: "handle",
            nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=42),
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=123, total=456),
            nvmlShutdown=lambda: None,
        )

        with patch.object(resource_usage, "pynvml", fake_nvml):
            monitor = resource_usage.ResourceUsageMonitor(
                sample_hz=1.0,
                container_name="",
                cpu_cores=1,
                mem_cap_gb=1,
                use_gpu=True,
            )
            sample = monitor._read_sample(1.0)
            monitor.close()

        self.assertEqual(sample.gpu_util_pct, 42.0)
        self.assertEqual(sample.gpu_mem_used_bytes, 123)
        self.assertEqual(sample.gpu_mem_total_bytes, 456)

    def test_unavailable_container_keeps_nan_result_without_raising(self) -> None:
        fake_completed = SimpleNamespace(returncode=1, stdout="", stderr="missing")
        with patch("resource_usage.subprocess.run", return_value=fake_completed):
            monitor = resource_usage.ResourceUsageMonitor(
                sample_hz=10.0,
                container_name="missing_container",
                cpu_cores=1,
                mem_cap_gb=1,
                use_gpu=False,
            )
            monitor.start()
            result, err, samples = monitor.stop()
            monitor.close()

        self.assertIn("missing", err)
        self.assertEqual(samples, [])
        self.assertEqual(result.resource_usage_iters, 0)
        self.assertTrue(math.isnan(result.container_cpu_util_avg_pct))
        self.assertTrue(math.isnan(result.container_mem_usage_avg_bytes))
        self.assertTrue(math.isnan(result.gpu_util_avg_pct))


if __name__ == "__main__":
    unittest.main()
