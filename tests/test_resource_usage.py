import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acprof.monitors import resource_usage


class ResourceUsageMonitorTests(unittest.TestCase):
    def test_result_calculates_container_and_gpu_usage_metrics(self) -> None:
        gib = 1024 ** 3
        samples = [
            resource_usage.ResourceUsageSample(
                0.0,
                0.0,
                100,
                10.0,
                1 * gib,
                4 * gib,
                gpu_sm_clock_mhz=1000.0,
                gpu_memory_clock_mhz=5000.0,
                gpu_pstate="P0",
                gpu_temp_c=50.0,
                container_swap_usage_bytes=10,
            ),
            resource_usage.ResourceUsageSample(
                1.0,
                1.0,
                200,
                30.0,
                2 * gib,
                4 * gib,
                gpu_sm_clock_mhz=1200.0,
                gpu_memory_clock_mhz=6000.0,
                gpu_pstate="P2",
                gpu_temp_c=52.0,
                container_swap_usage_bytes=20,
            ),
            resource_usage.ResourceUsageSample(
                2.0,
                3.0,
                300,
                20.0,
                3 * gib,
                4 * gib,
                gpu_sm_clock_mhz=1400.0,
                gpu_memory_clock_mhz=7000.0,
                gpu_pstate="P0",
                gpu_temp_c=54.0,
                container_swap_usage_bytes=30,
            ),
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
        self.assertAlmostEqual(result.container_swap_usage_avg_bytes, 20.0)
        self.assertAlmostEqual(result.container_swap_usage_peak_bytes, 30.0)
        self.assertAlmostEqual(result.gpu_util_avg_pct, 20.0)
        self.assertAlmostEqual(result.gpu_util_peak_pct, 30.0)
        self.assertAlmostEqual(result.gpu_sm_clock_mhz, 1200.0)
        self.assertAlmostEqual(result.gpu_memory_clock_mhz, 6000.0)
        self.assertEqual(result.gpu_pstate, "P0")
        self.assertAlmostEqual(result.gpu_temp_c, 52.0)
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

    def test_resolves_cgroup_v2_resource_readers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = os.path.join(tmp, "proc")
            cgroup_root = os.path.join(tmp, "cgroup")
            os.makedirs(os.path.join(proc_root, "123"))
            os.makedirs(os.path.join(cgroup_root, "docker", "abc"))

            with open(os.path.join(proc_root, "123", "cgroup"), "w", encoding="utf-8") as f:
                f.write("0::/docker/abc\n")
            with open(os.path.join(cgroup_root, "docker", "abc", "cpu.stat"), "w", encoding="utf-8") as f:
                f.write("usage_usec 1500000\n")
                f.write("nr_periods 20\n")
                f.write("nr_throttled 5\n")
                f.write("throttled_usec 250000\n")
            with open(os.path.join(cgroup_root, "docker", "abc", "memory.current"), "w", encoding="utf-8") as f:
                f.write("12345\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "memory.peak"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("23456\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "memory.stat"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("anon 10000\n")
                f.write("file 2000\n")
                f.write("slab 300\n")
                f.write("pgfault 40\n")
                f.write("pgmajfault 2\n")
                f.write("workingset_refault_anon 3\n")
                f.write("workingset_refault_file 4\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "memory.swap.current"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("2048\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "memory.swap.max"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("4096\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "io.stat"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("8:0 rbytes=100 wbytes=200 rios=1 wios=2\n")
                f.write("8:16 rbytes=300 wbytes=400 rios=3 wios=4\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "memory.events"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("low 0\nhigh 2\nmax 3\noom 1\noom_kill 1\n")
            for leaf, some_total, full_total in (
                ("cpu.pressure", 1000, 200),
                ("memory.pressure", 2000, 300),
                ("io.pressure", 3000, 400),
            ):
                with open(
                    os.path.join(cgroup_root, "docker", "abc", leaf),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        f"some avg10=0.00 avg60=0.00 avg300=0.00 total={some_total}\n"
                    )
                    f.write(
                        f"full avg10=0.00 avg60=0.00 avg300=0.00 total={full_total}\n"
                    )
            with open(
                os.path.join(cgroup_root, "docker", "abc", "pids.current"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("7\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "pids.peak"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("9\n")
            with open(
                os.path.join(cgroup_root, "docker", "abc", "pids.events"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("max 1\n")

            fake_completed = SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            with patch("acprof.monitors.resource_usage.subprocess.run", return_value=fake_completed):
                readers = resource_usage._resolve_container_metric_readers(
                    "case_container",
                    cgroup_root=cgroup_root,
                    proc_root=proc_root,
                )

            self.assertIsNotNone(readers.cpu)
            self.assertIsNotNone(readers.memory)
            self.assertIsNotNone(readers.swap)
            self.assertIsNotNone(readers.swap_limit)
            self.assertIsNotNone(readers.io)
            self.assertIsNotNone(readers.cpu_throttle)
            self.assertIsNotNone(readers.memory_events)
            self.assertIsNotNone(readers.cpu_pressure)
            self.assertIsNotNone(readers.memory_pressure)
            self.assertIsNotNone(readers.io_pressure)
            self.assertIsNotNone(readers.memory_peak)
            self.assertIsNotNone(readers.memory_stat)
            self.assertIsNotNone(readers.io_operations)
            self.assertIsNotNone(readers.pids)
            assert readers.cpu is not None
            assert readers.memory is not None
            assert readers.swap is not None
            assert readers.swap_limit is not None
            assert readers.io is not None
            assert readers.cpu_throttle is not None
            assert readers.memory_events is not None
            assert readers.cpu_pressure is not None
            assert readers.memory_pressure is not None
            assert readers.io_pressure is not None
            assert readers.memory_peak is not None
            assert readers.memory_stat is not None
            assert readers.io_operations is not None
            assert readers.pids is not None
            self.assertEqual(readers.cpu(), 1.5)
            self.assertEqual(readers.memory(), 12345)
            self.assertEqual(readers.swap(), 2048)
            self.assertEqual(readers.swap_limit(), 4096)
            self.assertEqual(readers.io(), (400, 600))
            self.assertEqual(readers.io_operations(), {"read_ops": 4.0, "write_ops": 6.0})
            self.assertEqual(readers.memory_peak(), {"peak": 23456.0})
            self.assertEqual(
                readers.memory_stat(),
                {
                    "anon": 10000.0,
                    "file": 2000.0,
                    "slab": 300.0,
                    "pgfault": 40.0,
                    "pgmajfault": 2.0,
                    "workingset_refault": 7.0,
                },
            )
            self.assertEqual(
                readers.pids(),
                {"current": 7.0, "peak": 9.0, "max_events": 1.0},
            )
            self.assertEqual(
                readers.cpu_throttle(),
                {
                    "nr_periods": 20.0,
                    "nr_throttled": 5.0,
                    "throttled_usec": 250000.0,
                },
            )
            self.assertEqual(
                readers.memory_events(),
                {"high": 2.0, "max": 3.0, "oom": 1.0, "oom_kill": 1.0},
            )
            self.assertEqual(readers.cpu_pressure(), {"some": 1000.0, "full": 200.0})
            self.assertEqual(
                readers.memory_pressure(),
                {"some": 2000.0, "full": 300.0},
            )
            self.assertEqual(readers.io_pressure(), {"some": 3000.0, "full": 400.0})

    def test_resolves_cgroup_v1_resource_readers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = os.path.join(tmp, "proc")
            cgroup_root = os.path.join(tmp, "cgroup")
            os.makedirs(os.path.join(proc_root, "123"))
            os.makedirs(os.path.join(cgroup_root, "cpu,cpuacct", "docker", "abc"))
            os.makedirs(os.path.join(cgroup_root, "memory", "docker", "abc"))
            os.makedirs(os.path.join(cgroup_root, "blkio", "docker", "abc"))

            with open(os.path.join(proc_root, "123", "cgroup"), "w", encoding="utf-8") as f:
                f.write("8:cpu,cpuacct:/docker/abc\n")
                f.write("9:memory:/docker/abc\n")
                f.write("10:blkio:/docker/abc\n")
            with open(
                os.path.join(cgroup_root, "cpu,cpuacct", "docker", "abc", "cpuacct.usage"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("2500000000\n")
            with open(
                os.path.join(cgroup_root, "cpu,cpuacct", "docker", "abc", "cpu.stat"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("nr_periods 10\nnr_throttled 2\nthrottled_time 500000000\n")
            with open(
                os.path.join(cgroup_root, "memory", "docker", "abc", "memory.usage_in_bytes"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("67890\n")
            with open(
                os.path.join(
                    cgroup_root,
                    "memory",
                    "docker",
                    "abc",
                    "memory.memsw.usage_in_bytes",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("68890\n")
            with open(
                os.path.join(
                    cgroup_root,
                    "memory",
                    "docker",
                    "abc",
                    "memory.limit_in_bytes",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("10000\n")
            with open(
                os.path.join(
                    cgroup_root,
                    "memory",
                    "docker",
                    "abc",
                    "memory.memsw.limit_in_bytes",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("14000\n")
            with open(
                os.path.join(
                    cgroup_root,
                    "blkio",
                    "docker",
                    "abc",
                    "blkio.throttle.io_service_bytes",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("8:0 Read 100\n")
                f.write("8:0 Write 200\n")
                f.write("8:16 Read 300\n")
                f.write("8:16 Write 400\n")
                f.write("Total 1000\n")

            fake_completed = SimpleNamespace(returncode=0, stdout="123\n", stderr="")
            with patch("acprof.monitors.resource_usage.subprocess.run", return_value=fake_completed):
                readers = resource_usage._resolve_container_metric_readers(
                    "case_container",
                    cgroup_root=cgroup_root,
                    proc_root=proc_root,
                )

            self.assertIsNotNone(readers.cpu)
            self.assertIsNotNone(readers.memory)
            self.assertIsNotNone(readers.swap)
            self.assertIsNotNone(readers.swap_limit)
            self.assertIsNotNone(readers.io)
            self.assertIsNotNone(readers.cpu_throttle)
            assert readers.cpu is not None
            assert readers.memory is not None
            assert readers.swap is not None
            assert readers.swap_limit is not None
            assert readers.io is not None
            assert readers.cpu_throttle is not None
            self.assertEqual(readers.cpu(), 2.5)
            self.assertEqual(readers.memory(), 67890)
            self.assertEqual(readers.swap(), 1000)
            self.assertEqual(readers.swap_limit(), 4000)
            self.assertEqual(readers.io(), (400, 600))
            self.assertEqual(
                readers.cpu_throttle(),
                {
                    "nr_periods": 10.0,
                    "nr_throttled": 2.0,
                    "throttled_usec": 500000.0,
                },
            )

    def test_window_counter_metrics_calculate_deltas_and_psi_stalls(self) -> None:
        result = resource_usage._nan_result(2)
        start = resource_usage._WindowCounterSnapshots(
            cpu_throttle={
                "nr_periods": 100.0,
                "nr_throttled": 10.0,
                "throttled_usec": 1_000_000.0,
            },
            memory_events={
                "high": 1.0,
                "max": 2.0,
                "oom": 0.0,
                "oom_kill": 0.0,
            },
            cpu_pressure={"some": 1_000_000.0, "full": 200_000.0},
            memory_pressure={"some": 300_000.0, "full": 100_000.0},
            io_pressure={"some": 400_000.0, "full": 50_000.0},
            memory_peak={"peak": 4096.0},
            memory_stat={
                "anon": 1000.0,
                "file": 200.0,
                "slab": 50.0,
                "pgfault": 10.0,
                "pgmajfault": 1.0,
                "workingset_refault": 2.0,
            },
            io_operations={"read_ops": 20.0, "write_ops": 30.0},
            pids={"current": 5.0, "peak": 6.0, "max_events": 0.0},
        )
        end = resource_usage._WindowCounterSnapshots(
            cpu_throttle={
                "nr_periods": 110.0,
                "nr_throttled": 12.0,
                "throttled_usec": 1_400_000.0,
            },
            memory_events={
                "high": 3.0,
                "max": 5.0,
                "oom": 1.0,
                "oom_kill": 1.0,
            },
            cpu_pressure={"some": 1_500_000.0, "full": 300_000.0},
            memory_pressure={"some": 500_000.0, "full": 140_000.0},
            io_pressure={"some": 700_000.0, "full": 70_000.0},
            memory_peak={"peak": 8192.0},
            memory_stat={
                "anon": 1500.0,
                "file": 300.0,
                "slab": 75.0,
                "pgfault": 25.0,
                "pgmajfault": 3.0,
                "workingset_refault": 7.0,
            },
            io_operations={"read_ops": 26.0, "write_ops": 39.0},
            pids={"current": 7.0, "peak": 9.0, "max_events": 1.0},
        )

        resource_usage._apply_window_counter_metrics(
            result,
            start,
            end,
            elapsed_s=2.0,
        )

        self.assertEqual(result.container_cpu_nr_periods_delta, 10.0)
        self.assertEqual(result.container_cpu_nr_throttled_delta, 2.0)
        self.assertEqual(result.container_cpu_throttled_period_ratio_pct, 20.0)
        self.assertEqual(result.container_cpu_throttled_time_s, 0.4)
        self.assertEqual(result.container_cpu_pressure_some_stall_pct, 25.0)
        self.assertEqual(result.container_cpu_pressure_full_stall_pct, 5.0)
        self.assertEqual(result.container_mem_high_events_delta, 2.0)
        self.assertEqual(result.container_mem_max_events_delta, 3.0)
        self.assertEqual(result.container_mem_oom_events_delta, 1.0)
        self.assertEqual(result.container_mem_oom_kill_events_delta, 1.0)
        self.assertEqual(result.container_mem_pressure_some_stall_pct, 10.0)
        self.assertEqual(result.container_mem_pressure_full_stall_pct, 2.0)
        self.assertEqual(result.container_mem_peak_cgroup_bytes, 8192.0)
        self.assertEqual(result.container_mem_anon_bytes_end, 1500.0)
        self.assertEqual(result.container_mem_file_bytes_end, 300.0)
        self.assertEqual(result.container_mem_slab_bytes_end, 75.0)
        self.assertEqual(result.container_mem_pgfault_delta, 15.0)
        self.assertEqual(result.container_mem_pgmajfault_delta, 2.0)
        self.assertEqual(result.container_mem_workingset_refault_delta, 5.0)
        self.assertEqual(result.container_io_read_ops, 6.0)
        self.assertEqual(result.container_io_write_ops, 9.0)
        self.assertEqual(result.container_io_pressure_some_stall_pct, 15.0)
        self.assertEqual(result.container_io_pressure_full_stall_pct, 1.0)
        self.assertEqual(result.container_pids_current_end, 7.0)
        self.assertEqual(result.container_pids_peak_cgroup, 9.0)
        self.assertEqual(result.container_pids_max_events_delta, 1.0)

    def test_cgroup_unlimited_limit_uses_negative_one_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            limit_path = os.path.join(tmp, "memory.swap.max")
            with open(limit_path, "w", encoding="utf-8") as f:
                f.write("max\n")
            self.assertEqual(resource_usage._read_cgroup_limit(limit_path), -1)

    def test_monitor_collects_swap_and_window_io_delta(self) -> None:
        state = {
            "swap": 10,
            "read": 100,
            "write": 200,
        }
        readers = resource_usage._ContainerReaders(
            swap=lambda: state["swap"],
            swap_limit=lambda: 4096,
            io=lambda: (state["read"], state["write"]),
        )

        with patch(
            "acprof.monitors.resource_usage._resolve_container_metric_readers",
            return_value=readers,
        ), patch(
            "acprof.monitors.resource_usage._read_cpu_frequency_hz",
            return_value=(None, None),
        ):
            monitor = resource_usage.ResourceUsageMonitor(
                sample_hz=1.0,
                container_name="case_container",
            )
            monitor.start()
            state.update({"swap": 30, "read": 700, "write": 1000})
            result, error, samples = monitor.stop()
            monitor.close()

        self.assertEqual(error, "")
        self.assertEqual(len(samples), 2)
        self.assertEqual(result.container_swap_limit_bytes, 4096.0)
        self.assertEqual(result.container_swap_usage_avg_bytes, 20.0)
        self.assertEqual(result.container_swap_usage_peak_bytes, 30.0)
        self.assertEqual(result.container_io_read_bytes, 600.0)
        self.assertEqual(result.container_io_write_bytes, 800.0)

    def test_reads_gpu_utilization_and_memory_from_nvml(self) -> None:
        fake_nvml = SimpleNamespace(
            NVML_CLOCK_SM=1,
            NVML_CLOCK_MEM=2,
            NVML_TEMPERATURE_GPU=0,
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda index: "handle",
            nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=42),
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=123, total=456),
            nvmlDeviceGetClockInfo=lambda handle, clock_type: {
                1: 1500,
                2: 7000,
            }[clock_type],
            nvmlDeviceGetPerformanceState=lambda handle: 2,
            nvmlDeviceGetTemperature=lambda handle, sensor_type: 63,
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
        self.assertEqual(sample.gpu_sm_clock_mhz, 1500.0)
        self.assertEqual(sample.gpu_memory_clock_mhz, 7000.0)
        self.assertEqual(sample.gpu_pstate, "P2")
        self.assertEqual(sample.gpu_temp_c, 63.0)

    def test_dominant_pstate_prefers_higher_performance_on_tie(self) -> None:
        self.assertEqual(
            resource_usage._dominant_pstate(["P2", "p0", "P2", "P0"]),
            "P0",
        )
        self.assertEqual(resource_usage._dominant_pstate(["invalid"]), "nan")

    def test_unavailable_container_keeps_nan_result_without_raising(self) -> None:
        fake_completed = SimpleNamespace(returncode=1, stdout="", stderr="missing")
        with patch("acprof.monitors.resource_usage.subprocess.run", return_value=fake_completed):
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
        self.assertTrue(math.isnan(result.container_swap_limit_bytes))
        self.assertTrue(math.isnan(result.container_swap_usage_avg_bytes))
        self.assertTrue(math.isnan(result.container_swap_usage_peak_bytes))
        self.assertTrue(math.isnan(result.container_io_read_bytes))
        self.assertTrue(math.isnan(result.container_io_write_bytes))
        self.assertTrue(math.isnan(result.container_cpu_nr_periods_delta))
        self.assertTrue(
            math.isnan(result.container_cpu_throttled_period_ratio_pct)
        )
        self.assertTrue(math.isnan(result.container_mem_oom_kill_events_delta))
        self.assertTrue(
            math.isnan(result.container_mem_pressure_full_stall_pct)
        )
        self.assertTrue(math.isnan(result.gpu_util_avg_pct))
        self.assertTrue(math.isnan(result.gpu_sm_clock_mhz))
        self.assertTrue(math.isnan(result.gpu_memory_clock_mhz))
        self.assertEqual(result.gpu_pstate, "nan")
        self.assertTrue(math.isnan(result.gpu_temp_c))


if __name__ == "__main__":
    unittest.main()
