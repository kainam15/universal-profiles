import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.monitors import energy_cpu


def _write_rapl_domain(
    root: str,
    entry_name: str,
    domain_name: str,
    energy_uj: int = 0,
    max_range_uj: int = 1_000_000_000,
) -> str:
    domain_dir = os.path.join(root, entry_name)
    os.makedirs(domain_dir)
    with open(os.path.join(domain_dir, "name"), "w", encoding="utf-8") as f:
        f.write(f"{domain_name}\n")
    with open(os.path.join(domain_dir, "energy_uj"), "w", encoding="utf-8") as f:
        f.write(f"{energy_uj}\n")
    with open(os.path.join(domain_dir, "max_energy_range_uj"), "w", encoding="utf-8") as f:
        f.write(f"{max_range_uj}\n")
    return domain_dir


class CPUEnergyMonitorTests(unittest.TestCase):
    def test_topology_retains_subdomains_aliases_and_unreadable_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = _write_rapl_domain(tmp, "intel-rapl:0", "package-0")
            dram = _write_rapl_domain(package, "intel-rapl:0:0", "dram")
            _write_rapl_domain(package, "intel-rapl:0:1", "core")
            missing = _write_rapl_domain(tmp, "intel-rapl:1", "package-1")
            os.unlink(os.path.join(missing, "energy_uj"))
            os.symlink(dram, os.path.join(tmp, "intel-rapl:0:0"))
            os.symlink(package, os.path.join(package, "intel-rapl:9"))
            topology = energy_cpu.discover_rapl_topology(tmp)
            self.assertEqual(len(topology['domains']), 4)
            by_id = {d['id']: d for d in topology['domains']}
            self.assertEqual(by_id['intel-rapl:0:0']['parent_id'], 'intel-rapl:0')
            self.assertEqual(len(by_id['intel-rapl:0:0']['aliases']), 2)
            self.assertEqual(by_id['intel-rapl:1']['status'], 'unavailable')
            self.assertFalse(by_id['intel-rapl:0:1']['selected'])

    def test_dram_uses_adjacent_wraps_and_explicit_request_denominator(self):
        from acprof.host.client_metrics import _dram_metrics_from_result
        domains = [energy_cpu.RaplDomain('dram-0', '', 100), energy_cpu.RaplDomain('dram-1', '', 200)]
        samples = [energy_cpu.CPUSample(0, [1000], 0, None, [90, 180]),
                   energy_cpu.CPUSample(1, [2000], 0, None, [10, 20]),
                   energy_cpu.CPUSample(2, [3000], 0, None, [90, 180]),
                   energy_cpu.CPUSample(3, [4000], 0, None, [10, 20])]
        result = energy_cpu._dram_result_from_samples(samples, domains, 10e-6)
        # Domain 0: 20 + 80 + 20 uJ; domain 1: 40 + 160 + 40 uJ.
        self.assertAlmostEqual(result.window_energy_j, 360e-6)
        self.assertAlmostEqual(result.window_effective_energy_j, 330e-6)
        self.assertAlmostEqual(result.avg_power_w, 120e-6)
        self.assertEqual(result.status, 'verified')
        metrics = _dram_metrics_from_result(result, 4)
        self.assertAlmostEqual(metrics['dram_energy_per_request_j'], 90e-6)
        self.assertAlmostEqual(metrics['dram_effective_energy_per_request_j'], 82.5e-6)
        self.assertAlmostEqual(metrics['dram_window_energy_j'], 360e-6)
        self.assertTrue(math.isnan(_dram_metrics_from_result(result, 0)['dram_energy_per_request_j']))

    def test_dram_missing_mid_window_and_unknown_wrap_never_become_zero(self):
        domain = energy_cpu.RaplDomain('dram', '', 0)
        for end in (None, [10]):
            samples = [energy_cpu.CPUSample(0, [1], 0, None, [90]),
                       energy_cpu.CPUSample(1, [2], 0, None, end)]
            result = energy_cpu._dram_result_from_samples(samples, [domain], 0)
            self.assertEqual(result.status, 'error')
            self.assertTrue(math.isnan(result.window_energy_j))

    def test_dram_optional_required_off_and_partial_socket_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_rapl_domain(tmp, 'intel-rapl:0', 'package-0')
            _write_rapl_domain(tmp, 'intel-rapl:1', 'package-1')
            _write_rapl_domain(tmp, 'intel-rapl:0:0', 'dram')
            monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp)
            self.assertTrue(monitor.available)
            self.assertEqual(monitor.dram_status, 'unavailable')
            with self.assertRaisesRegex(RuntimeError, 'required DRAM'):
                energy_cpu.CPUEnergyMonitor(powercap_root=tmp, dram_energy='required')
            disabled = energy_cpu.CPUEnergyMonitor(powercap_root=tmp, dram_energy='off')
            self.assertEqual(disabled.dram_status, 'not_requested')
            self.assertEqual(disabled.dram_domains, [])
            _write_rapl_domain(tmp, 'intel-rapl:1:0', 'dram')
            enabled = energy_cpu.CPUEnergyMonitor(powercap_root=tmp, dram_energy='required')
            self.assertEqual(enabled.dram_status, 'available')

    def test_dram_read_failure_preserves_package_sample(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(energy_cpu, '_read_host_active_s', return_value=1):
            _write_rapl_domain(tmp, 'intel-rapl:0', 'package-0', energy_uj=123)
            path = _write_rapl_domain(tmp, 'intel-rapl:0:0', 'dram', energy_uj=12)
            monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp)
            Path(path, 'energy_uj').unlink()
            sample = monitor._read_sample(1)
            self.assertEqual(sample.energy_uj, [123])
            self.assertIsNone(sample.dram_energy_uj)
            self.assertTrue(monitor._dram_runtime_error)

    def test_alternate_package_interfaces_are_not_additive(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_rapl_domain(tmp, 'intel-rapl:0', 'package-0')
            _write_rapl_domain(tmp, 'intel-rapl-mmio:0', 'package-0')
            domains = energy_cpu._discover_rapl_domains(tmp)
            self.assertEqual([d.name for d in domains], ['intel-rapl:0'])

    def test_dram_permission_denied_keeps_full_topology_and_package_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_rapl_domain(tmp, 'intel-rapl:0', 'package-0')
            dram = _write_rapl_domain(tmp, 'intel-rapl:0:0', 'dram')
            original_read = Path.read_text

            def read(path, *args, **kwargs):
                if path == Path(dram, 'energy_uj'):
                    raise PermissionError('DRAM counter denied')
                return original_read(path, *args, **kwargs)

            with patch.object(Path, 'read_text', read):
                monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp)
                self.assertTrue(monitor.available)
                self.assertEqual(monitor.dram_status, 'permission_denied')
                self.assertEqual(len(monitor.topology['domains']), 2)
                self.assertEqual(monitor.topology['dram_missing_packages'], ['intel-rapl:0'])
                with self.assertRaisesRegex(RuntimeError, 'permission_denied'):
                    energy_cpu.CPUEnergyMonitor(powercap_root=tmp, dram_energy='required')

    def test_disabled_power_capping_does_not_disable_energy_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            for identifier, name in (('intel-rapl:0', 'package-0'), ('intel-rapl:0:0', 'dram')):
                path = _write_rapl_domain(tmp, identifier, name)
                Path(path, 'enabled').write_text('0\n')
            monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp, dram_energy='required')
            self.assertTrue(monitor.available)
            self.assertEqual(monitor.dram_status, 'available')
            self.assertTrue(all(d['enabled'] is False for d in monitor.topology['domains']))

    def test_sysfs_alias_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_rapl_domain(tmp, "intel-rapl:0", "package-0")
            os.symlink(path, os.path.join(tmp, "intel-rapl:9"))
            domains = energy_cpu._discover_rapl_domains(tmp)
            self.assertEqual(len(domains), 1)

    def test_apply_control_baseline_uses_control_average_and_records_method(self) -> None:
        monitor = object.__new__(energy_cpu.CPUEnergyMonitor)
        monitor.idle_power_w = float("nan")
        monitor.idle_trace = {}
        monitor.dt = 0.5
        monitor.domains = [energy_cpu.RaplDomain("intel-rapl:0", "", 0)]
        samples = [
            energy_cpu.CPUSample(0.0, [0], 0.0, 0.0),
            energy_cpu.CPUSample(1.0, [4_000_000], 0.2, 0.01),
        ]
        result = energy_cpu._result_from_samples(
            samples,
            idle_power_w=float("nan"),
            domains=monitor.domains,
        )

        idle_power_w = monitor.apply_control_baseline(result, samples, trace=True)

        self.assertEqual(idle_power_w, 4.0)
        self.assertEqual(
            monitor.idle_trace["cpu_idle_baseline_method"],
            "matched_control_energy_delta",
        )
        self.assertEqual(
            monitor.idle_trace["idle_trace_schema"],
            "cpu_rapl_control_v1",
        )
        self.assertEqual(monitor.idle_trace["idle_proc_cpu_top"], [])

    def test_discover_rapl_domains_uses_package_domains_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_rapl_domain(tmp, "intel-rapl:0", "package-0")
            _write_rapl_domain(tmp, "intel-rapl:0:0", "core")

            domains = energy_cpu._discover_rapl_domains(tmp)

        self.assertEqual([domain.name for domain in domains], ["intel-rapl:0"])


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

    def test_peak_power_ignores_too_short_intervals(self) -> None:
        domain = energy_cpu.RaplDomain("intel-rapl:0", "unused", 10_000_000_000)
        samples = [
            energy_cpu.CPUSample(0.0, [0], 0.0, 0.0),
            energy_cpu.CPUSample(0.001, [500_000], 0.001, 0.001),
            energy_cpu.CPUSample(0.101, [600_000], 0.101, 0.101),
            energy_cpu.CPUSample(0.201, [700_000], 0.201, 0.201),
        ]

        result = energy_cpu._result_from_samples(
            samples,
            idle_power_w=0.0,
            domains=[domain],
            min_power_interval_s=0.05,
        )

        self.assertAlmostEqual(result.cpu_avg_power_total_w, 0.7 / 0.201)
        self.assertAlmostEqual(result.cpu_peak_power_total_w, 1.0)
        self.assertAlmostEqual(result.cpu_peak_power_eff_w, 1.0)
        self.assertAlmostEqual(result.vcpu_peak_power_total_w, 1.0)
        self.assertAlmostEqual(result.vcpu_peak_power_eff_w, 1.0)

    def test_no_rapl_returns_nan_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = energy_cpu.CPUEnergyMonitor(powercap_root=tmp, idle_seconds=0.0)
            monitor.start()
            result, err, samples = monitor.stop()

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
            with patch("acprof.monitors.energy_cpu.subprocess.run", return_value=fake_completed):
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
