"""固定历史 CSV 顺序，并核对跨消费者和关键口径。"""
import hashlib
import json
import unittest

from acprof.config import CSV_FIELDS


class MetricRegistryTests(unittest.TestCase):
    def test_historical_field_order_is_unchanged(self):
        self.assertEqual(hashlib.sha256(json.dumps(CSV_FIELDS).encode()).hexdigest(),
                         "1422b14ebaa48586573923cea2d33f615e6dc180d099cdf773678f453e3268f3")

    def test_consumers_share_registry_without_changing_tool_completeness(self):
        from acprof.metric_registry import CSV_FIELDS as registry_fields, tool_fields
        from acprof.host.posthoc.context import TOOL_FIELDS, TOOL_METRIC_FIELDS
        self.assertIs(CSV_FIELDS, registry_fields)
        for tool in TOOL_FIELDS:
            self.assertEqual(TOOL_FIELDS[tool], tool_fields(tool))
            self.assertEqual(TOOL_METRIC_FIELDS[tool], tool_fields(tool, numeric_only=True))
        self.assertEqual(tool_fields("torch", numeric_only=True),
                         ("model_logical_mflop_per_request_torch_profiler_eager",))

    def test_units_windows_and_text_are_explicit(self):
        from acprof.metric_registry import METRICS, NUMERIC_FIELDS
        self.assertEqual(METRICS["gpu_energy_total_j"].unit, "J/request")
        self.assertEqual(METRICS["container_mem_peak_cgroup_bytes"].window, "cgroup_lifetime")
        self.assertEqual(METRICS["cpu_heap_peak_bytes_massif"].window, "profiler_process_lifetime")
        self.assertNotIn("gpu_pstate", NUMERIC_FIELDS)
        self.assertNotIn("gpu_idle_measured_at", NUMERIC_FIELDS)
