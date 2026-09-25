import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.config import STATIC_META_SCHEMA_VERSION
from acprof.cli.run_args import build_parser
from acprof.host import client_metrics
from acprof.packet import merge_packet_latency


class LatencySLOPolicyTests(unittest.TestCase):
    def test_cli_rules_select_task_before_profile_before_default(self):
        args = build_parser().parse_args([
            "--model", "test/model", "--latency-slo", "default=0.5",
            "--latency-slo", "profile:nlp-cu128=1",
            "--latency-slo", "task:text-generation=5",
        ])
        from acprof.latency_slo import parse_latency_slo_rules, resolve_latency_slo
        policy = parse_latency_slo_rules(args.latency_slo, environment_threshold="0.06")
        for task, profile, expected, source in (
            ("text-generation", "nlp-cu128", 5.0, "task:text-generation"),
            ("fill-mask", "nlp-cu128", 1.0, "profile:nlp-cu128"),
            ("image-classification", "cv-cu128", 0.5, "default"),
        ):
            with self.subTest(task=task):
                selected = resolve_latency_slo(policy, pipeline_tag=task, runtime_profile_id=profile)
                self.assertEqual(selected["threshold_s"], expected)
                self.assertEqual(selected["source"], source)


class LatencyDistributionTests(unittest.TestCase):
    def test_tail_ratio_is_scale_free_and_uses_nearest_rank(self):
        for scale in (1.0, 1000.0):
            with self.subTest(scale=scale):
                result = client_metrics._latency_distribution_metrics(
                    "latency_app", [value * scale for value in (0.01, 0.02, 0.1, 0.2, 0.3)],
                    slow_latency_threshold_s=0.06,
                )
                self.assertAlmostEqual(result.get("latency_app_tail_ratio", math.nan), 3.0)
                packet = merge_packet_latency._distribution_metrics(
                    [value * scale for value in (0.01, 0.02, 0.1, 0.2, 0.3)]
                )
                self.assertAlmostEqual(packet.get("latency_tail_ratio", math.nan), 3.0)

    def test_tail_ratio_boundaries(self):
        for values, expected in (([0.2], 1.0), ([0.0, 0.0, 0.1], math.nan),
                                 ([], math.nan), ([math.nan, math.inf], math.nan)):
            with self.subTest(values=values):
                result = client_metrics._latency_distribution_metrics(
                    "latency_app", values, slow_latency_threshold_s=0.1,
                )
                self.assertIn("latency_app_tail_ratio", result)
                actual = result["latency_app_tail_ratio"]
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected)

    def test_unconfigured_threshold_does_not_report_zero_violations(self):
        result = client_metrics._latency_distribution_metrics(
            "latency_app", [0.01, 0.06, 0.2], slow_latency_threshold_s=math.nan,
        )
        self.assertTrue(math.isnan(result["latency_app_slow_ratio"]))


class PacketSLOTests(unittest.TestCase):
    def test_merge_uses_recorded_slo_and_never_the_current_environment(self):
        for slo, expected in (({"threshold_s": 0.2, "source": "task:fill-mask"}, "0.200000"),
                              ({"threshold_s": None, "source": "unconfigured"}, "nan"),
                              (None, "nan")):
            with self.subTest(slo=slo), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                metadata = {"schema_version": STATIC_META_SCHEMA_VERSION, "batch_size": 1}
                if slo is not None:
                    metadata["latency_slo"] = slo
                (root / "static_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
                source = root / "result.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "sniff_group_id", "latency_s", "latency_slow_ratio", "latency_tail_ratio",
                    ])
                    writer.writeheader()
                    writer.writerow({"sniff_group_id": "case", "latency_slow_ratio": "nan"})
                samples = root / "packet.json"
                samples.write_text(json.dumps({"schema_version": 2, "requests": {
                    f"case:{index}": {"latency_s": value}
                    for index, value in enumerate((0.01, 0.02, 0.1, 0.2, 0.3))
                }}), encoding="utf-8")
                output = root / "merged.csv"
                with patch.dict("os.environ", {"SLOW_LATENCY_THRESHOLD_S": "0.001"}):
                    merge_packet_latency.main([str(source), str(samples), str(output)])
                with output.open(encoding="utf-8") as stream:
                    result = next(csv.DictReader(stream))
                self.assertEqual(result["latency_slow_ratio"], expected)
                self.assertEqual(result["latency_tail_ratio"], "3.000000")


if __name__ == "__main__":
    unittest.main()
