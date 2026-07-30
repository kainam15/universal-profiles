import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


class MergePacketLatencyComputeTests(unittest.TestCase):
    def test_recomputes_both_explicit_flop_rates_from_packet_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")
            fieldnames = [
                "sniff_group_id",
                "latency_s",
                "compute_profile_tool",
                "model_mflop_per_request",
                "compute_mflops_app",
                "compute_mflops",
                "model_logical_mflop_per_request_torch_profiler_eager",
                "model_logical_mflops_app_torch_profiler_eager",
                "model_logical_mflops_packet_torch_profiler_eager",
                "gpu_executed_mflop_per_request_ncu",
                "gpu_executed_mflops_app_ncu",
                "gpu_executed_mflops_packet_ncu",
            ]
            with open(in_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "sniff_group_id": "case_seq1_r0",
                        "latency_s": "nan",
                        "compute_profile_tool": "torch_profiler_eager",
                        "model_mflop_per_request": "200",
                        "compute_mflops_app": "400",
                        "compute_mflops": "400",
                        "model_logical_mflop_per_request_torch_profiler_eager": "200",
                        "model_logical_mflops_app_torch_profiler_eager": "400",
                        "model_logical_mflops_packet_torch_profiler_eager": "nan",
                        "gpu_executed_mflop_per_request_ncu": "100",
                        "gpu_executed_mflops_app_ncu": "200",
                        "gpu_executed_mflops_packet_ncu": "nan",
                    }
                )
            with open(lat_json, "w", encoding="utf-8") as f:
                json.dump({"case_seq1_r0:0": 0.25}, f)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "acprof.packet.merge_packet_latency",
                    in_csv,
                    lat_json,
                    out_csv,
                ],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )
            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

        self.assertEqual(row["compute_mflops"], "800.000000")
        self.assertEqual(row["compute_mflops_app"], "400")
        self.assertEqual(
            row["model_logical_mflops_packet_torch_profiler_eager"],
            "800.000000",
        )
        self.assertEqual(
            row["model_logical_mflops_app_torch_profiler_eager"],
            "400",
        )
        self.assertEqual(
            row["gpu_executed_mflops_packet_ncu"],
            "400.000000",
        )
        self.assertEqual(row["gpu_executed_mflops_app_ncu"], "200")

    def test_recomputes_compute_mflops_from_packet_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")
            static_meta = os.path.join(tmp, "static_meta.json")

            with open(static_meta, "w", encoding="utf-8") as f:
                json.dump({"batch_size": 1}, f)

            fieldnames = [
                "sniff_group_id",
                "latency_s",
                "latency_p50_s",
                "latency_p90_s",
                "latency_p95_s",
                "latency_slow_ratio",
                "throughput_samples_per_s",
                "model_mflop_per_request",
                "compute_mflops",
                "cpu_cores",
                "container_cpu_util_avg_pct",
                "cpu_freq_avg_hz",
                "cpu_cycles_est_packet",
                "cpu_instructions_per_request",
                "cpu_mips_packet",
            ]
            with open(in_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "sniff_group_id": "case_seq1_r0",
                    "latency_s": "nan",
                    "latency_p50_s": "nan",
                    "latency_p90_s": "nan",
                    "latency_p95_s": "nan",
                    "latency_slow_ratio": "nan",
                    "throughput_samples_per_s": "2.000000",
                    "model_mflop_per_request": "200.000000",
                    "compute_mflops": "400.000000",
                    "cpu_cores": "2",
                    "container_cpu_util_avg_pct": "50.0",
                    "cpu_freq_avg_hz": "3000000000",
                    "cpu_cycles_est_packet": "nan",
                    "cpu_instructions_per_request": "500000",
                    "cpu_mips_packet": "nan",
                })

            with open(lat_json, "w", encoding="utf-8") as f:
                json.dump({"case_seq1_r0:0": 0.25, "case_seq1_r0:1": 0.25}, f)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "acprof.packet.merge_packet_latency",
                    in_csv,
                    lat_json,
                    out_csv,
                ],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )

            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

        self.assertEqual(rows[0]["latency_s"], "0.250000")
        self.assertEqual(rows[0]["latency_p50_s"], "0.250000")
        self.assertEqual(rows[0]["latency_p90_s"], "0.250000")
        self.assertEqual(rows[0]["latency_p95_s"], "0.250000")
        self.assertEqual(rows[0]["latency_slow_ratio"], "1.000000")
        self.assertEqual(rows[0]["throughput_samples_per_s"], "4.000000")
        self.assertEqual(rows[0]["compute_mflops"], "800.000000")
        self.assertEqual(rows[0]["cpu_cycles_est_packet"], "750000000.000000")
        self.assertEqual(rows[0]["cpu_mips_packet"], "2.000000")
        self.assertNotIn("sniff_group_id", fieldnames)

    def test_merges_packet_latency_distribution_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")

            fieldnames = [
                "sniff_group_id",
                "latency_s",
                "latency_p50_s",
                "latency_p90_s",
                "latency_p95_s",
                "latency_slow_ratio",
            ]
            with open(in_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "sniff_group_id": "case_seq1_r0",
                    "latency_s": "nan",
                    "latency_p50_s": "nan",
                    "latency_p90_s": "nan",
                    "latency_p95_s": "nan",
                    "latency_slow_ratio": "nan",
                })

            with open(lat_json, "w", encoding="utf-8") as f:
                json.dump({
                    "case_seq1_r0:0": 0.01,
                    "case_seq1_r0:1": 0.02,
                    "case_seq1_r0:2": 0.10,
                    "case_seq1_r0:3": 0.20,
                    "case_seq1_r0:4": 0.30,
                }, f)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "acprof.packet.merge_packet_latency",
                    in_csv,
                    lat_json,
                    out_csv,
                ],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )

            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["latency_s"], "0.126000")
        self.assertEqual(rows[0]["latency_p50_s"], "0.100000")
        self.assertEqual(rows[0]["latency_p90_s"], "0.300000")
        self.assertEqual(rows[0]["latency_p95_s"], "0.300000")
        self.assertEqual(rows[0]["latency_slow_ratio"], "0.600000")

    def test_merges_packet_latency_with_sidecar_when_csv_omits_sniff_group_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")
            static_meta = os.path.join(tmp, "static_meta.json")
            sidecar = f"{in_csv}.sniff_groups.jsonl"

            with open(static_meta, "w", encoding="utf-8") as f:
                json.dump({"batch_size": 2}, f)

            fieldnames = [
                "latency_s",
                "throughput_samples_per_s",
                "model_mflop_per_request",
                "compute_mflops",
            ]
            with open(in_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "latency_s": "nan",
                    "throughput_samples_per_s": "2.000000",
                    "model_mflop_per_request": "200.000000",
                    "compute_mflops": "400.000000",
                })
            with open(sidecar, "w", encoding="utf-8") as f:
                f.write(json.dumps({"sniff_group_id": "case_seq1_r0"}) + "\n")

            with open(lat_json, "w", encoding="utf-8") as f:
                json.dump({"case_seq1_r0:0": 0.5, "case_seq1_r0:1": 0.5}, f)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "acprof.packet.merge_packet_latency",
                    in_csv,
                    lat_json,
                    out_csv,
                ],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )

            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                output_fields = reader.fieldnames or []

        self.assertNotIn("sniff_group_id", output_fields)
        self.assertEqual(rows[0]["latency_s"], "0.500000")
        self.assertEqual(rows[0]["throughput_samples_per_s"], "4.000000")
        self.assertEqual(rows[0]["compute_mflops"], "400.000000")


if __name__ == "__main__":
    unittest.main()
