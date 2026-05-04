import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


class MergePacketLatencyComputeTests(unittest.TestCase):
    def test_recomputes_compute_mflops_from_packet_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")
            static_meta = os.path.join(tmp, "static_meta.csv")

            with open(static_meta, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["batch_size"])
                writer.writeheader()
                writer.writerow({"batch_size": "1"})

            fieldnames = [
                "sniff_group_id",
                "latency_s",
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
                [sys.executable, "merge_packet_latency.py", in_csv, lat_json, out_csv],
                check=True,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )

            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

        self.assertEqual(rows[0]["latency_s"], "0.250000")
        self.assertEqual(rows[0]["throughput_samples_per_s"], "4.000000")
        self.assertEqual(rows[0]["compute_mflops"], "800.000000")
        self.assertEqual(rows[0]["cpu_cycles_est_packet"], "750000000.000000")
        self.assertEqual(rows[0]["cpu_mips_packet"], "2.000000")
        self.assertNotIn("sniff_group_id", fieldnames)

    def test_merges_packet_latency_with_sidecar_when_csv_omits_sniff_group_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = os.path.join(tmp, "result.csv")
            lat_json = os.path.join(tmp, "lat.json")
            out_csv = os.path.join(tmp, "result.merged.csv")
            static_meta = os.path.join(tmp, "static_meta.csv")
            sidecar = f"{in_csv}.sniff_groups.jsonl"

            with open(static_meta, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["batch_size"])
                writer.writeheader()
                writer.writerow({"batch_size": "2"})

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
                [sys.executable, "merge_packet_latency.py", in_csv, lat_json, out_csv],
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
