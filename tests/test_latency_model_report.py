import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from acprof.cli import plot


class LatencyModelReportTests(unittest.TestCase):
    def test_plot_main_writes_latency_model_report_and_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "result_all.csv")
            fieldnames = [
                "cpu_cores",
                "mem_cap_gb",
                "gpu_mode",
                "input_scale",
                "warmup",
                "status",
                "latency_s",
            ]
            rows = []
            for cpu in (1, 2):
                for mem in (4, 8):
                    for gpu_mode in ("off", "on"):
                        for input_scale in (64, 128, 256):
                            latency = (
                                0.05
                                + input_scale * 0.001
                                + (0.2 / cpu)
                                - (0.002 * mem)
                                - (0.03 if gpu_mode == "on" else 0.0)
                            )
                            rows.append({
                                "cpu_cores": str(cpu),
                                "mem_cap_gb": str(mem),
                                "gpu_mode": gpu_mode,
                                "input_scale": str(input_scale),
                                "warmup": "0",
                                "status": "ok",
                                "latency_s": f"{latency:.6f}",
                            })

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with open(
                os.path.join(tmp, "static_meta.csv"),
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["model_name", "task_family", "input_scale_type"],
                )
                writer.writeheader()
                writer.writerow({
                    "model_name": "google-bert/bert-base-uncased",
                    "task_family": "nlp",
                    "input_scale_type": "seq_length",
                })

            with patch.object(sys, "argv", ["plot.py", csv_path]), patch.object(
                plot,
                "plot_metric",
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

            report_path = os.path.join(tmp, "latency_model_report.json")
            residuals_path = os.path.join(tmp, "latency_model_residuals.csv")

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            with open(residuals_path, "r", encoding="utf-8", newline="") as f:
                residual_rows = list(csv.DictReader(f))

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["target_metric"], "latency_s")
        self.assertEqual(report["model_name"], "google-bert/bert-base-uncased")
        self.assertEqual(report["task_family"], "nlp")
        self.assertGreaterEqual(report["train_rows"], 16)
        self.assertGreaterEqual(report["test_rows"], 4)
        self.assertIn("r2", report["metrics"]["test"])
        self.assertLess(report["metrics"]["test"]["rmse"], 1e-9)
        self.assertEqual(len(residual_rows), 24)
        self.assertIn("predicted_latency_s", residual_rows[0])
        self.assertIn("residual_s", residual_rows[0])


if __name__ == "__main__":
    unittest.main()
