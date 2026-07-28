import csv
import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from acprof.cli import plot


class LatencyModelReportTests(unittest.TestCase):
    FIELDNAMES = [
        "cpu_cores",
        "mem_cap_gb",
        "gpu_mode",
        "input_scale",
        "repeat_idx",
        "warmup",
        "status",
        "latency_s",
    ]

    @staticmethod
    def _base_latency(
        cpu: int,
        mem: int,
        gpu_mode: str,
        input_scale: int,
    ) -> float:
        log_scale = math.log(input_scale)
        log_mem = math.log(mem)
        if gpu_mode == "off":
            log_cpu = math.log(cpu)
            log_latency = (
                -4.0
                + 0.90 * log_scale
                - 0.65 * log_cpu
                + 0.02 * log_mem
                - 0.04 * log_scale * log_cpu
            )
        else:
            inverse_cpu = 1.0 / cpu
            log_latency = (
                -7.0
                + 0.55 * log_scale
                + 1.20 * inverse_cpu
                + 0.01 * log_mem
                - 0.06 * log_scale * inverse_cpu
            )
        return math.exp(log_latency)

    def _rows(
        self,
        gpu_modes: tuple[str, ...] = ("off", "on"),
        break_one_gpu_max_scale_case: bool = False,
        break_gpu_resource_config: bool = False,
    ) -> list[dict[str, str]]:
        rows = []
        for cpu in (1, 2, 4, 8):
            for mem in (4, 8, 16):
                for gpu_mode in gpu_modes:
                    for input_scale in (64, 128, 256, 512):
                        latency = self._base_latency(
                            cpu,
                            mem,
                            gpu_mode,
                            input_scale,
                        )
                        if (
                            break_one_gpu_max_scale_case
                            and gpu_mode == "on"
                            and cpu == 8
                            and mem == 16
                            and input_scale == 512
                        ):
                            latency *= 1.5
                        if (
                            break_gpu_resource_config
                            and gpu_mode == "on"
                            and cpu == 8
                            and mem == 16
                        ):
                            latency *= 0.5
                        for repeat_idx, repeat_factor in enumerate((0.99, 1.0, 1.01)):
                            rows.append({
                                "cpu_cores": str(cpu),
                                "mem_cap_gb": str(mem),
                                "gpu_mode": gpu_mode,
                                "input_scale": str(input_scale),
                                "repeat_idx": str(repeat_idx),
                                "warmup": "0",
                                "status": "ok",
                                "latency_s": f"{latency * repeat_factor:.12f}",
                            })
        return rows

    def _write_fixture(
        self,
        output_dir: str,
        rows: list[dict[str, str]],
    ) -> str:
        csv_path = os.path.join(output_dir, "result_all.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        with open(
            os.path.join(output_dir, "static_meta.csv"),
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
        return csv_path

    @staticmethod
    def _read_artifacts(output_dir: str) -> tuple[dict, list[dict[str, str]]]:
        with open(
            os.path.join(output_dir, "latency_model_report.json"),
            "r",
            encoding="utf-8",
        ) as f:
            report = json.load(f)
        with open(
            os.path.join(output_dir, "latency_model_residuals.csv"),
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            residual_rows = list(csv.DictReader(f))
        return report, residual_rows

    def test_plot_main_writes_group_validated_positive_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())

            with patch.object(sys, "argv", ["plot.py", csv_path]), patch.object(
                plot,
                "plot_metric",
            ), patch.object(plot, "plot_cold_start_bar"):
                plot.main()

            report, residual_rows = self._read_artifacts(tmp)

        self.assertEqual(report["report_schema_version"], 2)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["prediction_ready"])
        self.assertTrue(report["positive_prediction_form"])
        self.assertEqual(report["target_metric"], "latency_s")
        self.assertEqual(report["model_name"], "google-bert/bert-base-uncased")
        self.assertEqual(report["task_family"], "nlp")
        self.assertEqual(report["raw_rows"], 288)
        self.assertEqual(report["case_rows"], 96)
        self.assertEqual(report["aggregation"]["target_statistic"], "median")
        self.assertFalse(
            report["aggregation"]["repetitions_split_across_train_and_test"]
        )
        self.assertEqual(set(report["models"]), {"cpu", "gpu"})

        for hardware_model in ("cpu", "gpu"):
            model_report = report["models"][hardware_model]
            self.assertEqual(model_report["status"], "ok")
            self.assertTrue(model_report["prediction_ready"])
            self.assertIn(
                "log_input_scale_x_",
                " ".join(model_report["feature_columns"]),
            )
            config_validation = model_report["validation"][
                "resource_configuration_holdout"
            ]
            scale_validation = model_report["validation"]["input_scale_holdout"]
            self.assertTrue(config_validation["available"])
            self.assertEqual(
                config_validation["train_test_group_overlap_count"],
                0,
            )
            self.assertTrue(scale_validation["available"])
            self.assertTrue(scale_validation["strict_extrapolation"])
            self.assertLess(
                scale_validation["train_input_scale_max"],
                scale_validation["test_input_scale_min"],
            )
            self.assertGreater(
                model_report["metrics"]["resource_configuration_holdout"]["r2"],
                0.99,
            )
            self.assertGreater(
                model_report["metrics"]["input_scale_holdout"]["r2"],
                0.99,
            )
            self.assertEqual(
                model_report["metrics"]["resource_configuration_holdout"][
                    "nonpositive_prediction_count"
                ],
                0,
            )

        self.assertEqual(len(residual_rows), 96)
        self.assertTrue(
            all(row["split"] == "out_of_fold_test" for row in residual_rows)
        )
        self.assertTrue(all(int(row["repeat_count"]) == 3 for row in residual_rows))
        self.assertTrue(
            all(float(row["predicted_latency_s"]) > 0.0 for row in residual_rows)
        )
        self.assertTrue(
            all(row["report_schema_version"] == "2" for row in residual_rows)
        )
        for row in residual_rows:
            self.assertEqual(
                row["predicted_latency_s"],
                row["resource_config_oof_predicted_latency_s"],
            )
            self.assertAlmostEqual(
                float(row["residual_s"]),
                float(row["latency_s"]) - float(row["predicted_latency_s"]),
                places=8,
            )
            if float(row["input_scale"]) == 512.0:
                self.assertNotEqual(
                    row["max_scale_holdout_predicted_latency_s"],
                    "",
                )
            else:
                self.assertEqual(
                    row["max_scale_holdout_predicted_latency_s"],
                    "",
                )
        unique_cases = {
            (
                row["hardware_model"],
                row["cpu_cores"],
                row["mem_cap_gb"],
                row["input_scale"],
            )
            for row in residual_rows
        }
        self.assertEqual(len(unique_cases), len(residual_rows))

    def test_gpu_only_matrix_no_longer_fails_as_singular(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(gpu_modes=("on",))
            csv_path = self._write_fixture(tmp, rows)
            df = plot.prepare_df(csv_path)
            plot.write_latency_model_report(df, plot.read_static_meta(csv_path), tmp)
            report, residual_rows = self._read_artifacts(tmp)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(set(report["models"]), {"gpu"})
        gpu_report = report["models"]["gpu"]
        self.assertEqual(gpu_report["status"], "ok")
        self.assertEqual(gpu_report["quality_gate"]["failures"], [])
        config_validation = gpu_report["validation"][
            "resource_configuration_holdout"
        ]
        self.assertEqual(config_validation["folds"], 12)
        self.assertEqual(config_validation["completed_folds"], 12)
        self.assertEqual(config_validation["train_test_group_overlap_count"], 0)
        self.assertEqual(
            gpu_report["metrics"]["resource_configuration_holdout"][
                "prediction_count"
            ],
            48,
        )
        self.assertEqual(
            gpu_report["metrics"]["input_scale_holdout"]["prediction_count"],
            12,
        )
        for validation_name in (
            "resource_configuration_holdout",
            "input_scale_holdout",
        ):
            metrics = gpu_report["metrics"][validation_name]
            self.assertEqual(metrics["nonfinite_prediction_count"], 0)
            self.assertEqual(metrics["nonpositive_prediction_count"], 0)
        self.assertEqual(len(residual_rows), 48)
        self.assertTrue(
            all(float(row["fitted_predicted_latency_s"]) > 0.0 for row in residual_rows)
        )

    def test_bad_gpu_extrapolation_cannot_hide_behind_cpu_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(break_one_gpu_max_scale_case=True)
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(df, plot.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        self.assertEqual(report["models"]["cpu"]["status"], "ok")
        gpu_report = report["models"]["gpu"]
        pooled_metrics = gpu_report["metrics"]["input_scale_holdout"]
        self.assertGreaterEqual(
            pooled_metrics["r2"],
            plot.LATENCY_MODEL_MIN_VALIDATION_R2,
        )
        self.assertLessEqual(
            pooled_metrics["relative_mae"],
            plot.LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE,
        )
        self.assertGreater(
            gpu_report["validation"]["input_scale_holdout"][
                "worst_case_relative_error"
            ],
            plot.LATENCY_MODEL_MAX_SCALE_CASE_RELATIVE_ERROR,
        )
        self.assertEqual(gpu_report["status"], "poor_fit")
        self.assertEqual(report["status"], "poor_fit")
        self.assertFalse(report["prediction_ready"])
        self.assertFalse(report["quality_gate"]["passed"])
        self.assertTrue(
            any(
                "gpu: input_scale_holdout" in failure
                for failure in report["quality_gate"]["failures"]
            )
        )

    def test_bad_held_out_configuration_cannot_hide_in_pooled_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(break_gpu_resource_config=True)
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(df, plot.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        pooled_metrics = gpu_report["metrics"]["resource_configuration_holdout"]
        self.assertGreaterEqual(
            pooled_metrics["r2"],
            plot.LATENCY_MODEL_MIN_VALIDATION_R2,
        )
        self.assertLessEqual(
            pooled_metrics["relative_mae"],
            plot.LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE,
        )
        self.assertGreater(
            gpu_report["validation"]["resource_configuration_holdout"][
                "worst_fold_relative_mae"
            ],
            plot.LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE,
        )
        self.assertEqual(gpu_report["status"], "poor_fit")
        self.assertEqual(report["status"], "poor_fit")
        self.assertFalse(report["prediction_ready"])
        self.assertTrue(
            any(
                "held-out configuration fold" in failure
                for failure in gpu_report["quality_gate"]["failures"]
            )
        )

    def test_two_scales_are_not_misreported_as_valid_extrapolation(self) -> None:
        rows = [
            row
            for row in self._rows(gpu_modes=("on",))
            if int(row["input_scale"]) in (64, 128)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(df, plot.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        scale_validation = gpu_report["validation"]["input_scale_holdout"]
        self.assertFalse(scale_validation["available"])
        self.assertIn("at least 3 input scales", scale_validation["reason"])
        self.assertEqual(gpu_report["status"], "unvalidated")
        self.assertEqual(report["status"], "unvalidated")
        self.assertFalse(report["prediction_ready"])

    def test_invalid_gpu_mode_skips_and_replaces_stale_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(gpu_modes=("on",))
            for row in rows:
                row["gpu_mode"] = "mystery-device"
            csv_path = self._write_fixture(tmp, rows)
            residuals_path = os.path.join(tmp, "latency_model_residuals.csv")
            with open(residuals_path, "w", encoding="utf-8") as f:
                f.write("stale-marker\n")

            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(df, plot.read_static_meta(csv_path), tmp)
            report, residual_rows = self._read_artifacts(tmp)
            with open(residuals_path, "r", encoding="utf-8") as f:
                residual_text = f.read()

        self.assertEqual(report["status"], "skipped")
        self.assertFalse(report["prediction_ready"])
        self.assertIn("unsupported gpu_mode", report["reason"])
        self.assertEqual(residual_rows, [])
        self.assertNotIn("stale-marker", residual_text)
        self.assertIn("report_schema_version", residual_text.splitlines()[0])

    def test_residual_plot_writes_diagnostic_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())
            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(
                df,
                plot.read_static_meta(csv_path),
                tmp,
            )
            residuals_path = os.path.join(
                tmp,
                plot.LATENCY_MODEL_RESIDUALS,
            )
            out_png = os.path.join(
                tmp,
                plot.LATENCY_MODEL_RESIDUAL_PLOT,
            )

            plotted = plot.plot_latency_model_residuals(
                residuals_path,
                out_png,
            )

            self.assertTrue(plotted)
            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 0)

    def test_fit_curve_plot_writes_cpu_and_gpu_configuration_curves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())
            df = pd.read_csv(csv_path)
            plot.write_latency_model_report(
                df,
                plot.read_static_meta(csv_path),
                tmp,
            )
            residuals_path = os.path.join(
                tmp,
                plot.LATENCY_MODEL_RESIDUALS,
            )
            report_path = os.path.join(
                tmp,
                plot.LATENCY_MODEL_REPORT,
            )
            out_png = os.path.join(
                tmp,
                plot.LATENCY_MODEL_FIT_CURVES_PLOT,
            )

            plotted = plot.plot_latency_model_fit_curves(
                residuals_path,
                report_path,
                out_png,
            )

            self.assertTrue(plotted)
            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 0)

    def test_residual_plot_skips_header_only_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            residuals_path = os.path.join(
                tmp,
                plot.LATENCY_MODEL_RESIDUALS,
            )
            with open(
                residuals_path,
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=plot.LATENCY_MODEL_RESIDUAL_FIELDS,
                )
                writer.writeheader()
            out_png = os.path.join(
                tmp,
                plot.LATENCY_MODEL_RESIDUAL_PLOT,
            )

            plotted = plot.plot_latency_model_residuals(
                residuals_path,
                out_png,
            )

            self.assertFalse(plotted)
            self.assertFalse(os.path.exists(out_png))


if __name__ == "__main__":
    unittest.main()
