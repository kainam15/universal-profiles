import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from acprof.cli import plot
from acprof.config import CSV_FIELDS
from acprof.host import client
from acprof.host.orchestrator import _write_case_error_csv
from acprof.packet import merge_packet_latency


class PixelNormalizationTests(unittest.TestCase):
    def write_result(self, root, *, family="diffusion", scale_type="resolution_px",
                     batch=2, entries=None, rows=None, workload=None):
        root = Path(root)
        if entries is None:
            entries = [
                {"input_scale": side, "input_metadata": {
                    "output_pixel_count_per_image": pixels,
                }}
                for side, pixels in ((128, 16384), (256, 65536))
            ]
        plan = {"schema_version": 2, "task_family": family, "entries": entries}
        plan_bytes = json.dumps(plan).encode()
        (root / "input_scale_plan.json").write_bytes(plan_bytes)
        meta = {"task_family": family, "input_scale_type": scale_type,
                "batch_size": batch, "workload": workload or {},
                "input_scale_plan_sha256": hashlib.sha256(plan_bytes).hexdigest()}
        (root / "static_meta.json").write_text(json.dumps(meta))
        if rows is None:
            rows = [
                {"input_scale": 128, "input_units_per_request": 256,
                 "container_attributed_energy_eff_j": 327.68,
                 "container_attributed_j_per_input_unit": 1.28,
                 "latency_s": 0.032768, "latency_app_s": 0.065536},
                {"input_scale": 256, "input_units_per_request": 512,
                 "container_attributed_energy_eff_j": 1310.72,
                 "container_attributed_j_per_input_unit": 2.56,
                 "latency_s": 0.131072, "latency_app_s": 0.262144},
            ]
        path = root / "result_all.csv"
        pd.DataFrame([
            {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off",
             "status": "ok", "warmup": 0, **row}
            for row in rows
        ]).to_csv(path, index=False)
        return path

    def test_legacy_diffusion_uses_area_and_keeps_scale_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            originals = {p: p.read_bytes() for p in Path(tmp).iterdir()}
            df = plot.prepare_df(str(path))
            self.assertIn("output_pixels_per_request", df)
            self.assertEqual(df.output_pixels_per_request.tolist(), [32768, 131072])
            self.assertEqual(df.input_units_per_request.tolist(), [256, 512])
            self.assertEqual(df.container_attributed_j_per_input_unit.tolist(), [1.28, 2.56])
            self.assertEqual(df.container_attributed_j_per_output_megapixel.tolist(), [10000, 10000])
            self.assertEqual(df.latency_s_per_output_megapixel.tolist(), [1, 1])
            self.assertEqual(df.latency_app_s_per_output_megapixel.tolist(), [2, 2])
            self.assertTrue(df.input_pixels_per_request.isna().all())
            for p, before in originals.items():
                self.assertEqual(p.read_bytes(), before)

    def test_cv_video_uses_dimensions_and_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="cv", scale_type="resolution_scale", batch=1,
                entries=[{"input_scale": 0.5, "input_metadata": {
                    "image_width": 112, "image_height": 112, "num_frames": 3}}],
                rows=[{"input_scale": 0.5, "input_units_per_request": 0.5,
                       "container_attributed_energy_eff_j": 37.632}])
            df = plot.prepare_df(str(path))
        self.assertIn("input_pixels_per_request", df)
        self.assertEqual(df.input_pixels_per_request.tolist(), [37632])
        self.assertEqual(df.container_attributed_j_per_input_megapixel.tolist(), [1000])
        self.assertTrue(df.output_pixels_per_request.isna().all())

    def test_multimodal_uses_materialized_rectangular_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="multimodal", batch=2,
                entries=[{"input_scale": 20, "input_metadata": {
                    "image_width": 20, "image_height": 30}}],
                rows=[{"input_scale": 20, "container_attributed_energy_eff_j": 1.2}])
            df = plot.prepare_df(str(path))
        self.assertIn("input_pixels_per_request", df)
        self.assertEqual(df.input_pixels_per_request.tolist(), [1200])
        self.assertEqual(df.container_attributed_j_per_input_megapixel.tolist(), [1000])

    def test_diffusion_video_does_not_multiply_frames_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, entries=[{"input_scale": 128, "input_metadata": {
                "output_pixel_count_per_frame": 16384,
                "output_num_frames": 4, "output_pixel_count_per_video": 65536}}],
                rows=[{"input_scale": 128, "container_attributed_energy_eff_j": 1310.72}])
            df = plot.prepare_df(str(path))
        self.assertIn("output_pixels_per_request", df)
        self.assertEqual(df.output_pixels_per_request.tolist(), [131072])
        self.assertEqual(df.container_attributed_j_per_output_megapixel.tolist(), [10000])

    def test_non_image_scales_are_not_squared(self):
        for family, scale_type in (("nlp", "seq_length"), ("audio", "duration_s"),
                                   ("diffusion", "denoising_steps")):
            with self.subTest(scale_type=scale_type), tempfile.TemporaryDirectory() as tmp:
                path = self.write_result(tmp, family=family, scale_type=scale_type,
                    entries=[{"input_scale": 128, "input_metadata": {}}])
                df = plot.prepare_df(str(path))
                self.assertIn("output_pixels_per_request", df)
                self.assertTrue(df.output_pixels_per_request.isna().all())
                self.assertTrue(df.input_pixels_per_request.isna().all())
                self.assertEqual(df.container_attributed_j_per_input_unit.tolist(), [1.28, 2.56])

    def test_untrusted_or_unmatched_plan_leaves_pixels_unknown(self):
        for failure in ("hash", "scale", "batch", "missing"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                path = self.write_result(tmp)
                meta_path = Path(tmp) / "static_meta.json"
                meta = json.loads(meta_path.read_text())
                if failure == "hash":
                    meta["input_scale_plan_sha256"] = "0" * 64
                elif failure == "batch":
                    meta.pop("batch_size")
                elif failure == "scale":
                    rows = pd.read_csv(path)
                    rows["input_scale"] = [192, 320]
                    rows.to_csv(path, index=False)
                else:
                    (Path(tmp) / "input_scale_plan.json").unlink()
                meta_path.write_text(json.dumps(meta))
                df = plot.prepare_df(str(path))
                self.assertIn("output_pixels_per_request", df)
                self.assertTrue(df.output_pixels_per_request.isna().all())

    def test_explicit_pixel_counts_work_without_sidecars_and_refresh_stale_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, rows=[{
                "input_scale": 128, "output_pixels_per_request": 32768,
                "container_attributed_energy_eff_j": 327.68,
                "container_attributed_j_per_output_megapixel": 999,
                "latency_s": 0.032768, "latency_s_per_output_megapixel": 999}])
            (Path(tmp) / "static_meta.json").unlink()
            (Path(tmp) / "input_scale_plan.json").unlink()
            df = plot.prepare_df(str(path))
        self.assertEqual(df.container_attributed_j_per_output_megapixel.tolist(), [10000])
        self.assertEqual(df.latency_s_per_output_megapixel.tolist(), [1])

    def test_legacy_rows_with_empty_new_columns_can_use_the_verified_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            rows = pd.read_csv(path)
            rows["input_pixels_per_request"] = ""
            rows["output_pixels_per_request"] = ""
            rows.to_csv(path, index=False)
            df = plot.prepare_df(str(path))
        self.assertEqual(df.output_pixels_per_request.tolist(), [32768, 131072])
        self.assertEqual(df.container_attributed_j_per_output_megapixel.tolist(), [10000, 10000])

    def test_invalid_counts_and_energy_remain_nan_but_zero_energy_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, rows=[
                {"input_scale": 128, "output_pixels_per_request": pixels,
                 "container_attributed_energy_eff_j": energy}
                for pixels, energy in ((16384, 0), (0, 1), (-1, 1),
                                       (16384, -1), (16384, float("nan")), (1.5, 1))
            ])
            df = plot.prepare_df(str(path))
        values = df.container_attributed_j_per_output_megapixel.tolist()
        self.assertEqual(values[0], 0)
        self.assertTrue(all(math.isnan(value) for value in values[1:]))

    def test_resolution_panel_does_not_fall_back_to_edge_when_geometry_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            (Path(tmp) / "input_scale_plan.json").unlink()
            df = plot.prepare_df(str(path))
        title, _, rows, columns, panels, shared = next(
            spec for spec in plot.METRIC_OVERVIEW_PLOTS
            if spec[1] == "service_efficiency_overview_vs_scale.png")
        try:
            with patch.object(plot.plt, "close"):
                plot.plot_metric_overview(df, panels=panels, rows=rows,
                    columns=columns, shared_y_groups=shared, title=title,
                    xlabel="resolution_px", out_png=None)
                axis = plot.plt.gcf().axes[3]
            self.assertEqual(len(axis.lines), 0)
            self.assertIn("No data", [text.get_text() for text in axis.texts])
            self.assertIn("Output Megapixel", axis.get_title())
        finally:
            plot.plt.close("all")

    def test_non_image_panel_keeps_the_task_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="audio", scale_type="duration_s", entries=[])
            df = plot.prepare_df(str(path))
        try:
            with patch.object(plot.plt, "close"):
                plot.plot_metric(df, "container_attributed_j_per_input_unit",
                    "Energy per Input Unit", "J/input unit", "duration_s", None)
                axis = plot.plt.gca()
            self.assertEqual(list(axis.lines[0].get_ydata()), [1.28, 2.56])
            self.assertEqual(axis.get_ylabel(), "J/input unit")
        finally:
            plot.plt.close("all")

    def test_malformed_plan_does_not_block_other_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            plan_path = Path(tmp) / "input_scale_plan.json"
            content = json.dumps({"schema_version": 2, "entries": 42}).encode()
            plan_path.write_bytes(content)
            meta_path = Path(tmp) / "static_meta.json"
            meta = json.loads(meta_path.read_text())
            meta["input_scale_plan_sha256"] = hashlib.sha256(content).hexdigest()
            meta_path.write_text(json.dumps(meta))
            with self.assertWarnsRegex(RuntimeWarning, "input_scale_plan"):
                df = plot.prepare_df(str(path))
        self.assertTrue(df.output_pixels_per_request.isna().all())
        self.assertEqual(df.container_attributed_energy_eff_j.tolist(), [327.68, 1310.72])

    def test_energy_and_latency_panels_plot_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = plot.prepare_df(str(self.write_result(tmp)))
        for filename, index, expected, unit in (
            ("service_efficiency_overview_vs_scale.png", 3, [10000, 10000], "J/Mpixel"),
            ("latency_overview_vs_scale.png", 2, [1, 1], "s/Mpixel"),
        ):
            with self.subTest(filename=filename):
                title, _, rows, columns, panels, shared = next(
                    spec for spec in plot.METRIC_OVERVIEW_PLOTS if spec[1] == filename)
                try:
                    with patch.object(plot.plt, "close"):
                        plot.plot_metric_overview(df, panels=panels, rows=rows,
                            columns=columns, shared_y_groups=shared, title=title,
                            xlabel="resolution_px", out_png=None)
                        axis = plot.plt.gcf().axes[index]
                    self.assertIn("Output Megapixel", axis.get_title())
                    self.assertIn(unit, axis.get_ylabel())
                    self.assertEqual(list(axis.lines[0].get_ydata()), expected)
                finally:
                    plot.plt.close("all")

    def test_live_client_writes_pixels_and_application_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "result.csv")
            entry = {"input_scale": 128.0, "scale_label": "res128px", "payload": {},
                     "input_metadata": {"output_pixel_count_per_image": 16384}}
            with patch.multiple(client, OUT_CSV=path, WARMUP=0, REPEAT=1,
                    REPEAT_IN_WINDOW=2, BATCH_SIZE=2, TASK_FAMILY="diffusion",
                    USE_ENERGY=False, USE_MIPS=False, energy_mod=None,
                    cpu_energy_mod=None, resource_usage_mod=None,
                    input_scale_entries=[entry]), patch.object(client.requests, "get",
                    return_value=SimpleNamespace(status_code=200, text="ok")), patch.object(
                    client, "_one_request", return_value={"latency_app_s": 0.065536,
                    "effective_input_scale": 128.0}):
                client.main()
            with open(path) as f:
                row = next(csv.DictReader(f))
        self.assertIn("output_pixels_per_request", row)
        self.assertEqual(row["output_pixels_per_request"], "32768.000000")
        self.assertEqual(row["input_units_per_request"], "256.000000")
        self.assertEqual(row["latency_app_s_per_output_megapixel"], "2.000000")
        self.assertEqual(row["latency_s_per_output_megapixel"], "nan")
        self.assertEqual(row["container_attributed_j_per_output_megapixel"], "nan")

    def test_packet_merge_recomputes_pixel_latency_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "result.csv", root / "merged.csv"
            pd.DataFrame([{"sniff_group_id": "case", "input_scale": 128,
                "output_pixels_per_request": 32768,
                "latency_s": 99, "latency_s_per_output_megapixel": 99,
                "latency_app_s_per_output_megapixel": 2,
                "container_attributed_j_per_output_megapixel": 10000}]).to_csv(source, index=False)
            latencies = root / "latencies.json"
            latencies.write_text(json.dumps({"case:1": 0.032768}))
            merge_packet_latency.main([str(source), str(latencies), str(destination)])
            with destination.open() as f:
                row = next(csv.DictReader(f))
        self.assertEqual(row["latency_s_per_output_megapixel"], "1.000000")
        self.assertEqual(row["latency_app_s_per_output_megapixel"], "2")
        self.assertEqual(row["container_attributed_j_per_output_megapixel"], "10000")

    def test_partial_legacy_case_preserves_measurements_without_optional_pixel_columns(self):
        fields = [field for field in CSV_FIELDS
                  if not field.endswith(("_pixels_per_request", "_megapixel"))]
        original = dict.fromkeys(fields, "nan")
        original.update(input_scale="128", repeat_idx="0", warmup="0", status="ok",
                        error="", container_attributed_energy_eff_j="327.680000")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.csv"
            with path.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(original)
            preserved, added = _write_case_error_csv(
                task_info=SimpleNamespace(task_family="diffusion"), out_csv=str(path),
                cpu=1, mem=4, gpu="off", warmup=0, repeat=1, repeat_in_window=1,
                input_scales="128,256", error="request failed", preserve_existing=True)
            with path.open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual((preserved, added), (1, 1))
        self.assertEqual({field: rows[0][field] for field in fields}, original)
        self.assertEqual(rows[0]["output_pixels_per_request"], "nan")
        self.assertEqual(rows[1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
