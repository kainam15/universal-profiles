import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acprof.config import CSV_FIELDS
from acprof.host.orchestrator import merge_all_csvs


class ResultMergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.destination = self.directory / "result_all.csv"
        self.destination.write_bytes(b"previous result\n")

    def source(self, name="case.csv", *, rows=None, extra_fields=()):
        path = self.directory / name
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64",
                   warmup="0", repeat_idx="0", status="ok", error="")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*CSV_FIELDS, *extra_fields])
            writer.writeheader()
            writer.writerows(rows if rows is not None else [row])
        return path

    def assert_previous_result(self):
        self.assertEqual(self.destination.read_bytes(), b"previous result\n")

    def test_missing_source_rejected_before_replacing_previous_result(self):
        source = self.source()
        with self.assertRaisesRegex((ValueError, RuntimeError, FileNotFoundError), "missing|exist"):
            merge_all_csvs([str(source), str(self.directory / "missing.csv")], str(self.destination))
        self.assert_previous_result()

    def test_duplicate_source_rejected(self):
        source = self.source()
        with self.assertRaisesRegex((ValueError, RuntimeError), "duplicate"):
            merge_all_csvs([str(source), str(source)], str(self.destination))
        self.assert_previous_result()

    def test_duplicate_measurement_across_distinct_files_rejected(self):
        first, second = self.source("first.csv"), self.source("second.csv")
        with self.assertRaisesRegex((ValueError, RuntimeError), "duplicate"):
            merge_all_csvs([str(first), str(second)], str(self.destination))
        self.assert_previous_result()

    def test_write_failure_preserves_previous_result_and_removes_temporary_file(self):
        source = self.source()
        before = set(self.directory.iterdir())
        with patch("csv.DictWriter.writerows", side_effect=OSError("injected disk failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_publish_failure_preserves_previous_result_and_removes_temporary_file(self):
        source = self.source()
        before = set(self.directory.iterdir())
        with patch("os.replace", side_effect=OSError("injected publish failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_malformed_row_rejected(self):
        source = self.source()
        with source.open("a") as stream:
            stream.write("truncated,row\n")
        with self.assertRaisesRegex((ValueError, RuntimeError), "row|column"):
            merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()

    def test_empty_case_rejected(self):
        source = self.source(rows=[])
        with self.assertRaisesRegex((ValueError, RuntimeError), "empty|no.*rows"):
            merge_all_csvs([str(source)], str(self.destination))
        self.assert_previous_result()

    def test_success_preserves_file_permissions_and_unknown_historical_fields(self):
        source = self.source(extra_fields=("legacy_metric",))
        self.destination.chmod(0o640)
        merge_all_csvs([str(source)], str(self.destination))
        with self.destination.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertIn("legacy_metric", reader.fieldnames)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_scale"], "64")
        self.assertEqual(os.stat(self.destination).st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main()
