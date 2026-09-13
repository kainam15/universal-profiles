"""验证 CI 的退出状态和跳过证据，避免空测试/缺依赖伪装为通过。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ValidationRunnerTests(unittest.TestCase):
    def run_fixture(self, body, *options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_sample.py").write_text("import unittest\n" + body)
            report = root / "report.json"
            process = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_tests.py"),
                "--directory", directory, "--report", str(report), *options,
            ], capture_output=True, text=True)
            return process, json.loads(report.read_text()) if report.exists() else None

    def test_missing_runtime_fails_strict_job_and_records_reason(self):
        process, report = self.run_fixture(
            "class Sample(unittest.TestCase):\n"
            "    @unittest.skip('no runtime')\n"
            "    def test_runtime(self): pass\n", "--require-no-skips",
        )
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(report["counts"]["skipped"], 1)
        self.assertFalse(report["successful"])
        self.assertEqual(report["tests"][0]["reason"], "no runtime")

    def test_optional_skip_remains_visible(self):
        process, report = self.run_fixture(
            "class Sample(unittest.TestCase):\n"
            "    @unittest.skip('no runtime')\n"
            "    def test_runtime(self): pass\n",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIsNotNone(report)
        self.assertTrue(report["successful"])
        self.assertEqual(report["counts"]["passed"], 0)

    def test_empty_discovery_is_failure(self):
        process, report = self.run_fixture("")
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(report["counts"]["run"], 0)

    def test_failed_subtest_is_failure_and_has_evidence(self):
        process, report = self.run_fixture(
            "class Sample(unittest.TestCase):\n"
            "    def test_values(self):\n"
            "        with self.subTest(value=2): self.assertEqual(1, 2)\n",
        )
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertIsNotNone(report)
        self.assertIn("1 != 2", report["tests"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
