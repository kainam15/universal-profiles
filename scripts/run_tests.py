"""运行离线 unittest；供本地和 CI 复用。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}
        self.started = {}

    def record(self, test, outcome, reason=""):
        self.records[test.id()] = {
            "id": test.id(), "outcome": outcome, "reason": reason,
            "duration_s": time.perf_counter() - self.started.get(test.id(), time.perf_counter()),
        }

    def startTest(self, test):
        self.started[test.id()] = time.perf_counter()
        super().startTest(test)

    def addSuccess(self, test):
        self.record(test, "passed")
        super().addSuccess(test)

    def addSkip(self, test, reason):
        self.record(test, "skipped", reason)
        super().addSkip(test, reason)

    def addFailure(self, test, err):
        self.record(test, "failed", self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        self.record(test, "error", self._exc_info_to_string(err, test))
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            reason = self.records.get(test.id(), {}).get("reason", "")
            reason += f"{subtest.id()}\n{self._exc_info_to_string(err, subtest)}"
            self.record(test, "failed", reason)
        super().addSubTest(test, subtest, err)

    def addExpectedFailure(self, test, err):
        self.record(test, "expected_failure", self._exc_info_to_string(err, test))
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.record(test, "unexpected_success")
        super().addUnexpectedSuccess(test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(ROOT / "tests"))
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args(argv)
    suite = unittest.TestSuite()
    for pattern in args.pattern or ["test_*.py"]:
        suite.addTests(unittest.defaultTestLoader.discover(args.directory, pattern=pattern))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    successful = bool(result.testsRun) and result.wasSuccessful()
    if args.require_no_skips and (result.skipped or result.expectedFailures):
        successful = False
    atomic_write_json(args.report, {
        "schema_version": 1, "successful": successful,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": time.perf_counter() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                for dist in importlib.metadata.distributions())),
        "require_no_skips": args.require_no_skips,
        "patterns": args.pattern or ["test_*.py"],
        "counts": {
            "run": result.testsRun,
            "passed": sum(item["outcome"] == "passed" for item in result.records.values()),
            "failed": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses),
        },
        "tests": list(result.records.values()),
    })
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
