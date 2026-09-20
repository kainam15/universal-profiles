"""运行离线 unittest；供本地和 CI 复用。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json  # noqa: E402 -- 脚本先设置仓库导入路径。


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


def iter_tests(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(ROOT / "tests"))
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-no-skips", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0, help="从 0 开始的分片编号")
    parser.add_argument("--shard-count", type=int, default=1, help="完整测试集的分片总数")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("需要 0 <= --shard-index < --shard-count")
    suite = unittest.TestSuite()
    discovery_counts = {}
    for pattern in args.pattern or ["test_*.py"]:
        discovered = unittest.defaultTestLoader.discover(args.directory, pattern=pattern)
        discovery_counts[pattern] = discovered.countTestCases()
        suite.addTests(discovered)
    tests = sorted(iter_tests(suite), key=lambda test: test.id())
    selected = tests[args.shard_index::args.shard_count]
    suite = unittest.TestSuite(selected)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    successful = bool(result.testsRun) and result.wasSuccessful() and all(discovery_counts.values())
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
        "discovery_counts": discovery_counts,
        "shard": {
            "index": args.shard_index, "count": args.shard_count,
            "discovered": len(tests), "selected": len(selected),
            "suite_sha256": hashlib.sha256("\n".join(test.id() for test in tests).encode()).hexdigest(),
        },
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
