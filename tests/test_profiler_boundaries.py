"""验证旧解析入口及新纯模块的依赖边界。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ProfilerBoundaryTests(unittest.TestCase):
    def test_compatibility_parsers_have_pure_owners(self):
        from acprof.host import compute_profile, execution_profile
        self.assertEqual(compute_profile.parse_ncu_profile_csv.__module__, "acprof.host.profilers.compute_parsers")
        self.assertEqual(execution_profile.parse_massif_output.__module__, "acprof.host.profilers.execution_parsers")
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "massif.out"
            report.write_text("time_unit: ms\nsnapshot=0\ntime=5\nmem_heap_B=10\nmem_heap_extra_B=2\nmem_stacks_B=3\n")
            values = execution_profile.parse_massif_output(str(report))
            self.assertEqual(values["cpu_heap_peak_total_bytes_massif"], 15)

    def test_pure_parser_imports_do_not_load_hardware_or_orchestration(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from acprof.host.profilers import compute_parsers, execution_parsers; "
            "assert 'acprof.host.profiler_common' not in sys.modules; "
            "assert 'acprof.host.detect' not in sys.modules; assert 'torch' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
