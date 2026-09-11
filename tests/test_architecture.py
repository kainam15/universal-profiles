"""Import boundaries that keep application entry points out of core code."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_shared_preflight_does_not_depend_on_run_cli(self):
        for name in ("probe", "posthoc"):
            path = PROJECT_DIR / "acprof" / "cli" / f"{name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            dependencies = {
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            with self.subTest(module=name):
                self.assertNotIn("acprof.cli.run", dependencies)

    def test_implementation_packages_do_not_import_cli(self):
        for package in ("host", "container", "workloads", "monitors", "packet",
                        "analysis", "plotting", "tui"):
            for path in (PROJECT_DIR / "acprof" / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                dependencies = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        dependencies.append(node.module)
                    elif isinstance(node, ast.Import):
                        dependencies.extend(alias.name for alias in node.names)
                with self.subTest(module=str(path.relative_to(PROJECT_DIR))):
                    self.assertFalse(
                        any(name == "acprof.cli" or name.startswith("acprof.cli.")
                            for name in dependencies),
                        dependencies,
                    )

    def test_analysis_import_does_not_load_rendering(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; "
                "import acprof.analysis.latency_report; "
                "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
                "for name in sys.modules); "
                "assert 'acprof.cli.plot' not in sys.modules"
            )],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_metrics_import_does_not_initialize_measurement(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; "
                "import acprof.host.client_metrics; "
                "assert 'acprof.host.client' not in sys.modules; "
                "assert not any(name.startswith('acprof.monitors') "
                "for name in sys.modules)"
            )],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tui_commands_import_does_not_load_textual(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; import acprof.tui.commands; "
                "assert not any(name == 'textual' or name.startswith('textual.') "
                "for name in sys.modules)"
            )],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
