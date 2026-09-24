"""Exercise the real bootstrap shell at its package manager and Docker boundaries."""
import json
import os
from pathlib import Path
import pty
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_TOOL = r'''
import json
import os
from pathlib import Path
import shutil
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["SETUP_TEST_LOG"], "a") as stream:
    stream.write(json.dumps({"tool": name, "args": args, "cwd": os.getcwd()}) + "\n")
if name == "uname":
    print({"-s": os.environ.get("SETUP_TEST_OS", "Linux"), "-m": "x86_64", "-r": "6.8.0"}[args[0]])
elif name == "docker":
    sys.exit(int(os.environ.get("SETUP_TEST_DOCKER_EXIT", "0")))
elif name == "curl":
    if os.environ.get("SETUP_TEST_DOWNLOAD_FAIL"):
        sys.exit(22)
    destination = Path(args[args.index("--output") + 1])
    destination.write_text('#!/bin/sh\nmkdir -p "$UV_INSTALL_DIR"\ncp "$SETUP_TEST_UV" "$UV_INSTALL_DIR/uv"\n')
elif name == "uv":
    if args[:2] == ["tool", "install"]:
        if os.environ.get("SETUP_TEST_INSTALL_FAIL"):
            sys.exit(9)
        destination = Path(os.environ["UV_TOOL_BIN_DIR"]) / "acprof"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(__file__, destination)
        destination.chmod(0o755)
    elif args == ["tool", "dir", "--bin"]:
        if os.environ.get("SETUP_TEST_TOOL_DIR_FAIL"):
            sys.exit(8)
        print(os.environ["UV_TOOL_BIN_DIR"])
    elif args != ["tool", "update-shell"]:
        sys.exit(98)
elif name == "acprof":
    if args[0] == "doctor":
        print("doctor prerequisite result")
        sys.exit(int(os.environ.get("SETUP_TEST_DOCTOR_EXIT", "0")))
    if args[0] != "tui":
        sys.exit(97)
    sys.exit(int(os.environ.get("SETUP_TEST_TUI_EXIT", "0")))
elif name == "python":
    sys.exit(int(os.environ.get("SETUP_TEST_TUI_EXIT", "0")))
else:
    sys.exit(96)
'''


@unittest.skipUnless(sys.platform == "linux", "Bootstrap targets native Linux")
class SetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="acprof-setup-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "checkout with spaces"
        self.checkout.mkdir()
        self.bin = self.root / "commands"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.environment = {
            **os.environ,
            "PATH": str(self.bin),
            "UV_INSTALL_DIR": str(self.root / "uv bin"),
            "UV_TOOL_DIR": str(self.root / "tools"),
            "UV_TOOL_BIN_DIR": str(self.root / "tool bin"),
            "SETUP_TEST_LOG": str(self.log),
            "TERM": "xterm-256color",
        }
        for name in ("bash", "sh", "cat", "dirname", "mkdir", "mktemp", "rm", "cp", "df", "date"):
            executable = shutil.which(name)
            if executable is None:
                raise RuntimeError(f"Test host is missing {name}")
            (self.bin / name).symlink_to(executable)
        for name in ("uv", "docker", "uname", "curl"):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
            path.chmod(0o755)
        self.environment["SETUP_TEST_UV"] = str(self.root / "uv")
        shutil.copyfile(self.bin / "uv", self.root / "uv")
        (self.root / "uv").chmod(0o755)
        for name in ("pyproject.toml", "requirements.lock"):
            shutil.copyfile(ROOT / name, self.checkout / name)
        self.config = self.checkout / ".env.local"
        self.config.write_text("KEEP_EXISTING_CONFIG=yes\n")

    def invoke(self, *arguments, terminal=False, **environment):
        self.assertTrue((ROOT / "setup.sh").is_file(), "clone 后应提供 setup.sh 安装入口")
        shutil.copyfile(ROOT / "setup.sh", self.checkout / "setup.sh")
        command = ["bash", str(self.checkout / "setup.sh"), *arguments]
        options = dict(cwd=self.root, env={**self.environment, **environment}, timeout=20)
        if not terminal:
            return subprocess.run(command, capture_output=True, text=True, **options)
        master, slave = pty.openpty()
        try:
            result = subprocess.run(command, stdin=slave, stdout=slave, stderr=slave, **options)
            os.close(slave)
            slave = None
            output = bytearray()
            while True:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                output.extend(data)
            return subprocess.CompletedProcess(command, result.returncode, output.decode(), "")
        finally:
            if slave is not None:
                os.close(slave)
            os.close(master)

    def calls(self, tool):
        if not self.log.exists():
            return []
        return [entry for line in self.log.read_text().splitlines()
                if (entry := json.loads(line))["tool"] == tool]

    def invoke_launcher(self, *arguments, **environment):
        launcher = self.checkout / "acprof-tui"
        shutil.copyfile(ROOT / "acprof-tui", launcher)
        return subprocess.run(
            ["bash", str(launcher), *arguments],
            cwd=self.root,
            env={**self.environment, **environment},
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_installed_launcher_works_without_project_venv_or_refreshed_path(self):
        installed = self.invoke("--no-tui", "--no-modify-path")
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        arguments = ("--model", "owner/model", "--preset", "smoke", "--output-dir", "results/with spaces")
        result = self.invoke_launcher(*arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls("acprof")[-1]["args"], ["tui", *arguments])
        self.assertEqual(self.calls("acprof")[-1]["cwd"], str(self.root))
        self.assertFalse((self.checkout / ".venv").exists())
        self.assertEqual(self.config.read_text(), "KEEP_EXISTING_CONFIG=yes\n")

    def test_launcher_uses_acprof_on_path_without_uv(self):
        shutil.copy2(self.bin / "uv", self.bin / "acprof")
        (self.bin / "uv").unlink()
        result = self.invoke_launcher("--preset", "smoke")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls("acprof")[0]["args"], ["tui", "--preset", "smoke"])
        self.assertEqual(self.calls("uv"), [])

    def test_launcher_prefers_project_venv_over_installed_acprof(self):
        source_python = self.checkout / ".venv" / "bin" / "python"
        source_python.parent.mkdir(parents=True)
        shutil.copy2(self.bin / "uv", source_python)
        shutil.copy2(self.bin / "uv", self.bin / "acprof")
        result = self.invoke_launcher("--output-dir", "results/with spaces")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls("python")[0]["args"], [
            "-m", "acprof.cli.tui", "--output-dir", "results/with spaces",
        ])
        self.assertEqual(self.calls("acprof"), [])
        self.assertEqual(self.calls("uv"), [])

    def test_launcher_finds_uv_in_its_install_directory(self):
        installed = self.invoke("--no-tui", "--no-modify-path")
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        uv_directory = Path(self.environment["UV_INSTALL_DIR"])
        uv_directory.mkdir()
        (self.bin / "uv").rename(uv_directory / "uv")
        result = self.invoke_launcher("--preset", "smoke")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls("acprof")[-1]["args"], ["tui", "--preset", "smoke"])

    def test_launcher_falls_back_when_project_python_is_broken(self):
        source_python = self.checkout / ".venv" / "bin" / "python"
        source_python.parent.mkdir(parents=True)
        source_python.symlink_to(self.root / "removed-python")
        shutil.copy2(self.bin / "uv", self.bin / "acprof")
        result = self.invoke_launcher("--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls("acprof")[0]["args"], ["tui", "--help"])

    def test_launcher_preserves_failure_status_without_retrying_another_environment(self):
        shutil.copy2(self.bin / "uv", self.bin / "acprof")
        for mode in ("installed", "source"):
            with self.subTest(mode=mode):
                if mode == "source":
                    source_python = self.checkout / ".venv" / "bin" / "python"
                    source_python.parent.mkdir(parents=True)
                    shutil.copy2(self.bin / "uv", source_python)
                installed_calls = len(self.calls("acprof"))
                result = self.invoke_launcher("--preset", "smoke", SETUP_TEST_TUI_EXIT="17")
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(len(self.calls("acprof")), installed_calls + (mode == "installed"))
                self.assertEqual(len(self.calls("python")), int(mode == "source"))
                self.assertEqual(self.calls("uv"), [])

    def test_launcher_without_installation_gives_setup_guidance(self):
        for mode in ("empty", "query_failure", "missing_uv"):
            with self.subTest(mode=mode):
                if mode == "missing_uv":
                    (self.bin / "uv").unlink()
                result = self.invoke_launcher(SETUP_TEST_TOOL_DIR_FAIL="1" if mode == "query_failure" else "")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("setup.sh", result.stderr)
                self.assertEqual(self.calls("acprof"), [])
                self.assertTrue(all(call["args"] == ["tool", "dir", "--bin"] for call in self.calls("uv")))
                self.assertFalse((self.checkout / ".venv").exists())

    def test_noninteractive_install_uses_checkout_and_diagnoses_without_running_experiment(self):
        result = self.invoke("--no-modify-path")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        install = next(call for call in self.calls("uv") if call["args"][:2] == ["tool", "install"])
        self.assertIn(str(self.checkout), install["args"])
        self.assertIn(str(self.checkout / "requirements.lock"), install["args"])
        calls = self.calls("acprof")
        self.assertEqual([call["args"] for call in calls], [["doctor", "--profiling-mode", "basic", "--gpus", "off"]])
        self.assertEqual(calls[0]["cwd"], str(self.checkout))
        self.assertIn("tui", result.stdout)
        self.assertFalse(any(call["args"] == ["tool", "update-shell"] for call in self.calls("uv")))

    def test_interactive_install_launches_smoke_with_a_fresh_output_directory(self):
        for _ in range(2):
            result = self.invoke(terminal=True)
            self.assertEqual(result.returncode, 0, result.stdout)
        launches = [call["args"] for call in self.calls("acprof") if call["args"][0] == "tui"]
        self.assertEqual(len(launches), 2)
        outputs = []
        for launch in launches:
            self.assertEqual(launch[launch.index("--preset") + 1], "smoke")
            self.assertEqual(launch[launch.index("--model") + 1], "google-bert/bert-base-uncased")
            outputs.append(launch[launch.index("--output-dir") + 1])
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(self.config.read_text(), "KEEP_EXISTING_CONFIG=yes\n")

    def test_no_tui_option_works_in_a_terminal(self):
        result = self.invoke("--no-tui", terminal=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([call["args"][0] for call in self.calls("acprof")], ["doctor"])

    def test_doctor_failure_prevents_tui_and_can_be_retried(self):
        result = self.invoke(terminal=True, SETUP_TEST_DOCTOR_EXIT="1")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertFalse(any(call["args"][0] == "tui" for call in self.calls("acprof")))
        retried = self.invoke("--no-tui")
        self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
        self.assertEqual(self.config.read_text(), "KEEP_EXISTING_CONFIG=yes\n")

    def test_missing_docker_or_unsupported_os_fails_before_installation(self):
        for overrides in ({"SETUP_TEST_DOCKER_EXIT": "1"}, {"SETUP_TEST_OS": "Darwin"}):
            with self.subTest(overrides=overrides):
                result = self.invoke(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Docker" if "SETUP_TEST_DOCKER_EXIT" in overrides else "Linux", result.stderr)
                self.assertEqual(self.calls("uv"), [])

    def test_install_failure_does_not_start_doctor_or_tui(self):
        result = self.invoke(SETUP_TEST_INSTALL_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls("acprof"), [])

    def test_missing_uv_is_bootstrapped_without_shell_modifications(self):
        (self.bin / "uv").unlink()
        result = self.invoke("--no-tui", "--no-modify-path")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((Path(self.environment["UV_INSTALL_DIR"]) / "uv").is_file())
        self.assertEqual(len(self.calls("curl")), 1)
        self.assertEqual([call["args"][0] for call in self.calls("acprof")], ["doctor"])

    def test_failed_uv_download_does_not_execute_partial_installer(self):
        (self.bin / "uv").unlink()
        result = self.invoke(SETUP_TEST_DOWNLOAD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls("uv"), [])
        self.assertFalse((Path(self.environment["UV_INSTALL_DIR"]) / "uv").exists())

    def test_help_and_invalid_options_do_not_install_anything(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-tui", result.stdout)
        invalid = self.invoke("--unknown")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(self.calls("uv"), [])
        self.assertEqual(self.calls("docker"), [])
