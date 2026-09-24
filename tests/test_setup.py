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
        print(os.environ["UV_TOOL_BIN_DIR"])
    elif args != ["tool", "update-shell"]:
        sys.exit(98)
elif name == "acprof":
    if args[0] == "doctor":
        print("doctor prerequisite result")
        sys.exit(int(os.environ.get("SETUP_TEST_DOCTOR_EXIT", "0")))
    if args[0] != "tui":
        sys.exit(97)
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
