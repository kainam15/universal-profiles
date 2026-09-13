"""锁检查保持只读，并拒绝与目标解释器不兼容的制品。"""
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime_fixture import copy_dependency_tree
from scripts.compile_locks import check_catalog
from scripts.compile_system_lock import resolve_in_container, snapshot_url


class LockCompilerTests(unittest.TestCase):
    def test_check_is_read_only_and_does_not_use_docker_or_resolver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob("*") if path.is_file()}
            with patch("subprocess.run", side_effect=AssertionError("check spawned process")), patch(
                "urllib.request.urlopen", side_effect=AssertionError("check accessed network"),
            ):
                result = check_catalog(root)
            self.assertEqual(result["profiles"], 22)
            self.assertEqual(len(result["environments"]), 20)
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in root.rglob("*") if path.is_file()})

    def test_check_rejects_wheel_for_wrong_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            path = root / "dockerfiles/locks/nlp-cpu.txt"
            text = path.read_text()
            self.assertIn("cp310-cp310", text)
            path.write_text("\n".join(line.replace("cp310-cp310", "cp311-cp311")
                                      if line.startswith("numpy @") else line for line in text.splitlines()) + "\n")
            with self.assertRaisesRegex(ValueError, "wheel.*平台|平台.*wheel"):
                check_catalog(root)

    def test_snapshot_uses_actual_timestamp_not_requested_cutoff(self):
        listing = b"20260912T203535Z 20260913T031122Z 20260901T000000Z"
        with patch("urllib.request.urlopen", return_value=BytesIO(listing)) as opened:
            url = snapshot_url("debian", "20260913T000000Z")
        self.assertEqual(url, "https://snapshot.debian.org/archive/debian/20260912T203535Z/")
        self.assertIn("?year=2026&month=9", opened.call_args.args[0])

    def test_system_resolver_cannot_mutate_host(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "容器"):
                resolve_in_container("20260913T000000Z", Path("unused"))
        run.assert_not_called()
