"""Ship a self-contained Docker context without workspace files or credentials."""
from pathlib import Path
import shutil
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        self.bundle = tempfile.TemporaryDirectory(prefix="acprof-wheel-resources-")
        destination = Path(self.bundle.name)
        # Docker needs real Python source even when the host runs a frozen binary.
        suffixes = {".py", ".json", ".tcss", ".wav", ".md", ".txt", ".in", ".Dockerfile"}
        for directory in ("acprof", "dockerfiles", "assets", "examples"):
            for path in sorted((root / directory).rglob("*")):
                if (not path.is_file() or path.name == "AGENTS.md"
                        or "__pycache__" in path.parts or "_bundle" in path.parts
                        or path.suffix not in suffixes):
                    continue
                relative = path.relative_to(root).as_posix()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        shutil.copyfile(root / ".dockerignore", destination / ".dockerignore")
        build_data["force_include"][str(destination)] = "acprof/_bundle"

    def finalize(self, version, build_data, artifact_path):
        self.bundle.cleanup()
