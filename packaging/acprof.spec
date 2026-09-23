# Build from an installed wheel so the frozen app carries exactly the wheel resources.
from importlib.metadata import distribution
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

package = Path(distribution("acprof").locate_file("acprof")).resolve()
assert (package / "_bundle").is_dir(), "Install the AC-Prof wheel before freezing"
modules = []
for path in package.rglob("*.py"):
    relative = path.relative_to(package)
    if relative.parts[0] in {"_bundle", "container"}:
        continue
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    modules.append(".".join(["acprof", *parts]))

datas = [(str(package / "_bundle"), "acprof/_bundle"),
         (str(package / "extensions"), "acprof/extensions"),
         (str(package / "tui" / "tui.tcss"), "acprof/tui")]
datas += collect_data_files("textual") + copy_metadata("acprof", recursive=True)
a = Analysis(
    [str(Path(SPECPATH) / "standalone.py")],
    pathex=[str(package.parent)],
    datas=datas,
    hiddenimports=modules + collect_submodules("textual"),
    excludes=["torch", "transformers", "tensorflow", "onnxruntime", "pytest", "IPython", "tkinter"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="acprof", console=True, strip=False, upx=False)
