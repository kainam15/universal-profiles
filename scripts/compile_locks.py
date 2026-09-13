"""用 uv 解析主机和 Linux x86_64 / Python 3.10 容器锁；安装仍由 pip 完成。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("nlp", "cv", "audio", "diffusion", "structured", "timeseries",
            "multimodal-transformers4576")
TORCH = {"cu128": ("2.11.0", "0.26.0"), "cu124": ("2.6.0", "0.21.0"), "cpu": ("2.11.0", "0.26.0")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv", help="uv 可执行文件（验证版本 0.12.13）")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--variant", action="append", choices=TORCH)
    args = parser.parse_args()
    if args.host_only and args.runtime_only:
        parser.error("--host-only 与 --runtime-only 互斥")

    def compile_file(source, target, *options):
        subprocess.run([
            args.uv, "pip", "compile", source, "--python-version", "3.10",
            "--no-annotate", "--no-header", "--output-file", target, *options,
        ], cwd=ROOT, check=True)

    if not args.runtime_only:
        compile_file("requirements-host.in", "requirements.lock", "--universal", "--generate-hashes")
    if args.host_only:
        return
    for variant in args.variant or TORCH:
        torch_version, vision_version = TORCH[variant]
        with tempfile.TemporaryDirectory() as directory:
            pins = Path(directory) / "torch.txt"
            pins.write_text(f"torch=={torch_version}+{variant}\n"
                            f"torchaudio=={torch_version}+{variant}\n"
                            f"torchvision=={vision_version}+{variant}\n")
            options = ["--python-platform", "x86_64-manylinux_2_28", "--torch-backend", variant]
            common = f"dockerfiles/locks/common-{variant}.txt"
            compile_file("dockerfiles/requirements/common.in", common, *options, "--constraint", str(pins))
            families = (*FAMILIES, "moss-transformers560") if variant == "cu128" else FAMILIES
            for family in families:
                # 保留已发布的 multimodal / MOSS 文件名，便于历史引用。
                name = family if variant == "cu128" and family not in FAMILIES[:6] else f"{family}-{variant}"
                compile_file(f"dockerfiles/requirements/{family}.in", f"dockerfiles/locks/{name}.txt",
                             *options, "--constraint", common)


if __name__ == "__main__":
    main()
