"""主实验状态、目录互斥与 case 恢复；所有操作均在测量窗口之外。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from uuid import uuid4

from acprof.artifacts import atomic_write_json
from acprof.result_csv import expected_measurements, read_result_csv


RUN_STATE_NAME = "run_state.json"
RESULT_LOCK_NAME = ".acprof-result.lock"


class RunStateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_identity(project_dir: str | Path) -> dict:
    digest = hashlib.sha256()
    root = Path(project_dir) / "acprof"
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = sorted((dist.metadata.get("Name", ""), dist.version)
                      for dist in importlib.metadata.distributions())
    machine_id = Path("/etc/machine-id")
    return {
        "machine": platform.machine(), "kernel": platform.release(), "hostname": platform.node(),
        "machine_id_sha256": file_sha256(machine_id) if machine_id.is_file() else None,
        "python": platform.python_version(), "source_sha256": digest.hexdigest(),
        "packages_sha256": hashlib.sha256(json.dumps(packages).encode()).hexdigest(),
    }


def run_options(args) -> dict:
    ignored = {"resume", "skip_build", "output_dir", "notify"}
    options = {name: value for name, value in vars(args).items() if name not in ignored}
    for name in ("cpus", "mems", "gpus"):
        options[name] = ",".join(part.strip().lower() for part in options[name].split(",") if part.strip())
    inherited = ("AUTO_WARMUP_REQUESTS", "SLOW_LATENCY_THRESHOLD_S", "IDLE_DEBUG_TRACE_INTERVAL_S",
                 "DEVICE_INDEX", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    options["measurement_environment"] = {name: os.environ.get(name) for name in inherited}
    if options.get("workload_spec"):
        path = Path(options["workload_spec"]).expanduser().resolve()
        options["workload_spec"] = str(path)
        options["workload_spec_sha256"] = file_sha256(path)
    return options


def load_run_state(directory: str | Path) -> dict:
    path = Path(directory) / RUN_STATE_NAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RunStateError(f"无法读取恢复状态 {path}；历史实验请使用新输出目录") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RunStateError(f"不支持的主实验状态格式：{path}")
    return payload


class ResultDirectoryLock:
    """Linux advisory lock: process exit releases ownership without stale PID recovery."""
    def __init__(self, directory: str | Path):
        self.path = Path(directory) / RESULT_LOCK_NAME
        self.stream = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RunStateError(f"另一个采集或补采进程正在使用结果目录：{self.path.parent}") from exc
        return self

    def __exit__(self, *_args):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class MeasurementLock(ResultDirectoryLock):
    """同机同用户的实验串行化，防止跨结果目录共享端口及采样资源。"""
    def __init__(self):
        self.path = Path(tempfile.gettempdir()) / f"acprof-measurement-{os.getuid()}.lock"
        self.stream = None

    def __enter__(self):
        try:
            return super().__enter__()
        except RunStateError as error:
            raise RunStateError("本机已有同一用户的 AC-Prof 采集或补采在运行，请待其结束后重试") from error


class RunState:
    def __init__(self, directory: str | Path, options: dict, *, resume: bool, project_dir: str):
        self.directory = Path(directory).resolve()
        self.path = self.directory / RUN_STATE_NAME
        self.lock = ResultDirectoryLock(self.directory)
        self.measurement_lock = MeasurementLock()
        self.measurement_lock.__enter__()
        try:
            self.lock.__enter__()
            if resume:
                self.data = load_run_state(self.directory)
                if self.data.get("options") != options:
                    raise RunStateError("恢复参数与原实验不一致；请使用原命令加 --resume，或选择新输出目录")
                if self.data.get("host") != host_identity(project_dir):
                    raise RunStateError("恢复时主机、Python 依赖或 AC-Prof 源码已变化；请使用新输出目录")
                self.verify_artifacts()
            else:
                occupied = [p for p in self.directory.iterdir()
                            if p.name not in {RESULT_LOCK_NAME, "probes"}]
                if occupied:
                    raise RunStateError(f"结果目录已有实验产物：{self.directory}；续跑请加 --resume，新实验请更换 --output-dir")
                self.data = {"schema_version": 1, "run_id": uuid4().hex, "created_at": utc_now(),
                             "status": "preparing", "options": options, "host": host_identity(project_dir),
                             "cases": {}, "artifacts": {}, "attempts": []}
            self.data["attempts"].append({"started_at": utc_now(), "pid": os.getpid(), "resume": resume})
            self.save()
        except BaseException:
            self.lock.__exit__(None, None, None)
            self.measurement_lock.__exit__(None, None, None)
            raise

    @property
    def ready(self) -> bool:
        return "runtime" in self.data

    @property
    def complete(self) -> bool:
        return self.data.get("status") == "complete"

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def artifact_path(self, relative: str) -> Path:
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise RunStateError(f"恢复产物路径越出结果目录：{relative}")
        return path

    def verify_artifacts(self) -> None:
        # A finalized result may subsequently be changed by documented post-hoc tools.
        if self.complete:
            read_result_csv(self.directory / "result_all.csv", expected=self.expected())
            return
        for name, digest in self.data.get("artifacts", {}).items():
            path = self.artifact_path(name)
            if not path.is_file() or file_sha256(path) != digest:
                raise RunStateError(f"恢复产物缺失或已改变：{path}；请恢复原文件或使用新输出目录")

    def bind_runtime(self, task, image, planned, compute_plan: str, execution_plan: str) -> None:
        paths = ["static_meta.json", "collection_history.json", "input_scale_plan.json"]
        for value in (compute_plan, execution_plan):
            if value:
                paths.append(str(Path(value).resolve().relative_to(self.directory)))
        for name in paths:
            path = self.artifact_path(name)
            if path.is_file():
                self.data["artifacts"][name] = file_sha256(path)
        plan = asdict(planned)
        if plan["plan_file"]:
            plan["plan_file"] = str(Path(plan["plan_file"]).resolve().relative_to(self.directory))
        self.data["runtime"] = {
            "task": asdict(task), "image": asdict(image), "planned": plan,
            "compute_plan": str(Path(compute_plan).resolve().relative_to(self.directory)) if compute_plan else "",
            "execution_plan": str(Path(execution_plan).resolve().relative_to(self.directory)) if execution_plan else "",
        }
        self.data["status"] = "running"
        self.save()

    def restore_runtime(self):
        from acprof.host.detect import TaskInfo
        from acprof.host.docker_runtime import ImageInfo, require_image_identity
        from acprof.host.input_plan import PlannedInputScales
        snapshot = self.data["runtime"]
        image = ImageInfo(**snapshot["image"])
        if not image.tag.startswith("sha256:"):
            raise RunStateError("恢复要求原实验的不可变 image ID；历史标签镜像请使用新输出目录")
        require_image_identity(image.tag, image.runtime_environment)
        plan = dict(snapshot["planned"])
        if plan.get("plan_file"):
            plan["plan_file"] = str(self.artifact_path(plan["plan_file"]))
        return (TaskInfo(**snapshot["task"]), image, PlannedInputScales(**plan),
                str(self.artifact_path(snapshot["compute_plan"])) if snapshot["compute_plan"] else "",
                str(self.artifact_path(snapshot["execution_plan"])) if snapshot["execution_plan"] else "")

    def expected(self, cpu=None, mem=None, gpu=None):
        options = self.data["options"]
        return expected_measurements(
            [cpu] if cpu is not None else [int(x) for x in options["cpus"].split(",")],
            [mem] if mem is not None else [int(x) for x in options["mems"].split(",")],
            [gpu] if gpu is not None else options["gpus"].split(","),
            self.data["runtime"]["planned"]["scales"], options["warmup"], options["repeat"],
        )

    def prepare_case(self, filename: str, cpu: int, mem: int, gpu: str) -> str | None:
        record = self.data["cases"].get(filename, {})
        path = self.artifact_path(filename)
        if record.get("status") == "complete":
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                raise RunStateError(f"已完成 case 的 CSV 缺失或改变：{path}")
            read_result_csv(path, expected=self.expected(cpu, mem, gpu))
            return str(path)
        case_name = filename.removeprefix("result_").removesuffix(".csv")
        candidates = [path, Path(str(path) + ".sniff_groups.jsonl"), Path(str(path) + ".client_error.json"),
                      self.directory / f"sniff_{case_name}.pcap", self.directory / f"lat_{case_name}.json",
                      self.directory / "debug_idle_diag" / f"{filename}.idle_diag.jsonl"]
        existing = [candidate for candidate in candidates if candidate.exists()]
        if existing:
            backup = self.directory / "interrupted_cases" / case_name / uuid4().hex
            backup.mkdir(parents=True)
            # Preserve every source before removing any file, including partial PCAPs.
            for source in existing:
                if source.is_symlink() or not source.is_file():
                    raise RunStateError(f"不支持的 case 产物类型：{source}")
                target = backup / source.name
                shutil.copy2(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
            self.data["cases"][filename] = {"status": "archived", "backup": str(backup.relative_to(self.directory))}
            self.save()
            for source in existing:
                source.unlink()
        self.data["cases"][filename] = {"status": "running", "started_at": utc_now()}
        self.save()
        return None

    def finish_case(self, path: str, cpu: int, mem: int, gpu: str) -> None:
        _, rows = read_result_csv(path, expected=self.expected(cpu, mem, gpu))
        self.data["cases"][Path(path).name] = {
            "status": "complete", "completed_at": utc_now(), "sha256": file_sha256(path),
            "row_count": len(rows), "error_rows": sum(row.get("status") == "error" for row in rows),
        }
        self.save()

    def finish(self, final_csv: str) -> None:
        _, rows = read_result_csv(final_csv, expected=self.expected())
        self.data.update(status="complete", completed_at=utc_now(), result_sha256=file_sha256(final_csv),
                         row_count=len(rows), outcome="partial" if any(row.get("status") == "error" for row in rows) else "ok")
        self.save()

    def close(self, outcome: str = "interrupted") -> None:
        try:
            if not self.complete:
                self.data["status"] = outcome
            self.data["attempts"][-1]["ended_at"] = utc_now()
            self.data["attempts"][-1]["outcome"] = self.data["status"]
            self.save()
        finally:
            self.lock.__exit__(None, None, None)
            self.measurement_lock.__exit__(None, None, None)
