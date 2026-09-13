"""Post-hoc locking, backups, and transactional file publication."""
from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from acprof.host.collection_history import COLLECTION_HISTORY_NAME, normalize_collection_history
from acprof.host.posthoc.context import (
    BACKUP_DIRNAME,
    LOCK_FILENAME,
    PROJECT_DIR,
    PosthocError,
    RESULT_CSV_NAME,
    ResultContext,
    STATIC_META_NAME,
    _load_json_object,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _timestamp_token() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def create_backup(context: ResultContext) -> Path:
    root = context.result_dir / BACKUP_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    base = _timestamp_token()
    backup = root / base
    suffix = 1
    while backup.exists():
        backup = root / f"{base}-{suffix}"
        suffix += 1
    backup.mkdir()
    shutil.copy2(context.result_csv, backup / RESULT_CSV_NAME)
    shutil.copy2(context.static_meta_path, backup / STATIC_META_NAME)
    if context.collection_history_existed:
        shutil.copy2(
            context.collection_history_path,
            backup / COLLECTION_HISTORY_NAME,
        )
    return backup


def _write_csv_temporary(
    destination: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    encoding: str,
) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(fieldnames),
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if destination.exists()
            else 0o644
        )
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_temporary(destination: Path, payload: Mapping[str, Any]) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if destination.exists()
            else 0o644
        )
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _restore_from_backup(destination: Path, backup_file: Path) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.restore.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(backup_file, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def commit_result_files(
    context: ResultContext,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    static_meta: Mapping[str, Any],
    collection_history: Mapping[str, Any],
    backup_dir: Path,
) -> None:
    csv_temporary = _write_csv_temporary(
        context.result_csv,
        fieldnames=fieldnames,
        rows=rows,
        encoding=context.csv_encoding,
    )
    meta_temporary = _write_json_temporary(context.static_meta_path, static_meta)
    history_temporary = _write_json_temporary(
        context.collection_history_path,
        collection_history,
    )
    csv_replaced = False
    meta_replaced = False
    history_replaced = False
    try:
        # Validate all complete temporary documents before publishing any of them.
        with csv_temporary.open(
            "r", encoding=context.csv_encoding, newline=""
        ) as f:
            if sum(1 for _row in csv.DictReader(f)) != len(rows):
                raise PosthocError("temporary result CSV row-count validation failed")
        _load_json_object(meta_temporary, "temporary static metadata")
        temporary_history = _load_json_object(
            history_temporary,
            "temporary collection history",
        )
        normalize_collection_history(temporary_history)

        os.replace(csv_temporary, context.result_csv)
        csv_replaced = True
        os.replace(meta_temporary, context.static_meta_path)
        meta_replaced = True
        os.replace(history_temporary, context.collection_history_path)
        history_replaced = True
    except Exception:
        if csv_replaced:
            _restore_from_backup(
                context.result_csv, backup_dir / RESULT_CSV_NAME
            )
        if meta_replaced:
            _restore_from_backup(
                context.static_meta_path, backup_dir / STATIC_META_NAME
            )
        if history_replaced:
            history_backup = backup_dir / COLLECTION_HISTORY_NAME
            if history_backup.is_file():
                _restore_from_backup(
                    context.collection_history_path,
                    history_backup,
                )
            else:
                try:
                    context.collection_history_path.unlink()
                except FileNotFoundError:
                    pass
        raise
    finally:
        for temporary in (csv_temporary, meta_temporary, history_temporary):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_process_cmdline(pid_dir: Path) -> List[str]:
    try:
        data = (pid_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return []
    return [
        part.decode("utf-8", errors="replace")
        for part in data.split(b"\0")
        if part
    ]


def _option_value(args: Sequence[str], name: str) -> Optional[str]:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def find_active_processes(
    result_dir: Path,
    *,
    model_id: str = "",
) -> List[Tuple[int, str]]:
    expected = result_dir.resolve()
    matches: List[Tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        args = _read_process_cmdline(pid_dir)
        if not args:
            continue
        command = " ".join(args)
        executable = Path(args[0]).name.lower()
        profiler_process = executable not in {"bash", "sh", "dash", "zsh"} and any(
            marker in command
            for marker in (
                "run.py",
                "acprof.cli.run",
                "compute_profile_runner",
                " ncu ",
                "/ncu ",
                "nsys",
                "massif",
                "posthoc.py",
                "profile.py",
                "acprof.cli.posthoc",
            )
        )
        direct_match = str(expected) in command and profiler_process
        run_match = False
        if model_id and _option_value(args, "--model") == model_id:
            is_run_command = any(
                Path(arg).name == "run.py" for arg in args
            ) or (
                "-m" in args and "acprof.cli.run" in args
            )
            if is_run_command:
                output_root = _option_value(args, "--output-dir") or "results"
                output_root_path = Path(output_root)
                if not output_root_path.is_absolute():
                    output_root_path = PROJECT_DIR / output_root_path
                candidate = output_root_path / model_id.replace("/", "--")
                run_match = candidate.resolve() == expected
        if direct_match or run_match:
            matches.append((int(pid_dir.name), command[:500]))
    return sorted(matches)


class PosthocLock:
    def __init__(self, result_dir: Path):
        from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
        self._result_lock = ResultDirectoryLock(result_dir)
        self._measurement_lock = MeasurementLock()
        self.path = result_dir / LOCK_FILENAME
        self._owned = False

    def __enter__(self) -> "PosthocLock":
        try:
            self._measurement_lock.__enter__()
            self._result_lock.__enter__()
            return self._acquire_legacy_lock()
        except BaseException:
            self._result_lock.__exit__(None, None, None)
            self._measurement_lock.__exit__(None, None, None)
            raise

    def _acquire_legacy_lock(self) -> "PosthocLock":
        payload = json.dumps({"pid": os.getpid(), "created_at": _timestamp_token()})
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(existing.get("pid", -1))
                except (OSError, ValueError, TypeError):
                    pid = -1
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    raise PosthocError(
                        f"another profile.py process is active (pid={pid}): {self.path}"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._owned = True
            return self
        raise PosthocError(f"cannot acquire profile lock: {self.path}")

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            if self._owned:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                self._owned = False
        finally:
            self._result_lock.__exit__(_exc_type, _exc, _traceback)
            self._measurement_lock.__exit__(_exc_type, _exc, _traceback)
