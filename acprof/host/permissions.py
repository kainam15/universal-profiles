"""Explicit, administrator-approved setup; never called by measurement code."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess

from acprof.artifacts import atomic_write_json


@dataclass(frozen=True)
class PermissionTarget:
    name: str
    path: str
    capability: str
    group: str
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class PermissionPlan:
    uid: int
    targets: tuple[PermissionTarget, ...]
    commands: tuple[tuple[str, ...], ...]
    sudo: str


def _trusted_file(path: Path, *, elf: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError(f"Refusing non-system or writable executable: {resolved}")
    for parent in resolved.parents:
        info = parent.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(f"Refusing executable under writable directory: {resolved}")
    if elf:
        with resolved.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError(f"Expected a system ELF executable: {resolved}")
    return resolved


def system_executable(name: str) -> Path:
    located = shutil.which(name)
    if not located:
        raise ValueError(f"Missing system tool: {name}")
    path = _trusted_file(Path(located))
    with path.open('rb') as stream:
        is_elf = stream.read(4) == b'\x7fELF'
    if name == 'perf' and not is_elf and path == Path('/usr/bin/perf').resolve():
        path = Path('/usr/lib/linux-tools') / platform.release() / 'perf'
    return _trusted_file(path, elf=True)


def build_permission_plan(names=('perf', 'tcpdump')) -> PermissionPlan:
    """Restrict executables to a profiling group and the current user's ACL.

    A named ACL makes the grant effective immediately in the existing TUI
    session; group membership changes and a new login are unnecessary.
    """
    if platform.system() != 'Linux' or os.getuid() == 0:
        raise ValueError('Open the TUI as an ordinary user on native Linux')
    names = tuple(dict.fromkeys(names))
    if not names or set(names) - {'perf', 'tcpdump'}:
        raise ValueError('Select perf and/or tcpdump')
    helpers = {name: str(system_executable(name)) for name in
               ('sudo', 'groupadd', 'chown', 'chmod', 'setfacl', 'setcap', 'getfacl', 'getcap')}
    targets, commands = [], []
    for name in names:
        path = system_executable(name)
        info = path.stat()
        cap, group = ('cap_perfmon', 'acprof-perf') if name == 'perf' else ('cap_net_raw', 'acprof-capture')
        targets.append(PermissionTarget(name, str(path), cap, group,
                                        (info.st_dev, info.st_ino, info.st_mtime_ns)))
        commands.extend((
            (helpers['groupadd'], '-f', group),
            (helpers['chown'], f'root:{group}', str(path)),
            (helpers['setfacl'], '-m', f'u:{os.getuid()}:r-x', str(path)),
            (helpers['chmod'], '0750', str(path)),
            (helpers['setcap'], f'{cap}=ep', str(path)),
        ))
    return PermissionPlan(os.getuid(), tuple(targets), tuple(commands), helpers['sudo'])


def execute_permission_plan(plan: PermissionPlan, *, backup_path: Path) -> None:
    """Run only the reviewed fixed commands, with sudo reading its own terminal.

    The caller suspends the TUI for this explicit action. No password is accepted
    as an argument, environment variable, subprocess pipe or stored preference.
    Failures stop the plan and preserve the original ACL/capability snapshot.
    """
    if build_permission_plan(tuple(target.name for target in plan.targets)) != plan:
        raise ValueError('System executables changed; review a fresh permission plan')
    entries = []
    for target in plan.targets:
        info = Path(target.path).stat()
        entry = {'path': target.path, 'uid': info.st_uid, 'gid': info.st_gid,
                 'mode': oct(stat.S_IMODE(info.st_mode))}
        for tool, args in (('getfacl', ['-p', '-n']), ('getcap', [])):
            result = subprocess.run([str(system_executable(tool)), *args, target.path],
                                    capture_output=True, text=True, timeout=5, check=True)
            entry[tool] = result.stdout
        entries.append(entry)
    atomic_write_json(backup_path, {'uid': plan.uid, 'targets': entries})
    for index, command in enumerate(plan.commands, start=1):
        result = subprocess.run([plan.sudo, '--', *command], timeout=120, check=False)
        if result.returncode:
            raise RuntimeError(
                f'Permission setup stopped at step {index}/{len(plan.commands)} '
                f'(exit={result.returncode}). Original permissions: {backup_path}'
            )
