"""In-memory completion summaries for isolated profiler stages.

Collectors report once after a tool's probes have returned and released their
containers. Callbacks run synchronously before the next measurement starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Callable, Iterable, Mapping, Optional


@dataclass(frozen=True)
class ProfilerProgress:
    profiler: str
    status: str
    elapsed_seconds: float
    total_samples: int
    error_samples: int
    detail: str = ""


ProfilerProgressCallback = Callable[[ProfilerProgress], None]


def report_profiler_completion(
    progress_callback: Optional[ProfilerProgressCallback],
    *,
    profiler: str,
    profiles: Iterable[Mapping[str, Any]],
    elapsed_seconds: float,
) -> None:
    """Summarize source samples, excluding copies expanded over the matrix.

An entry represents one scale at a sampled resource configuration, regardless
of its repeat count. A stage ending does not imply its samples succeeded.
Notification failures must never change collection or its persisted results.
"""
    if progress_callback is None:
        return

    try:
        total_samples = 0
        error_samples = 0
        errors: list[str] = []
        for profile in profiles:
            entries = list(profile.get("entries") or [])
            parent_error = str(profile.get("error") or "").strip()
            entry_errors = [str(entry.get("error") or "").strip() for entry in entries]
            failed = sum(bool(error) for error in entry_errors)
            # Some tools return only a stage-level diagnostic. Do not call
            # their entries successful just because entry errors are absent.
            if parent_error and not failed:
                failed = len(entries)
            total_samples += len(entries)
            error_samples += failed
            for error in (entry_errors if any(entry_errors) else [parent_error]):
                if error and error not in errors and len(errors) < 3:
                    errors.append(error)

        if error_samples or errors:
            status = "partial" if total_samples > error_samples else "failed"
        else:
            status = "success" if total_samples else "no_results"
        progress_callback(
            ProfilerProgress(
                profiler=profiler,
                status=status,
                elapsed_seconds=max(0.0, elapsed_seconds),
                total_samples=total_samples,
                error_samples=error_samples,
                detail="; ".join(errors),
            )
        )
    except Exception as exc:
        # Callback exceptions can contain credentials; only expose the type.
        print(
            "[profile][WARN] Profiler completion callback failed: "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
