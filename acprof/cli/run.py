#!/usr/bin/env python3
"""AC-Prof Universal Profiler - One-click HuggingFace model profiling.

Usage:
    python run.py --model bert-base-uncased
    python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
    python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

from acprof.config import (
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    SCALING_DIMENSIONS,
)
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
    empty_collection_history,
    write_collection_history_json,
)
from acprof.host.orchestrator import (
    EnergyProfilingError,
    MIPSProfilingError,
    PacketLatencyError,
    require_packet_latency_prerequisites,
)
from acprof.monitors.perf_mips import require_mips_prerequisites

PROJECT_DIR = str(Path(__file__).resolve().parents[2])
NATIVE_DOCKER_SOCKET = "/var/run/docker.sock"
TMUX_TERMINAL_LOG_FILENAME = "tmux_all.log"
_ACTIVE_TMUX_TERMINAL_LOG: tuple[str, str, str] | None = None


def _parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_run_command(argv: list[str]) -> str:
    """Return a shell-safe command string matching the run.py invocation."""
    if not argv:
        return "python run.py"
    return shlex.join(["python", *argv])


def _start_tmux_terminal_log(
    output_dir: str,
    argv: list[str],
) -> tuple[str, str, str] | None:
    """Pipe all future output from the current tmux pane to a temporary log."""
    pane_id = os.environ.get("TMUX_PANE", "").strip()
    if not os.environ.get("TMUX") or not pane_id:
        return None

    try:
        pipe_status = subprocess.run(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                pane_id,
                "#{pane_pipe}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: {exc}")
        return None

    if pipe_status.returncode != 0:
        detail = (pipe_status.stderr or pipe_status.stdout or "").strip()
        print(
            f"[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: "
            f"{detail or f'exit {pipe_status.returncode}'}"
        )
        return None
    if pipe_status.stdout.strip().lower() in {"1", "on", "true", "yes"}:
        print(
            f"[terminal-log][WARN] tmux pane {pane_id} already has an active "
            "pipe; leaving it unchanged and skipping automatic tmux_all.log"
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, TMUX_TERMINAL_LOG_FILENAME)
    partial_path = f"{log_path}.part"
    try:
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write(f"$ {_format_run_command(argv)}\n")
    except OSError as exc:
        print(f"[terminal-log][WARN] Cannot initialize {partial_path}: {exc}")
        return None

    pipe_command = f"cat >> {shlex.quote(partial_path)}"
    try:
        pipe_result = subprocess.run(
            [
                "tmux",
                "pipe-pane",
                "-O",
                "-t",
                pane_id,
                pipe_command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[terminal-log][WARN] Cannot start tmux pane logging: {exc}")
        return None

    if pipe_result.returncode != 0:
        detail = (pipe_result.stderr or pipe_result.stdout or "").strip()
        print(
            "[terminal-log][WARN] Cannot start tmux pane logging: "
            f"{detail or f'exit {pipe_result.returncode}'}"
        )
        return None

    print(f"[terminal-log] Recording tmux pane {pane_id}: {log_path}")
    return pane_id, partial_path, log_path


def _stop_tmux_terminal_log(
    terminal_log: tuple[str, str, str],
) -> bool:
    """Stop the pane pipe and atomically publish the completed terminal log."""
    pane_id, partial_path, log_path = terminal_log
    try:
        close_result = subprocess.run(
            ["tmux", "pipe-pane", "-t", pane_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(
            f"[terminal-log][WARN] Cannot stop tmux pane logging; "
            f"partial log remains at {partial_path}: {exc}"
        )
        return False

    if close_result.returncode != 0:
        detail = (close_result.stderr or close_result.stdout or "").strip()
        print(
            f"[terminal-log][WARN] Cannot stop tmux pane logging; "
            f"partial log remains at {partial_path}: "
            f"{detail or f'exit {close_result.returncode}'}"
        )
        return False

    try:
        os.replace(partial_path, log_path)
    except OSError as exc:
        print(
            f"[terminal-log][WARN] Cannot finalize {log_path}; "
            f"partial log remains at {partial_path}: {exc}"
        )
        return False

    print(f"[terminal-log] Saved terminal display: {log_path}")
    return True


def _docker_info_is_docker_desktop(info: str) -> bool:
    normalized = (info or "").lower()
    return "docker desktop" in normalized or "name=docker-desktop" in normalized


def _docker_context_is_docker_desktop(context_name: str) -> bool:
    normalized = (context_name or "").strip().lower()
    return normalized in {"desktop-linux", "docker-desktop"} or normalized.startswith("desktop-")


def _process_is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _exit_unsupported_host(reason: str) -> None:
    print(
        "[infra][ERROR] AC-Prof requires a native Linux host; "
        f"{reason}.\n\n"
        "WSL and Docker Desktop do not reliably expose all host-side data "
        "sources used by this project, including RAPL, perf PMU events, the "
        "Docker bridge, cgroups, and the NVIDIA runtime.\n\n"
        "Run AC-Prof directly from native Ubuntu as a normal user:\n"
        f"  cd {PROJECT_DIR}\n"
        "  source .venv/bin/activate\n"
        "  unset DOCKER_HOST DOCKER_CONTEXT\n"
        "  docker context use default\n"
        "  python run.py --model <model-id> ...\n",
        file=sys.stderr,
    )
    sys.exit(1)


def require_native_linux_host() -> None:
    """Exit before profiling when the process is not on native Linux."""
    try:
        system = platform.system()
    except Exception:
        system = ""

    if system != "Linux":
        _exit_unsupported_host(f"detected host OS {system or 'unknown'}")
    if _process_is_wsl():
        _exit_unsupported_host("WSL was detected")


def _docker_host_is_native_socket(docker_host: str) -> bool:
    normalized = (docker_host or "").strip()
    if not normalized:
        return False
    if normalized.startswith("unix://"):
        normalized = normalized[len("unix://"):]
    elif normalized.startswith("unix:"):
        normalized = normalized[len("unix:"):]
    else:
        return False
    return os.path.normpath(normalized) == NATIVE_DOCKER_SOCKET


def _exit_nonlocal_docker(docker_host: str) -> None:
    print(
        "[infra][ERROR] AC-Prof is not connected to the native Docker socket "
        f"{NATIVE_DOCKER_SOCKET}.\n\n"
        f"Detected Docker endpoint: {docker_host or 'unknown'}\n\n"
        "Packet capture, container cgroups, perf PID attachment, and host "
        "profilers must observe containers created by the local Ubuntu daemon.\n\n"
        "Switch back to the native daemon before running again:\n"
        "  unset DOCKER_HOST DOCKER_CONTEXT\n"
        "  docker context use default\n"
        f"  test \"$(docker context inspect default --format "
        f"'{{{{(index .Endpoints \"docker\").Host}}}}')\" = "
        f"\"unix://{NATIVE_DOCKER_SOCKET}\"\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _exit_docker_desktop() -> None:
    print(
        "[infra][ERROR] AC-Prof is currently connected to Docker Desktop, not "
        "the native Linux Docker daemon.\n\n"
        "Docker Desktop cannot reliably expose the host /opt profiler installs, "
        "docker0 traffic, or NVIDIA GPU runtime needed by this project.\n\n"
        "Switch to native Docker before running again, for example:\n"
        "  docker context use default\n"
        "or run one command with:\n"
        "  DOCKER_HOST=unix:///var/run/docker.sock python run.py ...\n",
        file=sys.stderr,
    )
    sys.exit(1)


def require_native_docker() -> None:
    """Require the local native-Linux Docker daemon used by host monitors."""
    try:
        context_result = subprocess.run(
            ["docker", "context", "show"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if (
            context_result.returncode == 0
            and _docker_context_is_docker_desktop(context_result.stdout)
        ):
            _exit_docker_desktop()

        docker_host = os.environ.get("DOCKER_HOST", "").strip()
        if not docker_host and context_result.returncode == 0:
            context_name = context_result.stdout.strip()
            endpoint_result = subprocess.run(
                [
                    "docker",
                    "context",
                    "inspect",
                    context_name,
                    "--format",
                    '{{(index .Endpoints "docker").Host}}',
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            if endpoint_result.returncode == 0:
                docker_host = endpoint_result.stdout.strip()

        if docker_host and not _docker_host_is_native_socket(docker_host):
            _exit_nonlocal_docker(docker_host)

        result = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "Name={{.Name}}\n"
                "OperatingSystem={{.OperatingSystem}}\n"
                "DockerRootDir={{.DockerRootDir}}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except FileNotFoundError:
        print(
            "[infra][ERROR] Docker CLI was not found. Install Docker and run AC-Prof "
            "against the native Linux Docker daemon.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(
            "[infra][ERROR] `docker info` timed out. Check that the native Linux "
            "Docker daemon is running before starting AC-Prof.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        print(
            "[infra][ERROR] Could not talk to Docker. Start the native Linux Docker "
            f"daemon and retry.\n\nDocker output:\n{detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    if _docker_info_is_docker_desktop(result.stdout):
        _exit_docker_desktop()


def require_cpu_energy_prerequisites() -> None:
    """Exit early when CPU/vCPU energy profiling cannot be collected."""
    try:
        from acprof.monitors import energy_cpu

        cpu_power_source = energy_cpu.detect_cpu_power_source()
        vcpu_power_method = energy_cpu.detect_vcpu_power_method()
    except Exception as exc:
        print(
            "[cpu-energy][ERROR] CPU/vCPU energy profiling is required, but "
            f"AC-Prof could not run the CPU energy detector: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if cpu_power_source == "rapl" and vcpu_power_method == "rapl_cgroup_cpu_share":
        return

    print(
        "[cpu-energy][ERROR] CPU/vCPU energy profiling is required, but AC-Prof "
        "cannot read the Linux RAPL powercap counters needed for CPU package "
        "energy and estimated vCPU energy.\n\n"
        f"Detected: cpu_power_source={cpu_power_source}, "
        f"vcpu_power_method={vcpu_power_method}\n\n"
        "Common cause on Linux: /sys/class/powercap/intel-rapl:*/energy_uj "
        "exists but is only readable by root.\n\n"
        "Check current permissions:\n"
        "  ls -l /sys/class/powercap/intel-rapl:*/energy_uj\n\n"
        "Temporary fix for the current boot:\n"
        "  sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj\n\n"
        "Persistent fix with systemd-tmpfiles:\n"
        "  echo 'z /sys/class/powercap/intel-rapl:*/energy_uj 0444 root root -' | "
        "sudo tee /etc/tmpfiles.d/acprof-rapl.conf\n"
        "  sudo systemd-tmpfiles --create /etc/tmpfiles.d/acprof-rapl.conf\n\n"
        "After fixing permissions, rerun AC-Prof as your normal user. Avoid "
        "`sudo python run.py ...` because it can leave result files owned by root.",
        file=sys.stderr,
    )
    sys.exit(1)


def _cleanup_intermediate_results(csv_paths: list[str], output_dir: str, final_csv: str) -> None:
    """Delete per-run intermediate artifacts after the merged CSV is safely written."""
    if not csv_paths:
        return

    if not os.path.exists(final_csv):
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is missing: {final_csv}")
        return

    if os.path.getsize(final_csv) <= 0:
        print(f"[cleanup][WARN] Skip cleanup because merged CSV is empty: {final_csv}")
        return

    targets = set()
    for csv_path in csv_paths:
        targets.add(csv_path)
        targets.add(f"{csv_path}.sniff_groups.jsonl")

        base_name = os.path.basename(csv_path)
        if not (base_name.startswith("result_case_") and base_name.endswith(".csv")):
            print(f"[cleanup][WARN] Skip derived cleanup for unexpected CSV name: {csv_path}")
            continue

        case_name = base_name[len("result_"):-len(".csv")]
        targets.add(os.path.join(output_dir, f"lat_{case_name}.json"))
        targets.add(os.path.join(output_dir, f"sniff_{case_name}.pcap"))

    removed = 0
    missing = 0
    failed = 0

    for path in sorted(targets):
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            os.remove(path)
            removed += 1
            print(f"[cleanup] Removed: {path}")
        except OSError as exc:
            failed += 1
            print(f"[cleanup][WARN] Failed to remove {path}: {exc}")

    print(f"[cleanup] Done. removed={removed}, missing={missing}, failed={failed}")


def _run_main():
    global _ACTIVE_TMUX_TERMINAL_LOG

    start_time = time.perf_counter()
    bootstrap_project_env(PROJECT_DIR)

    parser = argparse.ArgumentParser(
        description="AC-Prof: Universal HuggingFace Model Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --model bert-base-uncased
  python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
  python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
        """,
    )

    # Required
    parser.add_argument("--model", required=True, help="HuggingFace model ID")

    # Detection overrides
    parser.add_argument("--task", default=None, help="Override pipeline_tag (e.g., text-generation)")
    parser.add_argument("--task-family", default=None, help="Override task family (nlp/cv/audio/timeseries)")
    parser.add_argument("--backend", default=None, help="Override runtime backend (transformers_pipeline/chronos/...)")

    # Resource matrix
    parser.add_argument("--cpus", default="1,2,4,8", help="CPU core counts (comma-separated)")
    parser.add_argument("--mems", default="2,4,8,16", help="Memory caps in GB (comma-separated)")
    parser.add_argument("--gpus", default="off,on", help="GPU modes (comma-separated: off,on)")

    # Experiment parameters
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="Measurement repeat count")
    parser.add_argument(
        "--repeat-in-window",
        type=int,
        default=DEFAULT_REPEAT_IN_WINDOW,
        help="Requests per energy window; 0 enables auto calibration",
    )
    parser.add_argument(
        "--repeat-window-seconds",
        type=float,
        default=DEFAULT_REPEAT_WINDOW_SECONDS,
        help="Target workload window duration for auto repeat-in-window",
    )
    parser.add_argument("--sample-hz", type=float, default=20.0, help="GPU energy sampling rate")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=3.0,
        help="Idle baseline measurement duration before each workload window",
    )
    parser.add_argument(
        "--idle-cooldown-seconds",
        type=float,
        default=DEFAULT_IDLE_COOLDOWN_SECONDS,
        help="Cooldown duration before collecting idle baselines for each workload window",
    )
    parser.add_argument(
        "--idle-debug",
        action="store_true",
        help="Write CPU idle baseline timestamps and per-row diagnostic JSONL sidecars",
    )
    parser.add_argument("--input-scales", default=None, help="Override input scale values (comma-separated)")
    parser.add_argument(
        "--workload-spec",
        default=None,
        help=(
            "Path to an audio workload manifest. ASR defaults to the "
            "bundled LibriSpeech short-form manifest."
        ),
    )

    # Compute profiling
    # Kept as a hidden compatibility alias for existing scripts. New commands
    # should use --compute-profile-tool none, matching execution profiling.
    parser.add_argument(
        "--no-compute-profile",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--compute-profile-tool",
        choices=("none", "both", "auto", "torch", "ncu", "vendor"),
        default="both",
        help=(
            "Compute FLOP profiler: none skips all compute probes; both "
            "(default) independently collects torch_profiler_eager logical "
            "FLOP and ncu GPU executed FLOP; auto is a deprecated alias for both"
        ),
    )
    parser.add_argument("--advisor-root", default=None, help="Host Intel Advisor install root or advisor executable")
    parser.add_argument("--ncu-root", default=None, help="Host Nsight Compute install root or ncu executable")
    parser.add_argument("--advisor-repeat", type=int, default=20, help="Intel Advisor profiled inference repetitions")
    parser.add_argument(
        "--torch-profiler-repeat",
        type=int,
        default=1,
        help="torch_profiler_eager profiled inference repetitions",
    )
    parser.add_argument("--ncu-repeat", type=int, default=1, help="ncu profiled inference repetitions")
    parser.add_argument("--compute-profile-cpus", type=int, default=None, help="CPU cores for temporary compute profiler containers (default: host logical CPUs)")
    parser.add_argument("--compute-profile-mem", type=int, default=None, help="Memory GB for temporary compute profiler containers (default: 75%% of host memory)")
    profile_artifact_group = parser.add_mutually_exclusive_group()
    profile_artifact_group.add_argument(
        "--keep-compute-profiles",
        dest="keep_compute_profiles",
        action="store_true",
        help="Keep raw Advisor/ncu profiler artifacts (default)",
    )
    profile_artifact_group.add_argument(
        "--discard-compute-profiles",
        dest="keep_compute_profiles",
        action="store_false",
        help="Discard raw profiler artifacts after summaries are recorded",
    )
    parser.set_defaults(keep_compute_profiles=True)

    # High-overhead execution profiling. These probes are intentionally
    # opt-in and use reduced resource sampling by default.
    parser.add_argument(
        "--execution-profile-tool",
        choices=("none", "both", "massif", "nsys"),
        default="none",
        help=(
            "Optional execution profiler: Massif for CPU heap peaks, Nsight "
            "Systems for CUDA/GPU timelines, both, or none (default)"
        ),
    )
    parser.add_argument(
        "--massif-sampling",
        choices=("per-scale", "full"),
        default="per-scale",
        help=(
            "Massif resource sampling: per-scale profiles the largest selected "
            "CPU/memory case and reuses it across CPU-only rows (default); "
            "full profiles every selected CPU/memory case"
        ),
    )
    parser.add_argument(
        "--massif-reference-cpu",
        type=int,
        default=None,
        help="Representative CPU for Massif per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--massif-reference-mem",
        type=int,
        default=None,
        help="Representative memory GB for Massif per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--massif-repeat",
        type=int,
        default=1,
        help="Inference repetitions inside each Valgrind Massif probe",
    )
    parser.add_argument(
        "--nsys-sampling",
        choices=("per-cpu-scale", "per-scale", "full"),
        default="per-cpu-scale",
        help=(
            "Nsight Systems resource sampling: per-cpu-scale profiles every "
            "selected CPU at the largest selected memory (default); per-scale "
            "uses one representative CPU/memory case; full profiles every case"
        ),
    )
    parser.add_argument(
        "--nsys-reference-cpu",
        type=int,
        default=None,
        help="Representative CPU for Nsys per-scale sampling (default: largest selected)",
    )
    parser.add_argument(
        "--nsys-reference-mem",
        type=int,
        default=None,
        help=(
            "Representative memory GB for Nsys reduced sampling "
            "(default: largest selected)"
        ),
    )
    parser.add_argument(
        "--nsys-repeat",
        type=int,
        default=1,
        help="Inference repetitions inside each Nsight Systems capture range",
    )
    parser.add_argument(
        "--nsys-root",
        default=None,
        help="Host Nsight Systems install root or nsys executable",
    )
    execution_artifact_group = parser.add_mutually_exclusive_group()
    execution_artifact_group.add_argument(
        "--keep-execution-profiles",
        dest="keep_execution_profiles",
        action="store_true",
        help=(
            "Keep raw Massif .out and Nsight Systems .nsys-rep artifacts "
            "(default); derived Nsight SQLite caches are always discarded"
        ),
    )
    execution_artifact_group.add_argument(
        "--discard-execution-profiles",
        dest="keep_execution_profiles",
        action="store_false",
        help="Discard raw execution-profiler artifacts after summaries are recorded",
    )
    parser.set_defaults(keep_execution_profiles=True)

    # Infrastructure
    parser.add_argument("--sniff-iface", default="docker0", help="Network interface for tcpdump")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker image build (use existing)")

    args = parser.parse_args()
    run_command = _format_run_command(sys.argv)
    compute_profile_disabled = (
        args.no_compute_profile or args.compute_profile_tool == "none"
    )
    if args.repeat_in_window < 0:
        parser.error("--repeat-in-window must be >= 0")
    if args.repeat_window_seconds <= 0.0:
        parser.error("--repeat-window-seconds must be > 0")
    if args.torch_profiler_repeat <= 0:
        parser.error("--torch-profiler-repeat must be > 0")
    if args.ncu_repeat <= 0:
        parser.error("--ncu-repeat must be > 0")
    if args.massif_repeat <= 0:
        parser.error("--massif-repeat must be > 0")
    if args.nsys_repeat <= 0:
        parser.error("--nsys-repeat must be > 0")
    for option, value in (
        ("--massif-reference-cpu", args.massif_reference_cpu),
        ("--massif-reference-mem", args.massif_reference_mem),
        ("--nsys-reference-cpu", args.nsys_reference_cpu),
        ("--nsys-reference-mem", args.nsys_reference_mem),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be > 0")

    terminal_output_dir = os.path.join(
        PROJECT_DIR,
        args.output_dir,
        args.model.replace("/", "--"),
    )
    _ACTIVE_TMUX_TERMINAL_LOG = _start_tmux_terminal_log(
        terminal_output_dir,
        sys.argv,
    )

    require_native_linux_host()
    require_native_docker()

    try:
        require_packet_latency_prerequisites(
            project_dir=PROJECT_DIR,
            sniff_iface=args.sniff_iface,
        )
    except PacketLatencyError as exc:
        print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    require_cpu_energy_prerequisites()
    require_mips_prerequisites()

    # ── Step 1: Detect task ──
    print("=" * 60)
    print("AC-Prof Universal Profiler")
    print("=" * 60)

    from acprof.host.detect import detect_task

    task_info = detect_task(
        model_id=args.model,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
    )

    print(f"\n  Model:    {task_info.model_id}")
    print(f"  Task:     {task_info.pipeline_tag} (family={task_info.task_family})")
    print(f"  Backend:  {task_info.runtime_backend}")
    print(f"  Library:  {task_info.library_name}")
    print(f"  Revision: {task_info.model_revision}")
    print(f"  Detected: {task_info.detection_method}")

    # ── Step 2: Build Docker image ──
    from acprof.host.orchestrator import (
        build_image,
        collect_static_meta,
        enrich_static_meta_from_input_plan,
        enrich_static_meta_from_compute_plan,
        enrich_static_meta_from_execution_plan,
        merge_all_csvs,
        plan_input_scales,
        run_matrix,
        serialize_input_scales,
        write_static_meta_json,
    )

    if args.skip_build:
        from acprof.host.orchestrator import ImageInfo
        model_tag = task_info.model_id.replace("/", "--").replace(".", "_").lower()
        tag = f"acprof-{task_info.task_family}-{model_tag}:latest"
        image_info = ImageInfo(tag=tag)
        print(f"\n[build] Skipping build, using: {tag}")
    else:
        image_info = build_image(task_info, PROJECT_DIR)

    # ── Step 3: Collect static metadata ──
    cpu_list = _parse_int_list(args.cpus)
    mem_list = _parse_int_list(args.mems)
    gpu_list = _parse_str_list(args.gpus)
    massif_selected = args.execution_profile_tool in {"massif", "both"}
    nsys_selected = args.execution_profile_tool in {"nsys", "both"}
    reference_checks = []
    if massif_selected and args.massif_sampling == "per-scale":
        reference_checks.extend(
            [
                ("--massif-reference-cpu", args.massif_reference_cpu, cpu_list),
                ("--massif-reference-mem", args.massif_reference_mem, mem_list),
            ]
        )
    if nsys_selected and args.nsys_sampling != "full":
        reference_checks.append(
            ("--nsys-reference-mem", args.nsys_reference_mem, mem_list)
        )
    if nsys_selected and args.nsys_sampling == "per-scale":
        reference_checks.append(
            ("--nsys-reference-cpu", args.nsys_reference_cpu, cpu_list)
        )
    for option, value, resources in reference_checks:
        if value is not None and value not in resources:
            parser.error(
                f"{option}={value} must be present in the selected resource "
                f"matrix {resources}"
            )

    output_dir = os.path.join(PROJECT_DIR, args.output_dir, task_info.model_id.replace("/", "--"))
    os.makedirs(output_dir, exist_ok=True)

    static_meta_json = os.path.join(output_dir, "static_meta.json")
    collection_history_json = os.path.join(output_dir, COLLECTION_HISTORY_NAME)
    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    input_scale_type = scaling_cfg.param_name if scaling_cfg else ""

    static_meta = collect_static_meta(
        task_info=task_info,
        image_info=image_info,
        batch_size=args.batch_size,
        input_scale_type=input_scale_type,
        run_command=run_command,
        compute_profile_enabled=not compute_profile_disabled,
        execution_profile_enabled=args.execution_profile_tool != "none",
    )
    write_static_meta_json(static_meta, static_meta_json)
    write_collection_history_json(
        empty_collection_history(),
        collection_history_json,
    )

    # ── Step 4: Run profiling matrix ──
    try:
        planned_input_scales = plan_input_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=output_dir,
            input_scales=args.input_scales,
            workload_spec_path=args.workload_spec,
        )
    except Exception as exc:
        print(f"\n[scale][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    input_scales_arg = serialize_input_scales(planned_input_scales.scales)
    static_meta = enrich_static_meta_from_input_plan(
        static_meta,
        planned_input_scales,
    )
    write_static_meta_json(static_meta, static_meta_json)
    compute_profile_plan_file = ""
    if compute_profile_disabled:
        reason = (
            "--no-compute-profile (compatibility alias)"
            if args.no_compute_profile
            else "--compute-profile-tool none"
        )
        print(f"[compute] Compute profiling disabled by {reason}")
    else:
        try:
            from acprof.host.compute_profile import collect_compute_profile_plan

            compute_profile_plan_file = collect_compute_profile_plan(
                task_info=task_info,
                image_tag=image_info.tag,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                output_dir=output_dir,
                input_scale_plan_file=planned_input_scales.plan_file,
                advisor_root=args.advisor_root,
                ncu_root=args.ncu_root,
                advisor_repeat=args.advisor_repeat,
                torch_profiler_repeat=args.torch_profiler_repeat,
                ncu_repeat=args.ncu_repeat,
                keep_profiles=args.keep_compute_profiles,
                compute_profile_cpus=args.compute_profile_cpus,
                compute_profile_mem=args.compute_profile_mem,
                compute_profile_tool=args.compute_profile_tool,
            )
            static_meta = enrich_static_meta_from_compute_plan(
                static_meta,
                compute_profile_plan_file,
            )
            write_static_meta_json(static_meta, static_meta_json)
        except Exception as exc:
            print(f"[compute][WARN] Compute profiling unavailable: {exc}")

    execution_profile_plan_file = ""
    if args.execution_profile_tool == "none":
        print(
            "[execution-profile] Massif/Nsight Systems profiling disabled "
            "(enable with --execution-profile-tool)"
        )
    else:
        try:
            from acprof.host.execution_profile import (
                collect_execution_profile_plan,
            )

            execution_profile_plan_file = collect_execution_profile_plan(
                task_info=task_info,
                image_tag=image_info.tag,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                output_dir=output_dir,
                input_scale_plan_file=planned_input_scales.plan_file,
                project_dir=PROJECT_DIR,
                tool_mode=args.execution_profile_tool,
                massif_sampling=args.massif_sampling,
                massif_reference_cpu=args.massif_reference_cpu,
                massif_reference_mem=args.massif_reference_mem,
                massif_repeat=args.massif_repeat,
                nsys_sampling=args.nsys_sampling,
                nsys_reference_cpu=args.nsys_reference_cpu,
                nsys_reference_mem=args.nsys_reference_mem,
                nsys_repeat=args.nsys_repeat,
                nsys_root=args.nsys_root,
                keep_profiles=args.keep_execution_profiles,
            )
            static_meta = enrich_static_meta_from_execution_plan(
                static_meta,
                execution_profile_plan_file,
            )
            write_static_meta_json(static_meta, static_meta_json)
        except Exception as exc:
            print(
                "[execution-profile][WARN] Execution profiling unavailable: "
                f"{exc}"
            )

    total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)
    n_scales = len(planned_input_scales.scales)
    total_iters = total_cases * n_scales * (args.warmup + args.repeat)

    print(f"\n  Resource matrix: {len(cpu_list)} CPUs x {len(mem_list)} MEMs x {len(gpu_list)} GPUs = {total_cases} cases")
    print(f"  Input scales: {n_scales} levels")
    print(f"  Scale source: {planned_input_scales.source}")
    print(f"  Validated scales: {input_scales_arg}")
    print(f"  Iterations per case: {args.warmup} warmup + {args.repeat} repeat")
    if args.repeat_in_window > 0:
        repeat_desc = str(args.repeat_in_window)
    else:
        repeat_desc = f"auto target {args.repeat_window_seconds:.1f}s"
    print(f"  Requests per iteration: {repeat_desc}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Output: {output_dir}")
    print()

    try:
        csv_paths = run_matrix(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            output_dir=output_dir,
            project_dir=PROJECT_DIR,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeat=args.repeat,
            repeat_in_window=args.repeat_in_window,
            repeat_window_seconds=args.repeat_window_seconds,
            sample_hz=args.sample_hz,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            idle_debug=args.idle_debug,
            sniff_iface=args.sniff_iface,
            input_scales=input_scales_arg,
            input_scale_plan_file=planned_input_scales.plan_file,
            compute_profile_plan_file=compute_profile_plan_file,
            execution_profile_plan_file=execution_profile_plan_file,
        )
    except PacketLatencyError as exc:
        print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except EnergyProfilingError as exc:
        print(f"\n[energy][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except MIPSProfilingError as exc:
        print(f"\n[mips][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 5: Merge all CSVs ──
    if csv_paths:
        final_csv = os.path.join(output_dir, "result_all.csv")
        merge_all_csvs(csv_paths, final_csv)
        _cleanup_intermediate_results(csv_paths, output_dir, final_csv)
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n{'='*60}")
        print(f"Profiling complete!")
        print(f"  Static meta:      {static_meta_json}")
        print(f"  Collection log:   {collection_history_json}")
        print(f"  Merged results:   {final_csv}")
        print(f"  Total elapsed:    {elapsed}")
        print(f"  Intermediate files from this run were cleaned up.")
        print(f"{'='*60}")
    else:
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n[WARN] No results produced after {elapsed}. Static meta is still available: {static_meta_json}")


def main():
    """Run profiling and always finalize an active tmux terminal recording."""
    global _ACTIVE_TMUX_TERMINAL_LOG

    _ACTIVE_TMUX_TERMINAL_LOG = None
    try:
        return _run_main()
    finally:
        terminal_log = _ACTIVE_TMUX_TERMINAL_LOG
        _ACTIVE_TMUX_TERMINAL_LOG = None
        if terminal_log is not None:
            _stop_tmux_terminal_log(terminal_log)


if __name__ == "__main__":
    main()
