"""Argument declarations for the main profiling command."""
import argparse

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
)


def build_parser(*, default_notify_provider: str = "auto") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AC-Prof: Universal HuggingFace Model Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --model bert-base-uncased
  python run.py --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
  python run.py --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
  python run.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 --gpus on
        """,
    )

    # Required
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same experiment using its recorded image, input plan and completed cases")

    # Detection overrides
    parser.add_argument("--task", default=None, help="Override pipeline_tag (e.g., text-generation)")
    parser.add_argument(
        "--task-family",
        default=None,
        help="Override task family (nlp/cv/audio/timeseries/diffusion/multimodal/structured)",
    )
    parser.add_argument("--backend", default=None, help="Override runtime backend (transformers_pipeline/chronos/...)")

    # Resource matrix
    parser.add_argument("--cpus", default="1,2,4,8", help="CPU core counts (comma-separated)")
    parser.add_argument("--mems", default="2,4,8,16", help="Memory caps in GB (comma-separated)")
    parser.add_argument("--gpus", default="off,on", help="GPU modes (comma-separated: off,on)")
    startup_oom_pruning_group = parser.add_mutually_exclusive_group()
    startup_oom_pruning_group.add_argument(
        "--prune-startup-oom",
        dest="prune_startup_oom",
        action="store_true",
        help=(
            "Explicitly enable the default startup-OOM pruning behavior"
        ),
    )
    startup_oom_pruning_group.add_argument(
        "--no-prune-startup-oom",
        dest="prune_startup_oom",
        action="store_false",
        help=(
            "Disable startup-OOM pruning and independently attempt every "
            "selected CPU/memory/GPU case"
        ),
    )
    parser.set_defaults(prune_startup_oom=True)

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
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout for each formal /predict request",
    )
    parser.add_argument("--sample-hz", type=float, default=20.0, help="GPU energy sampling rate")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=DEFAULT_IDLE_SECONDS,
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
            "Path to a CV, audio, multimodal, diffusion or structured workload manifest. ASR defaults to the "
            "bundled LibriSpeech short-form manifest."
        ),
    )

    # Compute profiling
    parser.add_argument(
        "--compute-profile-tool",
        choices=("none", "both", "torch", "ncu", "vendor"),
        default=DEFAULT_COMPUTE_PROFILE_TOOL,
        help=(
            "Compute FLOP profiler (default: none): none skips all compute "
            "probes; both independently collects torch_profiler_eager logical "
            "FLOP and ncu GPU executed FLOP"
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
    parser.add_argument(
        "--model-download-policy", choices=("auto", "full"), default="auto",
        help="Model files: auto selects verified loader formats; full keeps the complete repository",
    )
    parser.add_argument("--sniff-iface", default="docker0", help="Network interface for tcpdump")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the local model image if present; automatically build it if missing",
    )
    parser.add_argument(
        "--notify",
        choices=("auto", "none", "wecom"),
        default=default_notify_provider,
        help=(
            "Notification mode: auto (default) enables WeCom when "
            "ACPROF_WECOM_WEBHOOK_URL is configured; none disables notifications"
        ),
    )

    return parser
