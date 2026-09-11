"""保留原 TUI helper 导入路径的兼容入口。"""

from acprof.config import (
    DEFAULT_COMPUTE_PROFILE_TOOL,
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
)
from acprof.host.env_utils import load_project_env
from acprof.monitors.perf_mips import MIPSProfilingError, resolve_perf_command_prefix
from acprof.tui.i18n import message

from acprof.tui.commands import (
    COMPUTE_PROFILE_TOOLS,
    EXECUTION_PROFILE_TOOLS,
    GPU_MODES,
    NOTIFY_MODES,
    RunConfig,
    TASK_FAMILIES,
    TuiConfigError,
    _csv_values,
    _format_number,
    _number,
    _positive_float_csv,
    _positive_int_csv,
    build_plot_command,
    build_probe_command,
    build_profile_command,
    build_run_command,
    format_command,
    parse_slash_command,
)

from acprof.tui.progress import (
    ANSI_ESCAPE_RE,
    CASE_RE,
    FINAL_CSV_RE,
    LARGEST_PROBE_MEMORY_RESULT_RE,
    LARGEST_PROBE_MEMORY_TRY_RE,
    LARGEST_PROBE_RESULT_RE,
    LARGEST_PROBE_SCAN_RE,
    LARGEST_PROBE_START_RE,
    LARGEST_PROBE_SUMMARY_RE,
    MATRIX_RE,
    MERGE_CSV_RE,
    ProgressSnapshot,
    RunProgressTracker,
    TASK_SUPPORT_ERROR_RE,
    _format_probe_duration,
)

from acprof.tui.diagnostics import (
    PreflightCheck,
    ResultSummary,
    _completed_command,
    _readable_rapl_paths,
    quick_preflight,
    summarize_result_csv,
)
