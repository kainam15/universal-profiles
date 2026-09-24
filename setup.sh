#!/usr/bin/env bash
# Bootstrap the host CLI; model/runtime preparation stays in the existing run path.
set -euo pipefail

launch_tui=true
modify_path=true
for argument in "$@"; do
    case "$argument" in
        --no-tui) launch_tui=false ;;
        --no-modify-path) modify_path=false ;;
        -h|--help)
            cat <<'USAGE'
用法：./setup.sh [--no-tui] [--no-modify-path]

安装当前 checkout 的 AC-Prof，检查 basic CPU 环境，并在交互终端打开入门 TUI。
  --no-tui          只安装和检查，输出后续启动命令
  --no-modify-path  不修改 shell 启动文件；仍可通过完整路径启动

需要原生 Linux x86_64、本机 Docker Engine/Buildx、cgroup v2 和网络。
自动准备 uv/Python；不安装系统软件、不更改 Docker 权限或 GPU 驱动。
选定模型并开始采集后，才按需准备镜像和模型权重。
USAGE
            exit 0 ;;
        *) printf '未知参数：%s；运行 ./setup.sh --help 查看用法。\n' "$argument" >&2; exit 2 ;;
    esac
done

fail() {
    printf '[setup] %s\n' "$1" >&2
    exit 1
}

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
[[ -f pyproject.toml && -f requirements.lock ]] || fail "请从完整 AC-Prof 源码目录运行 setup.sh。"

printf '[1/3] 检查主机与 Docker\n'
[[ "$(uname -s)" == Linux ]] || fail "采集需要原生 Linux；Windows/macOS 不支持。"
[[ "$(uname -m)" == x86_64 ]] || fail "当前依赖镜像只支持 Linux x86_64。"
kernel_release="$(uname -r)"
[[ "${kernel_release,,}" != *microsoft* && "${kernel_release,,}" != *wsl* ]] || fail "WSL 不支持采集，请使用原生 Linux。"
[[ -f /sys/fs/cgroup/cgroup.controllers ]] || fail "需要统一 cgroup v2；配置后重启主机再运行。"
command -v docker >/dev/null 2>&1 || fail "未找到 Docker。安装本机 Docker Engine/Buildx：https://docs.docker.com/engine/install/"
docker info >/dev/null || fail "当前用户无法访问 Docker。请启动本机 Docker Engine，检查 socket 权限和 context；修复后重试。"
docker buildx version >/dev/null || fail "缺少 Docker Buildx plugin；请安装 docker-buildx-plugin 后重试。"
df -h .

printf '[2/3] 安装当前 checkout 的 AC-Prof\n'
uv_install_dir="${UV_INSTALL_DIR:-${HOME}/.local/bin}"
uv_command="$(command -v uv || true)"
if [[ -z "$uv_command" ]]; then
    for candidate in "$uv_install_dir/uv" "$project_dir/.venv/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            uv_command="$candidate"
            break
        fi
    done
fi
if [[ -z "$uv_command" ]]; then
    # Match the development/Release toolchain and use the upstream installer.
    installer_url='https://astral.sh/uv/0.12.13/install.sh'
    installer_dir="$(mktemp -d)"
    trap 'rm -rf -- "$installer_dir"' EXIT
    printf '[setup] 下载 uv 0.12.13 到 %s\n' "$uv_install_dir"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 2 --connect-timeout 15 --max-time 120 \
            --output "$installer_dir/install.sh" "$installer_url" || fail "uv 下载失败；检查网络后重新运行 setup.sh。"
    elif command -v wget >/dev/null 2>&1; then
        wget --timeout=120 --tries=3 -O "$installer_dir/install.sh" "$installer_url" || fail "uv 下载失败；检查网络后重新运行 setup.sh。"
    else
        fail "下载 uv 需要 curl 或 wget；也可先自行安装 uv 后重试。"
    fi
    UV_INSTALL_DIR="$uv_install_dir" UV_NO_MODIFY_PATH=1 sh "$installer_dir/install.sh" || fail "uv 安装失败；检查下载与目录权限后重试。"
    uv_command="$uv_install_dir/uv"
fi

# Reinstall this checkout even when its version is unchanged; uv reuses dependency caches.
"$uv_command" tool install --python 3.10 --constraint "$project_dir/requirements.lock" \
    --reinstall-package acprof "$project_dir" || fail "AC-Prof 安装失败；检查上方错误，修复网络或依赖问题后重新运行 setup.sh。"
tool_bin_dir="$("$uv_command" tool dir --bin)"
acprof_command="$tool_bin_dir/acprof"
[[ -x "$acprof_command" ]] || fail "安装后未找到 $acprof_command。"
path_updated=false
if "$modify_path"; then
    if "$uv_command" tool update-shell; then
        path_updated=true
    else
        printf '[setup] 未能更新 shell PATH；仍可使用下方完整路径命令。\n' >&2
    fi
fi

printf '[3/3] 检查 basic CPU 采集条件\n'
"$acprof_command" doctor --profiling-mode basic --gpus off || fail "环境检查未通过；按 doctor 建议修复后重新运行 setup.sh。"

mkdir -p results
run_output="$(mktemp -d "results/first-run-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
tui_arguments=(tui --model google-bert/bert-base-uncased --preset smoke --output-dir "$run_output")
printf '\n[setup] 安装与前置检查完成。入门配置：basic / CPU / 单次请求 / 通知关闭。\n'
printf '[setup] 结果目录：%s/%s\n' "$project_dir" "$run_output"
printf '[setup] GPU 和 full 按需检查：'
printf '%q ' "$acprof_command" doctor --profiling-mode full --gpus on
printf '\n'
if "$path_updated"; then
    printf '[setup] 后续可在新终端使用 acprof。\n'
fi
printf '[setup] 当前终端可执行以下完整路径命令：\n  '
printf '%q ' "$acprof_command" "${tui_arguments[@]}"
printf '\n[setup] 选择模型并点击“开始采集”后，才会下载模型、准备镜像和运行实验。\n'
if "$launch_tui" && [[ -t 0 && -t 1 && "${TERM:-dumb}" != dumb ]]; then
    "$acprof_command" "${tui_arguments[@]}"
else
    printf '[setup] 已跳过 TUI 自动启动；请使用上方命令继续。\n'
fi
