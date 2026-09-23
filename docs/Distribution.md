# 安装包、standalone 与发布

AC-Prof 支持源码开发、`uv tool install` 隔离安装和 Linux x86_64 standalone。
三种入口执行相同的主机采集代码；Docker Engine、cgroup v2、GPU driver 和采集工具仍由主机提供。
安装与首次运行见[安装指南](Getting_Started.md)，环境检查参数见 [doctor](CLI_Reference.md#acprof-doctor)。

## Python 工具安装

在包含 `pyproject.toml` 的源码目录中：

```bash
uv tool install .
uv tool update-shell
```

重新打开终端或按 uv 提示刷新 `PATH` 后运行：

```bash
acprof --version
acprof doctor --profiling-mode basic
acprof tui
```

也可以安装 Release 的 wheel：`uv tool install ./acprof-0.2.0-py3-none-any.whl`。
远端源码包含本版本后，可直接运行
`uv tool install git+https://github.com/kainam15/universal-profiles.git`；复现实验应固定 Git tag 或 commit。
这里只使用源码和 Release 制品，不假设 PyPI 已有同名官方发行包。

安装后的公共命令是 `acprof run / tui / probe / plot / doctor / profile / audit / stats`，
也支持 `python -m acprof`。根目录的 Python 脚本保留给源码使用。
`run --help` 等命令沿用各自的参数定义；顶层帮助和版本查询不会加载 Textual、绘图库或推理框架。
Python 依赖声明位于 `pyproject.toml`；开发和 Release 构建采用 `requirements.lock` 中已验证的制品。

## 工作目录与资源

输出目录、用户 workload 相对路径及 `.env` / `.env.local` 相对于**启动时的当前工作目录**。
TUI 设置仍写入 XDG 用户配置目录，以工作目录的摘要隔离不同实验工作区。
在源码仓库根目录启动时，路径行为与原来的操作示例相同。

wheel 内置 Dockerfile、平台/环境锁、扩展声明、音频素材及构建所需的 Python 源码。
`installation.resource_root()` 定位这些只读资源；它不是输出目录。
构建 hook 使用明确的目录、文件后缀白名单，排除 `.env`、缓存、结果和 Agent 规则。
Docker 模型层仍由本机按固定 revision 下载，令牌经 BuildKit secret 传入。

## Linux standalone

[`release.yml`](../.github/workflows/release.yml) 在 Ubuntu 22.04、Python 3.10 上构建
`acprof-linux-x86_64`，目标为 glibc 2.35+ 的原生 Linux x86_64。
它包含 Python 解释器、主机依赖和构建资源；目标机无需先安装 Python 或 uv。
Docker、RAPL、perf、抓包及 NVIDIA 的要求仍按所选模式检查。

Release 发布后，从 [GitHub Releases](https://github.com/kainam15/universal-profiles/releases)
下载 `acprof-linux-x86_64` 和 `SHA256SUMS`，在下载目录核对并运行：

```bash
sha256sum --check --ignore-missing SHA256SUMS
chmod +x acprof-linux-x86_64
./acprof-linux-x86_64 doctor --profiling-mode basic
./acprof-linux-x86_64 tui
```

PyInstaller 单文件模式启动时会解压到临时目录，该目录需要支持可执行文件和符号链接。
需要时通过 `TMPDIR` 选择合适的临时目录。采集子进程使用同一可执行文件的内部 worker 入口，
不会把它误当成系统 Python；worker 只允许 client 和两个 packet 模块。

本地重现构建（已有 `.venv/bin/uv` 时可替换下列 `uv`）：

```bash
uv build --out-dir internal-testing/distribution/dist
uv venv internal-testing/distribution/venv --python 3.10
uv pip install --python internal-testing/distribution/venv/bin/python -r requirements.lock
uv pip install --python internal-testing/distribution/venv/bin/python --no-deps internal-testing/distribution/dist/*.whl
uv pip install --python internal-testing/distribution/venv/bin/python 'pyinstaller==6.22.3'
internal-testing/distribution/venv/bin/python scripts/check_distribution.py
internal-testing/distribution/venv/bin/python -m PyInstaller --noconfirm packaging/acprof.spec \
  --distpath internal-testing/distribution/standalone --workpath internal-testing/distribution/build
internal-testing/distribution/venv/bin/python scripts/check_distribution.py \
  --binary internal-testing/distribution/standalone/acprof
```

本机生成的 binary 使用本机构建环境的 glibc 下限；只有相应 runner 上构建并验证的资产才能按
Release 的平台范围分发。wheel/standalone 的 smoke 验证不能替代真实 Docker/GPU 采集。

## 发布入口与范围

维护者更新 `acprof.__version__`，验证后推送对应 `v<version>` tag：

- `release.yml` 构建 sdist、wheel、standalone 与 SHA256 清单；完成隔离安装和 worker 验证后发布 GitHub Release。
- `runtime-images.yml` 先构建/核验 4 个平台，再让 24 个环境 job 拉取已发布平台，构建、核验并发布环境。
- 手动运行 Release workflow 只构建并保存 Actions artifacts；手动运行 GHCR workflow 会发布镜像。

工作流文件存在不代表远端资产已发布。实际发布需要仓库中的 Actions 正常完成，以及 GitHub 的
`contents: write` / `packages: write` 权限。GHCR package 首次发布后，维护者需在 package 设置中
确认 public 可见性，匿名用户才能直接拉取；私有 package 需要先 `docker login ghcr.io`。

GHCR 只预构建平台和依赖环境，不发布模型权重、用户数据或包含令牌的层。
拉取策略、内容身份和失败回退见[预构建依赖镜像](Runtime_Compatibility.md#ghcr-预构建依赖镜像)。
发布脚本的 `--report` 保存 image ID、内容 tag 和清单核验范围；依赖核验不代表 GPU 推理已经通过。

## 参考实现与取舍

- [uv tools](https://github.com/astral-sh/uv/blob/main/docs/guides/tools.md)（MIT / Apache-2.0）：
  采用标准 console script 与隔离工具环境，不增加自己的安装器。
- [Hatch build hooks](https://github.com/pypa/hatch/tree/master/backend/src/hatchling/builders/hooks)
  （MIT）：用一个小型 build hook 打包既有资源，不改变运行时依赖和镜像配方。
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)（GPL 与分发例外）：使用官方冻结工具，
  按其[资源与子进程说明](https://pyinstaller.org/en/stable/runtime-information.html)处理真实源码、动态模块和系统库路径。
  构建工具不进入主机运行依赖，不将 Torch/CUDA 安装到主机包。
- [Docker Actions](https://github.com/docker/build-push-action)（Apache-2.0）：借鉴 GHCR 登录与分阶段发布方式，
  继续使用项目自己的依赖构建器，保留完整包清单、父镜像 ID 和内容指纹验证。

这些工具均有官方维护仓库；这里只复用公开接口和发布方式。安装、doctor、下载和镜像核验都位于测量窗口之外。
