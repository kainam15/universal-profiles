---
name: acprof-textual-regression
description: 用于修改或排查 AC-Prof 的 Textual TUI 布局、焦点与光标、日志交互、设置和中英文文案；验证终端显示及交互，并保护正式测量窗口。
---

# AC-Prof TUI 回归检查

## 使用范围

针对 Textual 终端界面的可观察行为。先确认用户所在终端、复现操作和窗口尺寸，再选择相关检查；不把网页设计或浏览器测试直接套到 TUI。

本流程参考 [Textual 官方测试指南](https://textual.textualize.io/guide/testing/) 及其 [GitHub 文档](https://github.com/Textualize/textual/blob/main/docs/guide/testing.md)，复用项目现有 `unittest` 和 Textual `Pilot`。无需引入 `pytest` 或快照插件来遵循本技能。

## 按需查阅

| 需要确认的内容 | 文档入口 |
| --- | --- |
| 页面操作、快捷键与用户可见行为 | [README 的 TUI 说明](../../../README.md#交互式终端界面) |
| 设置持久化、语言或历史格式 | [TUI 本地设置](../../../docs/CLI_Reference.md#tui-本地设置) |
| 模块职责、兼容入口或 mock 位置 | [TUI 与兼容维护](../../../docs/Architecture.md#tui-与兼容维护) |

本技能维护交互验证流程；操作和设置契约由上述文档维护。只读与当前改动相关的章节；纯布局或文案调整无需读取整份指标字典。若改动会影响命令参数、测量流程或产物，再沿调用链读取相应契约。

## 从实际模块和测试进入

按[架构分工](../../../docs/Architecture.md#tui-与兼容维护)定位实现，再用[测试入口](../../../docs/Testing.md#自动化验证入口)选择相关测试。
只覆盖受影响路径，遇到 API 差异时核对当前 `.venv` 中的 Textual 版本。

## 自动化交互检查

沿用 `unittest.IsolatedAsyncioTestCase`、`app.run_test(size=(width, height))` 和 `Pilot`；用临时 `settings_path` 隔离用户设置。模拟真正启动采集、Docker 和通知的边界，不能因点击“开始”就在测试中跑真实实验。

检查实际键盘输入、点击、焦点、滚动和输出状态，而不只断言 CSS 类名或内部字段存在。布局变化后用框架的事件处理及等待机制，不用不断增加固定 `sleep` 掩盖竞态。

优先覆盖用户报告的尺寸，并按[终端验证范围](../../../docs/Testing.md#tui-与终端证据)检查适用尺寸和运行中 resize。确认按钮有面积、位于可见区域且没有被其他控件覆盖，点击会执行预期动作。

## 按改动选择回归项

- **输入：** 中文宽字符、横向滚动、选择替换、placeholder、失焦、窗口缩放和退出后的终端恢复；光标位置按终端 cell 宽度判断，不只数 Python 字符。
- **日志：** 选择与复制、双击和放大手势、滚动条、分页、清空、新输出追加。用户开始浏览或选择后，不能被新日志强制带回末尾；返回末尾使用明确的 `follow_tail()` 行为。
- **设置：** 即时生效、保存、重启恢复、默认值、错误输入和原子写入；使用临时目录，不覆盖真实偏好。
- **语言：** 中英文切换后的可达性与长度变化，文案通过 `i18n.py` 的既有机制维护。
- **操作反馈：** 开始、停止、取消和错误提示要在当前页面可见；确认控件交互不误触发重复任务。

## 保护测量窗口

检查 `ProgressSnapshot.measurement_active` 的进入和退出路径，以及 `set_input_cursor_blink_enabled()`、计时刷新和日志缓冲等相关行为。修改不能让绘图、状态轮询、额外采集或光标动画在正式测量期间持续运行。

验证测量结束或失败后交互能恢复，CLI 命令与运行参数仍一致。TUI 的显示变化不能改变实验的样本、请求次数、指标口径或 CSV 内容。

## 真实终端验证

Headless 测试能验证布局和交互，但不能证明 VS Code 终端、SSH 或 tmux 中的原生光标、剪贴板和闪烁效果。涉及这些现象时，在相应终端中复现并保留必要的截图或可观察证据。

无法访问用户的终端时，明确已验证的是 headless、tmux 还是其他环境，不把 SVG 截图或终端控制序列已输出当成用户端效果已经确认。实际采集期间不启动额外 TUI 来截图。

## 验证命令与交付

按变更选择相关测试，例如：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_tui_input.py' -v
```

只有确实需要整组回归时再使用 `-p 'test_tui*.py'`，并完成项目规定的其他检查。记录测试结果、终端尺寸、复现动作与仍未确认的环境。

操作行为变化时更新 README 的对应说明；设置格式或持久化语义变化时更新 docs/CLI_Reference.md；模块边界变化时更新架构文档。通用约束遵循已加载的项目规则。
