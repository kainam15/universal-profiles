"""界面数值与空值显示；不改变配置、进度或结果协议。"""

from acprof.tui.i18n import message


NOT_APPLICABLE = "—"
CALCULATING = "…"
UNKNOWN = message("未知")
STATUS_LEGEND = message("— 不适用 · … 正在计算 · 未知 无法确定")


def format_input_number(value: int | float | str) -> str:
    """去掉整数的 .0，保留非整数的完整精度与尚未校验的输入。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
