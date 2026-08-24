"""Best-effort lifecycle notifications for host-side AC-Prof commands.

Notification delivery is deliberately kept outside measurement windows.  This
module sends only before collection begins, after a case has released its
resources, or after the whole command ends.  It never starts a background
worker or opens a connection while profiling is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import socket
import time
from typing import Mapping, Optional
from urllib.parse import parse_qs, urlparse

import requests


WECOM_WEBHOOK_ENV = "ACPROF_WECOM_WEBHOOK_URL"
WECOM_WEBHOOK_HOST = "qyapi.weixin.qq.com"
WECOM_WEBHOOK_PATH = "/cgi-bin/webhook/send"
DEFAULT_NOTIFY_TIMEOUT_SECONDS = 5.0
DEFAULT_NOTIFY_ATTEMPTS = 2
DEFAULT_NOTIFY_RETRY_DELAY_SECONDS = 0.5

_WECOM_WEBHOOK_RE = re.compile(
    r"https://qyapi\.weixin\.qq\.com/cgi-bin/webhook/send\?[^\s]+",
    flags=re.IGNORECASE,
)


class NotificationError(RuntimeError):
    """Base class for safe-to-display notification errors."""


class NotificationConfigError(NotificationError):
    """Raised when notification configuration is missing or malformed."""


class NotificationDeliveryError(NotificationError):
    """Raised when a configured provider does not accept a message."""


@dataclass(frozen=True)
class NotificationEvent:
    """Provider-independent summary of one profiling command."""

    status: str
    model_id: str
    output_dir: str
    elapsed_seconds: float
    run_command: Optional[str] = None
    total_cases: Optional[int] = None
    completed_cases: Optional[int] = None
    result_rows: Optional[int] = None
    error_rows: Optional[int] = None
    final_csv: Optional[str] = None
    terminal_log: Optional[str] = None
    detail: Optional[str] = None
    host: str = field(default_factory=socket.gethostname)


def redact_notification_secrets(value: object) -> str:
    """Remove webhook credentials from text before it reaches logs/messages."""
    text = str(value or "")
    return _WECOM_WEBHOOK_RE.sub(
        "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=<redacted>",
        text,
    )


def _single_line(value: object, limit: int = 500) -> str:
    text = redact_notification_secrets(value).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def render_notification_text(event: NotificationEvent) -> str:
    """Render a compact Enterprise WeChat text message."""
    status_labels = {
        "started": ("AC-Prof 实验开始", "已启动"),
        "success": ("AC-Prof 采集完成", "成功"),
        "partial": ("AC-Prof 采集部分完成", "部分成功"),
        "no_results": ("AC-Prof 采集无结果", "无结果"),
        "failed": ("AC-Prof 采集失败", "失败"),
        "cancelled": ("AC-Prof 采集已取消", "已取消"),
        "progress": ("AC-Prof 采集进度", "进行中"),
    }
    title, status_label = status_labels.get(
        event.status,
        ("AC-Prof 采集状态", _single_line(event.status, limit=80) or "未知"),
    )
    lines = [
        title,
        f"状态：{status_label}",
        f"模型：{_single_line(event.model_id, limit=200)}",
        f"主机：{_single_line(event.host, limit=200)}",
        f"耗时：{_format_elapsed(event.elapsed_seconds)}",
    ]

    if event.run_command:
        lines.append(f"指令：{_single_line(event.run_command, limit=1500)}")
    if event.total_cases is not None:
        completed = "?" if event.completed_cases is None else event.completed_cases
        progress_suffix = ""
        if event.completed_cases is not None and event.total_cases > 0:
            percent = 100.0 * event.completed_cases / event.total_cases
            progress_suffix = f"（{percent:.1f}%）"
        lines.append(f"资源组合：{completed}/{event.total_cases}{progress_suffix}")
    if event.result_rows is not None:
        label = "当前 case 结果行" if event.status == "progress" else "结果行"
        lines.append(f"{label}：{event.result_rows}")
    if event.error_rows is not None:
        label = "当前 case 异常行" if event.status == "progress" else "异常行"
        lines.append(f"{label}：{event.error_rows}")
    if event.final_csv:
        lines.append(f"结果：{_single_line(event.final_csv, limit=500)}")
    if event.terminal_log:
        lines.append(f"日志：{_single_line(event.terminal_log, limit=500)}")
    if event.detail:
        lines.append(f"说明：{_single_line(event.detail, limit=500)}")
    if not event.final_csv:
        lines.append(f"输出目录：{_single_line(event.output_dir, limit=500)}")

    return "\n".join(lines)


def validate_wecom_webhook_url(webhook_url: str) -> str:
    """Validate the public Enterprise WeChat group-robot endpoint shape."""
    value = (webhook_url or "").strip()
    if not value:
        raise NotificationConfigError(
            f"环境变量 {WECOM_WEBHOOK_ENV} 未配置"
        )

    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    query = parse_qs(parsed.query, keep_blank_values=True)
    key_values = query.get("key", [])
    valid = (
        parsed.scheme == "https"
        and parsed.hostname == WECOM_WEBHOOK_HOST
        and port in {None, 443}
        and not parsed.username
        and not parsed.password
        and parsed.path == WECOM_WEBHOOK_PATH
        and not parsed.fragment
        and set(query) == {"key"}
        and len(key_values) == 1
        and bool(key_values[0].strip())
    )
    if not valid:
        raise NotificationConfigError(
            f"环境变量 {WECOM_WEBHOOK_ENV} 不是有效的企业微信群机器人 Webhook"
        )
    return value


class WeComWebhookNotifier:
    """Synchronous, post-run Enterprise WeChat group-robot notifier."""

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout_seconds: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
        attempts: int = DEFAULT_NOTIFY_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_NOTIFY_RETRY_DELAY_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if attempts <= 0:
            raise ValueError("attempts must be > 0")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")

        self._webhook_url = validate_wecom_webhook_url(webhook_url)
        self.timeout_seconds = float(timeout_seconds)
        self.attempts = int(attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: object,
    ) -> "WeComWebhookNotifier":
        env = os.environ if environ is None else environ
        return cls(env.get(WECOM_WEBHOOK_ENV, ""), **kwargs)

    def __repr__(self) -> str:
        return (
            "WeComWebhookNotifier(webhook_url=<redacted>, "
            f"timeout_seconds={self.timeout_seconds!r}, attempts={self.attempts!r})"
        )

    def send(self, event: NotificationEvent) -> None:
        payload = {
            "msgtype": "text",
            "text": {"content": render_notification_text(event)},
        }
        last_error = "未知错误"

        for attempt_index in range(self.attempts):
            try:
                response = requests.post(
                    self._webhook_url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                status_code = int(getattr(response, "status_code", 0) or 0)
                if not 200 <= status_code < 300:
                    raise NotificationDeliveryError(
                        f"企业微信返回 HTTP {status_code or 'unknown'}"
                    )
                try:
                    body = response.json()
                except (TypeError, ValueError):
                    raise NotificationDeliveryError("企业微信响应不是有效 JSON")
                if not isinstance(body, dict):
                    raise NotificationDeliveryError("企业微信响应格式无效")

                errcode = body.get("errcode")
                if errcode != 0:
                    errmsg = _single_line(body.get("errmsg", "unknown"), limit=200)
                    raise NotificationDeliveryError(
                        f"企业微信拒绝消息：errcode={errcode!r}, errmsg={errmsg}"
                    )
                return
            except requests.RequestException as exc:
                # requests exceptions often embed the full request URL.  Only
                # retain the exception type so the webhook key cannot leak.
                last_error = f"企业微信请求失败（{type(exc).__name__}）"
            except NotificationDeliveryError as exc:
                last_error = str(exc)

            if attempt_index + 1 < self.attempts and self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)

        raise NotificationDeliveryError(last_error)
