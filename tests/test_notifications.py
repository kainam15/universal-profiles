import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from acprof.notifications import (
    NotificationConfigError,
    NotificationDeliveryError,
    NotificationEvent,
    WECOM_WEBHOOK_ENV,
    WeComWebhookNotifier,
    redact_notification_secrets,
    render_notification_text,
    validate_wecom_webhook_url,
)


WEBHOOK_KEY = "00000000-1111-2222-3333-444444444444"
WEBHOOK_URL = (
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=" + WEBHOOK_KEY
)


class WeComNotificationTests(unittest.TestCase):
    def _event(self, **overrides) -> NotificationEvent:
        values = {
            "status": "success",
            "model_id": "org/model",
            "output_dir": "/tmp/results/org--model",
            "elapsed_seconds": 65.0,
            "total_cases": 2,
            "completed_cases": 2,
            "result_rows": 12,
            "error_rows": 0,
            "final_csv": "/tmp/results/org--model/result_all.csv",
            "host": "test-host",
        }
        values.update(overrides)
        return NotificationEvent(**values)

    def test_validate_accepts_only_wecom_group_robot_endpoint(self) -> None:
        self.assertEqual(validate_wecom_webhook_url(WEBHOOK_URL), WEBHOOK_URL)

        invalid_urls = (
            "",
            "http://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x",
            "https://example.com/cgi-bin/webhook/send?key=x",
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=x",
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/send",
            "https://qyapi.weixin.qq.com:invalid/cgi-bin/webhook/send?key=x",
        )
        for value in invalid_urls:
            with self.subTest(value=value), self.assertRaises(NotificationConfigError):
                validate_wecom_webhook_url(value)

    def test_from_env_requires_configuration_without_echoing_secret(self) -> None:
        with self.assertRaises(NotificationConfigError) as raised:
            WeComWebhookNotifier.from_env({})
        self.assertIn(WECOM_WEBHOOK_ENV, str(raised.exception))

        notifier = WeComWebhookNotifier.from_env({WECOM_WEBHOOK_ENV: WEBHOOK_URL})
        self.assertNotIn(WEBHOOK_KEY, repr(notifier))
        self.assertIn("<redacted>", repr(notifier))

    def test_render_includes_partial_counts_and_redacts_webhook(self) -> None:
        text = render_notification_text(
            self._event(
                status="partial",
                error_rows=2,
                detail=f"request failed for {WEBHOOK_URL}",
            )
        )

        self.assertIn("采集部分完成", text)
        self.assertIn("资源组合：2/2", text)
        self.assertIn("异常行：2", text)
        self.assertNotIn(WEBHOOK_KEY, text)
        self.assertIn("key=<redacted>", text)
        self.assertNotIn(WEBHOOK_KEY, redact_notification_secrets(WEBHOOK_URL))

    def test_render_progress_reports_elapsed_case_and_percentage(self) -> None:
        text = render_notification_text(
            self._event(
                status="progress",
                elapsed_seconds=3661.0,
                completed_cases=1,
                total_cases=4,
                result_rows=7,
                error_rows=1,
                final_csv=None,
                detail="刚完成：CPU=2, MEM=4GB, GPU=off",
            )
        )

        self.assertIn("AC-Prof 采集进度", text)
        self.assertIn("耗时：1小时1分1秒", text)
        self.assertIn("资源组合：1/4（25.0%）", text)
        self.assertIn("当前 case 结果行：7", text)
        self.assertIn("当前 case 异常行：1", text)
        self.assertIn("CPU=2, MEM=4GB, GPU=off", text)

    def test_send_posts_text_payload_with_short_timeout(self) -> None:
        response = SimpleNamespace(status_code=200, json=lambda: {"errcode": 0})
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=1,
            timeout_seconds=3.0,
        )

        with patch(
            "acprof.notifications.requests.post",
            return_value=response,
        ) as post:
            notifier.send(self._event())

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args, (WEBHOOK_URL,))
        self.assertEqual(kwargs["timeout"], 3.0)
        self.assertEqual(kwargs["json"]["msgtype"], "text")
        self.assertIn("AC-Prof 采集完成", kwargs["json"]["text"]["content"])

    def test_send_retries_then_succeeds(self) -> None:
        response = SimpleNamespace(status_code=200, json=lambda: {"errcode": 0})
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=2,
            retry_delay_seconds=0.5,
        )

        with patch(
            "acprof.notifications.requests.post",
            side_effect=[requests.ConnectionError("offline"), response],
        ) as post, patch("acprof.notifications.time.sleep") as sleep:
            notifier.send(self._event())

        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_request_failure_never_exposes_webhook(self) -> None:
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=1,
        )
        unsafe_error = requests.ConnectionError(f"failed to reach {WEBHOOK_URL}")

        with patch(
            "acprof.notifications.requests.post",
            side_effect=unsafe_error,
        ), self.assertRaises(NotificationDeliveryError) as raised:
            notifier.send(self._event())

        self.assertNotIn(WEBHOOK_KEY, str(raised.exception))
        self.assertIn("ConnectionError", str(raised.exception))

    def test_api_error_is_reported_without_request_url(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"errcode": 93000, "errmsg": "invalid webhook"},
        )
        notifier = WeComWebhookNotifier(WEBHOOK_URL, attempts=1)

        with patch(
            "acprof.notifications.requests.post",
            return_value=response,
        ), self.assertRaises(NotificationDeliveryError) as raised:
            notifier.send(self._event())

        self.assertIn("errcode=93000", str(raised.exception))
        self.assertNotIn(WEBHOOK_KEY, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
