import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import env_utils
from acprof.config import (
    CONTAINER_HF_HOME,
    CONTAINER_MODEL_LOCAL_PATH,
    HF_MIRROR_ENDPOINT,
)


class BootstrapProjectEnvTests(unittest.TestCase):
    def test_local_env_can_be_loaded_without_changing_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ, {"EXPLICIT_SETTING": "process value"}, clear=True,
        ):
            root = Path(tmp_dir)
            (root / ".env").write_text("EXPLICIT_SETTING=file value\n", encoding="utf-8")
            (root / ".env.local").write_text(
                "ACPROF_SUDO_PASSWORD='test-only-password'\n", encoding="utf-8",
            )
            probe_environ = os.environ.copy()

            env_utils.load_project_env(root, environ=probe_environ)

            self.assertEqual(probe_environ["EXPLICIT_SETTING"], "process value")
            self.assertEqual(probe_environ["ACPROF_SUDO_PASSWORD"], "test-only-password")
            self.assertNotIn("ACPROF_SUDO_PASSWORD", os.environ)

    def test_bootstrap_sets_default_hf_endpoint_and_bypasses_proxy_for_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "acprof.host.env_utils.os.environ",
            {
                "HTTP_PROXY": "http://127.0.0.1:7890",
                "HTTPS_PROXY": "http://127.0.0.1:7890",
                "ALL_PROXY": "socks5h://127.0.0.1:7891",
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
            },
            clear=True,
        ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
            env_utils.bootstrap_project_env(tmp_dir)

            self.assertEqual(env_utils.os.environ["HF_ENDPOINT"], HF_MIRROR_ENDPOINT)
            self.assertEqual(env_utils.os.environ["HF_HUB_ENDPOINT"], HF_MIRROR_ENDPOINT)
            self.assertIn("hf-mirror.com", env_utils.os.environ["NO_PROXY"].split(","))
            self.assertIn("hf-mirror.com", env_utils.os.environ["no_proxy"].split(","))

    def test_bootstrap_preserves_explicit_endpoint_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "acprof.host.env_utils.os.environ",
            {},
            clear=True,
        ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
            Path(tmp_dir, ".env").write_text("HF_ENDPOINT=https://example.invalid\n", encoding="utf-8")

            env_utils.bootstrap_project_env(tmp_dir)

            self.assertEqual(env_utils.os.environ["HF_ENDPOINT"], "https://example.invalid")
            self.assertEqual(env_utils.os.environ["HF_HUB_ENDPOINT"], "https://example.invalid")
            self.assertIn("example.invalid", env_utils.os.environ["NO_PROXY"].split(","))

    def test_bootstrap_replaces_blank_endpoint_env_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "acprof.host.env_utils.os.environ",
            {"HF_ENDPOINT": "", "HF_HUB_ENDPOINT": "   "},
            clear=True,
        ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
            env_utils.bootstrap_project_env(tmp_dir)

            self.assertEqual(env_utils.os.environ["HF_ENDPOINT"], HF_MIRROR_ENDPOINT)
            self.assertEqual(env_utils.os.environ["HF_HUB_ENDPOINT"], HF_MIRROR_ENDPOINT)

    def test_blank_primary_endpoint_uses_configured_fallback(self) -> None:
        for blank in ("", " \t "):
            with self.subTest(blank=repr(blank)), patch.dict(
                os.environ,
                {"HF_ENDPOINT": blank, "HF_HUB_ENDPOINT": "https://example.invalid"},
                clear=True,
            ):
                self.assertEqual(env_utils.configure_hf_network(), "https://example.invalid")
                self.assertEqual(os.environ["HF_ENDPOINT"], "https://example.invalid")
                self.assertEqual(os.environ["HF_HUB_ENDPOINT"], "https://example.invalid")
                self.assertIn("example.invalid", os.environ["NO_PROXY"].split(","))

    def test_explicit_endpoint_retains_priority_and_preserves_nonblank_alias(self) -> None:
        with patch.dict(
            os.environ,
            {"HF_ENDPOINT": "https://primary.invalid", "HF_HUB_ENDPOINT": "https://secondary.invalid"},
            clear=True,
        ):
            self.assertEqual(env_utils.configure_hf_network(), "https://primary.invalid")
            self.assertEqual(os.environ["HF_HUB_ENDPOINT"], "https://secondary.invalid")

    def test_token_fallback_fills_blank_primary_without_reading_login(self) -> None:
        for blank in ("", " \t "):
            with self.subTest(blank=repr(blank)), patch.dict(
                os.environ, {"HF_TOKEN": blank, "HUGGING_FACE_HUB_TOKEN": "test-only-legacy"},
                clear=True,
            ), patch("huggingface_hub.utils.get_token", return_value=None) as get_token:
                self.assertEqual(env_utils.resolve_hf_token(), "test-only-legacy")
                self.assertEqual(os.environ["HF_TOKEN"], "test-only-legacy")
                self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "test-only-legacy")
                get_token.assert_not_called()

    def test_primary_token_fills_blank_legacy_alias(self) -> None:
        for blank in ("", " \t "):
            with self.subTest(blank=repr(blank)), patch.dict(
                os.environ, {"HF_TOKEN": "test-only-primary", "HUGGING_FACE_HUB_TOKEN": blank},
                clear=True,
            ), patch("huggingface_hub.utils.get_token") as get_token:
                self.assertEqual(env_utils.resolve_hf_token(), "test-only-primary")
                self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "test-only-primary")
                get_token.assert_not_called()

    def test_local_login_fills_missing_or_blank_token_aliases(self) -> None:
        for values in ({}, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": " \t "}):
            with self.subTest(values=values), patch.dict(
                os.environ, values, clear=True,
            ), patch("huggingface_hub.utils.get_token", return_value="test-only-local") as get_token:
                self.assertEqual(env_utils.resolve_hf_token(), "test-only-local")
                self.assertEqual(os.environ["HF_TOKEN"], "test-only-local")
                self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "test-only-local")
                get_token.assert_called_once_with()

    def test_primary_token_retains_priority_without_overwriting_legacy_alias(self) -> None:
        with patch.dict(
            os.environ,
            {"HF_TOKEN": "test-only-primary", "HUGGING_FACE_HUB_TOKEN": "test-only-legacy"},
            clear=True,
        ), patch("huggingface_hub.utils.get_token") as get_token:
            self.assertEqual(env_utils.resolve_hf_token(), "test-only-primary")
            self.assertEqual(os.environ["HF_TOKEN"], "test-only-primary")
            self.assertEqual(os.environ["HUGGING_FACE_HUB_TOKEN"], "test-only-legacy")
            get_token.assert_not_called()

    def test_missing_or_unavailable_login_keeps_anonymous_environment(self) -> None:
        for outcome in (None, OSError("test-only unavailable login")):
            with self.subTest(outcome=outcome), patch.dict(os.environ, {}, clear=True), patch(
                "huggingface_hub.utils.get_token",
                **({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}),
            ):
                self.assertIsNone(env_utils.resolve_hf_token())
                self.assertNotIn("HF_TOKEN", os.environ)
                self.assertNotIn("HUGGING_FACE_HUB_TOKEN", os.environ)

    def test_offline_docker_env_disables_hub_and_exposes_local_snapshot(self) -> None:
        args = env_utils.hf_offline_docker_env_args()
        env_values = {
            args[index + 1]
            for index, value in enumerate(args[:-1])
            if value == "-e"
        }

        self.assertIn("HF_HUB_OFFLINE=1", env_values)
        self.assertIn("TRANSFORMERS_OFFLINE=1", env_values)
        self.assertIn(f"HF_HOME={CONTAINER_HF_HOME}", env_values)
        self.assertIn(f"HF_HUB_CACHE={CONTAINER_HF_HOME}", env_values)
        self.assertIn(f"TRANSFORMERS_CACHE={CONTAINER_HF_HOME}", env_values)
        self.assertIn(f"MODEL_LOCAL_PATH={CONTAINER_MODEL_LOCAL_PATH}", env_values)


if __name__ == "__main__":
    unittest.main()
