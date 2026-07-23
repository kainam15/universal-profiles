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
