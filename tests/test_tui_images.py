"""镜像管理应提供可操作列表，并与采集和其它 Docker 操作互斥。"""

from pathlib import Path
from dataclasses import replace
import os
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent, TabPane

from acprof.tui.app import AcprofTui, PendingLaunch
from acprof.tui.commands import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.host.image_management import ImageManagementError
from test_image_management import DockerFixture, FINAL, RUNTIME, WEIGHTS


class TuiImagesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.docker = DockerFixture()
        docker_patch = patch("acprof.host.image_management.subprocess.run", side_effect=self.docker.run)
        docker_patch.start()
        self.addCleanup(docker_patch.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def test_images_page_is_reachable_without_automatic_docker_queries(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            self.assertIn("images-tab", [pane.id for pane in tabs.query(TabPane)],
                          "用户应能从 TUI 进入 Docker 镜像管理")
            field = app.query_one("#slash-command", Input)
            field.value = "/images"
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(tabs.active, "images-tab")
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 0)
            self.assertFalse(list(app.workers), "打开页面不能自动查询 Docker")
            self.assertFalse(app.query_one("#image-refresh", Button).disabled)
            self.assertTrue(app.query_one("#image-delete", Button).disabled)
            self.assertFalse(self.docker.commands)

    async def load_images(self, app, pilot):
        await pilot.pause()
        app.action_show_images()
        await pilot.pause()
        self.assertTrue(await pilot.click("#image-refresh"))
        await app.workers.wait_for_complete()
        await pilot.pause()
        self.assertFalse(app._is_busy())
        self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "images-tab")

    async def test_filter_selection_details_and_language_work_at_all_sizes(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            with self.subTest(size=size):
                app = self.make_app()
                async with app.run_test(size=size) as pilot:
                    await self.load_images(app, pilot)
                    table = app.query_one("#image-table", DataTable)
                    self.assertEqual(table.row_count, 3)
                    table.focus()
                    await pilot.pause()
                    await pilot.press("space")
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL})
                    self.assertTrue(await pilot.click("#image-model"))
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL, WEIGHTS})
                    self.assertNotIn(RUNTIME, app._selected_image_ids)
                    self.assertEqual(table.row_count, 2)
                    table.focus()
                    await pilot.pause()
                    await pilot.press("down")
                    await pilot.pause()
                    self.assertIn("acprof-build-source:", str(app.query_one("#image-detail", Static).content))
                    self.assertNotIn("HF_TOKEN", str(app.query_one("#image-detail", Static).content))
                    app.ui_preferences = replace(app.ui_preferences, language="en")
                    app._apply_ui_preferences()
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL, WEIGHTS})
                    self.assertEqual(app.query_one("#image-search", Input).value, "demo/model")
                    self.assertEqual(app.query_one("#image-delete", Button).label.plain, "Delete")
                    self.assertIn("All tags", str(app.query_one("#image-detail", Static).content))
                    for selector in ("#image-search", "#image-scope", "#image-refresh", "#image-model",
                                     "#image-clear", "#image-delete", "#image-table"):
                        widget = app.query_one(selector)
                        self.assertGreater(widget.region.height, 0, selector)
                        self.assertGreaterEqual(widget.region.x, 0, selector)
                        self.assertLessEqual(widget.region.right, size[0], selector)
                        self.assertLessEqual(widget.region.bottom, size[1] - 3, selector)
                    # 过滤别名仍命中合并后的同一行，清空按钮清掉筛选外的选择。
                    app.query_one("#image-search", Input).value = "acprof-build-source"
                    await pilot.pause()
                    self.assertEqual(table.row_count, 1)
                    self.assertIn("(1 hidden)", str(app.query_one("#image-status", Static).content))
                    self.assertTrue(await pilot.click("#image-clear"))
                    await pilot.pause()
                    self.assertFalse(app._selected_image_ids)

    async def test_confirmation_cancel_and_delete_all_model_tags_with_buttons_visible(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            self.assertTrue(app._is_busy())
            message = str(app.screen.query_one("#image-confirm-text", Static).content)
            self.assertIn("acprof-build-source:", message)
            self.assertIn("acprof-audio-demo--model:code", message)
            self.assertIn("续采或补采", message)
            dialog = app.screen.query_one("#confirm-dialog")
            self.assertGreater(dialog.region.x, 0, "删除确认应沿用居中的有边框弹窗")
            self.assertLess(dialog.region.width, 80)
            for selector in ("#confirm-no", "#confirm-yes"):
                button = app.screen.query_one(selector, Button)
                self.assertGreater(button.region.height, 0)
                self.assertLessEqual(button.region.bottom, 24)
            await pilot.press("escape")
            await pilot.pause()
            self.assertFalse(self.docker.removals)
            self.assertFalse(app._is_busy())
            self.assertEqual(app._selected_image_ids, {FINAL, WEIGHTS})
            await pilot.click("#image-delete")
            await pilot.pause()
            self.assertTrue(await pilot.click("#confirm-yes"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(list(self.docker.images), [RUNTIME])
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 0)
            self.assertIn("已处理 2", str(app.query_one("#image-status", Static).content))
            self.assertFalse(app._is_busy())
            self.assertFalse(app.query_one("#start-run", Button).disabled)

    async def test_read_failure_clears_old_selection_and_refresh_can_recover(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            await pilot.pause()
            with patch("acprof.tui.app.list_images", side_effect=ImageManagementError("无法执行 Docker", "permission denied")):
                await pilot.click("#image-refresh")
                await app.workers.wait_for_complete()
                await pilot.pause()
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 0)
            self.assertFalse(app._selected_image_ids)
            self.assertIn("permission denied", str(app.query_one("#image-status", Static).content))
            self.assertTrue(app.query_one("#image-delete", Button).disabled)
            self.assertFalse(app._is_busy())
            await self.load_images(app, pilot)
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 3)

    async def test_measurement_and_image_operations_are_mutually_exclusive(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            before = len(self.docker.commands)
            app.refresh_images()
            app.request_delete_images()
            app.select_model_images()
            app.action_show_images()
            await pilot.pause()
            self.assertEqual(len(self.docker.commands), before)
            self.assertTrue(all(widget.disabled for widget in app.query(".image-control")))
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            with patch.object(app, "_execute_image_refresh") as read, patch.object(app, "_execute_command") as execute:
                app.refresh_images()
                self.assertTrue(app._is_busy())
                app._launch(PendingLaunch(("must-not-start",), "run"))
                app.action_quick_check()
                app.action_request_stop()
                execute.assert_not_called()
                read.assert_called_once()
                self.assertTrue(app.query_one("#stop-run", Button).disabled)
                app._show_images(app._image_inventory)
            self.assertFalse(app._is_busy())

    async def test_container_reference_prevents_selecting_that_image(self):
        self.docker.containers["used"] = dict(Image=FINAL, Name="/kept-container", State=dict(Status="exited"))
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            self.assertFalse(app._selected_image_ids)
            self.assertTrue(app.query_one("#image-toggle", Button).disabled)
            self.assertIn("kept-container", str(app.query_one("#image-detail", Static).content))
            await pilot.click("#image-model")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {WEIGHTS})

    async def test_long_confirmation_can_scroll_and_keyboard_cancel_survives_resize(self):
        self.docker.images[WEIGHTS]["RepoTags"].extend(f"acprof-weights-audio-demo--model:old-{index}" for index in range(30))
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            before = set(app._selected_image_ids)
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, before)
            self.assertLessEqual(app.query_one("#image-table", DataTable).columns["repository"].width, 38)
            await pilot.click("#image-delete")
            await pilot.pause()
            content = app.screen.query_one("#image-confirm-content")
            self.assertGreater(content.max_scroll_y, 0)
            content.focus()
            await pilot.pause()
            await pilot.press("end")
            await pilot.pause()
            self.assertGreater(content.scroll_y, 0)
            button = app.screen.query_one("#confirm-no", Button)
            self.assertLessEqual(button.region.bottom, 24)
            button.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(self.docker.removals)
            self.assertFalse(app._is_busy())

    async def test_changed_tags_after_confirmation_are_reported_without_deleting(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            self.docker.images[WEIGHTS]["RepoTags"].append("new-owner:keep")
            app.screen.query_one("#confirm-no", Button).focus()
            await pilot.pause()
            await pilot.press("tab", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(self.docker.removals)
            self.assertIn("标签已改变", str(app.query_one("#image-status", Static).content))
            self.assertFalse(app._selected_image_ids)
            self.assertFalse(app._is_busy())


if __name__ == "__main__":
    unittest.main()
