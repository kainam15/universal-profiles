"""镜像管理应提供可操作列表，并与采集和其它 Docker 操作互斥。"""

from pathlib import Path
from dataclasses import replace
import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.widgets import Button, Collapsible, ContentSwitcher, DataTable, Input, Select, Static, TabbedContent, TabPane, Tree

from acprof.tui.app import AcprofTui, PendingLaunch
from acprof.tui.commands import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.host.image_management import ImageManagementError, ManagedImage, list_images
from acprof.tui.images import filtered_images, image_metadata, image_display_name, layer_image_detail
from test_image_management import DockerFixture, FINAL, RUNTIME, WEIGHTS, dependency_images, image
from test_tui_table_resize import drag, header_offset


class ImageDisplayNameTests(unittest.TestCase):
    def test_platform_is_omitted_only_when_parent_provides_matching_context(self):
        parent = ManagedImage(RUNTIME, ("acprof-platform-cu124:base",), 100, "", "base", platform_id="cu124")
        child = ManagedImage(WEIGHTS, ("acprof-runtime-env:opaque",), 400, "", "runtime",
                             platform_id="cu124", environment_id="env-known", profiles=("nlp-cu124",))
        self.assertEqual(image_display_name(parent), "CUDA 12.4")
        self.assertEqual(image_display_name(child, parent), "nlp")
        for context in (None, replace(parent, platform_id="cu128"), replace(parent, kind="model")):
            self.assertEqual(image_display_name(child, context), "nlp · CUDA 12.4")
        self.assertEqual(image_display_name(replace(child, profiles=())), "env-env-known · CUDA 12.4")
        self.assertEqual(image_display_name(replace(parent, platform_id="cpu")), "CPU")
        self.assertEqual(child.profiles, ("nlp-cu124",))
        self.assertEqual(child.tags, ("acprof-runtime-env:opaque",))

    def test_dotted_versions_and_model_names_are_preserved_without_guessing(self):
        item = ManagedImage(WEIGHTS, (), 400, "", "runtime", platform_id="cpu", environment_id="known",
                            profiles=("audio-cpu", "multimodal-transformers4576-cpu", "custom-v1.12.3-cpu", "custom123-cpu"))
        self.assertEqual(image_display_name(item),
                         "audio / multimodal-transformers4.57.6 / custom-v1.12.3 / custom123 · CPU")
        self.assertEqual(image_display_name(replace(item, kind="model", model_id="Qwen/Qwen2.5-0.5B")),
                         "Qwen/Qwen2.5-0.5B")


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
        # 定时刷新单独验证；其它交互测试不依赖机器运行速度。
        interval = patch.object(AcprofTui, "IMAGE_REFRESH_INTERVAL", 3600)
        interval.start()
        self.addCleanup(interval.stop)

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def test_images_page_loads_automatically_only_after_opening(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            self.assertIn("images-tab", [pane.id for pane in tabs.query(TabPane)],
                          "用户应能从 TUI 进入 Docker 镜像管理")
            self.assertFalse(self.docker.commands)
            field = app.query_one("#slash-command", Input)
            field.value = "/images"
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(tabs.active, "images-tab")
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 3)
            self.assertFalse(app.query("#image-refresh"))
            self.assertTrue(app.query_one("#image-delete", Button).disabled)
            self.assertTrue(self.docker.commands)

    async def test_detail_summary_separates_packages_and_diagnostics_with_interactive_folds(self):
        dependency_images(self.docker, profile="nlp-cu128")
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="list")
            inventory = app._image_inventory
            packages = (("transformers", "4.57.6"), *(
                (f"example-package-{index:02}", "1.0") for index in range(43)))
            app._show_images(replace(inventory, images=tuple(
                replace(item, python_dependencies=packages) if item.image_id == WEIGHTS else item
                for item in inventory.images)))
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=table.get_row_index(WEIGHTS), animate=False)
            await pilot.pause()
            summary = str(app.query_one("#image-detail", Static).content)
            self.assertNotIn("sha256:", summary, "完整 SHA 应只出现在默认折叠的诊断信息中")
            self.assertNotIn("transformers==", summary, "包清单不应挤占摘要")
            self.assertNotIn("history", summary)
            self.assertIn("完整大小", summary)
            self.assertIn("删除预计释放", summary)
            dependencies = app.query_one("#image-dependencies", Collapsible)
            metadata = app.query_one("#image-metadata", Collapsible)
            diagnostics = app.query_one("#image-diagnostics", Collapsible)
            self.assertIn("44", dependencies.title)
            before = len(self.docker.commands)
            app.query_one("#image-detail-scroll").focus()
            await pilot.press("tab")
            await pilot.pause()
            self.assertIs(app.focused, dependencies.query_one("CollapsibleTitle"))
            for group, content_id in ((dependencies, "image-dependency-detail"),
                                       (metadata, "image-metadata-detail"),
                                       (diagnostics, "image-diagnostic-detail")):
                self.assertTrue(group.collapsed)
                content = app.query_one("#" + content_id, Static)
                self.assertEqual(content.region.height, 0)
                title = group.query_one("CollapsibleTitle")
                title.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                self.assertTrue(await pilot.click(title))
                await pilot.pause()
                self.assertFalse(group.collapsed)
                self.assertGreater(content.region.height, 0)
                if group is dependencies:
                    self.assertIn("transformers==4.57.6", str(content.content))
                    self.assertIn("example-package-42==1.0", str(content.content))
                    self.assertNotIn("依赖来源", str(content.content))
                    scroll = app.query_one("#image-detail-scroll")
                    scroll.focus()
                    scroll.scroll_end(animate=False, immediate=True)
                    await pilot.pause()
                    screen_text = ""
                    for _ in range(6):
                        screen_text += "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                        await pilot.press("up")
                        await pilot.pause()
                    self.assertIn("example-package-42==1.0", screen_text)
                elif group is diagnostics:
                    self.assertIn(WEIGHTS, str(content.content))
                    self.assertIn("依赖来源", str(content.content))
                    self.assertIn("history", str(content.content))
                title.focus()
                await pilot.press("enter")
                await pilot.pause()
                self.assertTrue(group.collapsed)
            self.assertEqual(len(self.docker.commands), before, "折叠交互不能扫描 Docker")

    async def test_detail_folds_preserve_state_on_resize_and_language_but_reset_for_another_image(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            self.assertNotIn("sha256:", str(app.query_one("#image-detail", Static).content))
            diagnostics = app.query_one("#image-diagnostics", Collapsible)
            title = diagnostics.query_one("CollapsibleTitle")
            title.focus()
            await pilot.press("enter")
            await pilot.pause()
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    self.assertFalse(diagnostics.collapsed)
                    self.assertEqual(diagnostics.title, "诊断信息" if language == "zh" else "Diagnostics")
                    title.focus()
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertTrue(diagnostics.collapsed)
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertFalse(diagnostics.collapsed)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.press("space")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {FINAL})
            self.assertFalse(diagnostics.collapsed, "勾选同一镜像时应保留展开状态")
            table.move_cursor(row=table.get_row_index(WEIGHTS), animate=False)
            await pilot.pause()
            self.assertTrue(diagnostics.collapsed)
            self.assertEqual(app.query_one("#image-detail-scroll").scroll_y, 0)
            self.assertIn(WEIGHTS, str(app.query_one("#image-diagnostic-detail", Static).content))
            await pilot.click("#image-view-layers")
            await pilot.pause()
            self.assertFalse(app.query_one("#image-dependencies").display)
            self.assertTrue(diagnostics.collapsed)
            self.assertIn("Chain ID", str(app.query_one("#image-diagnostic-detail", Static).content))
            app.query_one("#image-search", Input).value = "no-such-image"
            await pilot.pause()
            for selector in ("#image-dependencies", "#image-metadata", "#image-diagnostics"):
                self.assertFalse(app.query_one(selector).display)
            self.assertNotIn("sha256:", str(app.query_one("#image-detail", Static).content))

    async def test_tree_dependencies_stay_in_details_and_search_after_resize_and_language(self):
        dependency_images(self.docker, profile="nlp-cu128")
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            before = len(self.docker.commands)
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    with self.subTest(size=size, language=language):
                        app.ui_preferences = replace(app.ui_preferences, language=language)
                        app._apply_ui_preferences()
                        await pilot.pause()
                        tree = app.query_one("#image-tree", Tree)
                        platform = tree.root.children[0]
                        runtime = platform.children[0]
                        self.assertNotIn("torch==", tree.render_line(platform.line).text)
                        self.assertNotIn("transformers==", tree.render_line(runtime.line).text)
                        tree.move_cursor(platform, animate=False)
                        await pilot.pause()
                        self.assertIn("torch==2.11.0+cu128", str(app.query_one("#image-dependency-detail", Static).content))
                        self.assertTrue(app.query_one("#image-dependencies", Collapsible).collapsed)
                        tree.move_cursor(runtime, animate=False)
                        await pilot.pause()
                        detail = str(app.query_one("#image-dependency-detail", Static).content)
                        self.assertIn("transformers==4.57.6", detail)
                        self.assertIn("sentence-transformers==5.1.2", detail)
                        self.assertNotIn("torch==", detail)
                        self.assertNotIn("sha256:", detail)
                        self.assertNotIn("依赖来源", detail)
                        panel = app.query_one("#image-detail-scroll")
                        for group in panel.query(Collapsible):
                            title = group.query_one("CollapsibleTitle")
                            self.assertLessEqual(title.region.bottom, panel.region.bottom)
                            self.assertIs(app.screen.get_widget_at(*title.region.offset)[0], title)
                        weights = runtime.children[0]
                        tree.move_cursor(weights, animate=False)
                        await pilot.pause()
                        self.assertNotIn("无新增包", tree.render_line(weights.line).text)
                        self.assertNotIn("No new packages", tree.render_line(weights.line).text)
                        detail = str(app.query_one("#image-dependency-detail", Static).content)
                        self.assertIn("无新增包" if language == "zh" else "No new packages", detail)
                        self.assertNotIn("transformers==", detail)
            app.query_one("#image-search", Input).value = "transformers==4.57.6"
            await pilot.pause()
            self.assertEqual([item.image_id for item in app._visible_images], [WEIGHTS])
            tree = app.query_one("#image-tree", Tree)
            self.assertEqual(tree.root.children[0].data.image_id, RUNTIME)
            self.assertEqual(tree.root.children[0].children[0].data.image_id, WEIGHTS)
            self.assertEqual(len(self.docker.commands), before, "切换与搜索不能重新扫描 Docker")

    async def load_images(self, app, pilot, *, view="list"):
        await pilot.pause()
        app.action_show_images()
        await pilot.pause()
        await app.workers.wait_for_complete()
        app._image_refresh_timer.pause()
        await pilot.pause()
        self.assertFalse(app._is_busy())
        self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "images-tab")
        if view == "list" and app.query("#image-view-list"):
            await pilot.click("#image-view-list")
            table = app.query_one("#image-table", DataTable)
            if FINAL in table.rows:
                table.move_cursor(row=table.get_row_index(FINAL))
            await pilot.pause()

    def marker_offset(self, widget, row):
        # 从实际渲染字符定位，避免用控件的点击元数据验证自身。
        text = widget.render_line(row).text
        index = next(index for index, char in enumerate(text) if char in "□☑✓—")
        return cell_len(text[:index]), row

    def add_tree_branches(self):
        for key, name, layers in (
            ("d", "alpha/model", ["os", "deps", "alpha"]),
            ("e", "alpha/model", ["os", "deps", "alpha", "code"]),
            ("f", "zulu/model", ["os", "deps", "zulu"]),
        ):
            image_id = "sha256:" + key * 64
            self.docker.images[image_id] = image(
                image_id, [f"acprof-weights-audio-{key}:test"], 400, layers, name)

    def assert_tree_path(self, app, tree, color, expected):
        # 检查最终屏幕的线条颜色，也能发现移动光标后祖先行没有重绘的问题。
        screen = app.screen._compositor.render_strips()
        region = tree.scrollable_content_region
        scroll_x, scroll_y = tree.scroll_offset
        highlighted = set()
        for y in range(region.height):
            x = 0
            for segment in screen[region.y + y].crop(region.x, region.right):
                for char in segment.text:
                    if char in "│├└─┃┣┗━" and segment.style.color == color:
                        highlighted.add((x + scroll_x, y + scroll_y))
                    x += cell_len(char)
        visible_expected = {(x, y) for x, y in expected
                            if scroll_x <= x < scroll_x + region.width
                            and scroll_y <= y < scroll_y + region.height}
        self.assertEqual(highlighted, visible_expected)

    async def test_tree_path_follows_click_keyboard_and_collapse_without_highlighting_siblings(self):
        self.add_tree_branches()
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            before = len(self.docker.commands)
            await pilot.click(tree, offset=(14, 4))
            await pilot.pause()
            self.assertEqual(tree.cursor_node.data.image_id, FINAL)
            color = tree.get_component_rich_style("tree--cursor").bgcolor
            final_path = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (4, 4), (5, 4), (6, 4)}
            self.assert_tree_path(app, tree, color, final_path)
            await pilot.hover(tree, offset=(14, 1))
            await pilot.pause()
            self.assert_tree_path(app, tree, color, final_path)
            await pilot.press("down")
            await pilot.pause()
            self.assert_tree_path(app, tree, color, {(0, y) for y in range(1, 6)} | {(1, 5), (2, 5)})
            await pilot.press("up", "left")
            await pilot.pause()
            self.assertEqual(tree.cursor_node.data.image_id, WEIGHTS)
            parent_path = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
            self.assert_tree_path(app, tree, color, parent_path)
            await pilot.press("left")
            await pilot.pause()
            self.assertFalse(tree.cursor_node.is_expanded)
            self.assert_tree_path(app, tree, color, parent_path)
            await pilot.press("left")
            await pilot.pause()
            self.assert_tree_path(app, tree, color, set())
            self.assertEqual(app._selected_image_ids, set())
            self.assertEqual(len(self.docker.commands), before)

    async def test_tree_path_survives_language_resize_scroll_blur_and_search(self):
        self.add_tree_branches()
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            before = len(self.docker.commands)
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language, theme in (("zh", "acprof-dark"), ("en", "acprof-light")):
                    with self.subTest(size=size, language=language, theme=theme):
                        app.ui_preferences = replace(app.ui_preferences, language=language, theme=theme)
                        app._apply_ui_preferences()
                        await pilot.pause()
                        tree = app.query_one("#image-tree", Tree)
                        target = tree.root.children[0].children[1].children[0]
                        tree.move_cursor(target, animate=False)
                        tree.focus()
                        await pilot.pause()
                        color = tree.get_component_rich_style("tree--cursor").bgcolor
                        expected = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (4, 4), (5, 4), (6, 4)}
                        self.assert_tree_path(app, tree, color, expected)
                        app.query_one("#image-search", Input).focus()
                        await pilot.pause()
                        self.assert_tree_path(app, tree, color, expected)
            tree.styles.height = 3
            target.set_label(target.label.copy().append(" extra" * 30))
            await pilot.pause()
            tree.scroll_to(x=2, y=2, animate=False, force=True)
            await pilot.pause()
            self.assertEqual(tuple(tree.scroll_offset), (2, 2))
            self.assert_tree_path(app, tree, color, expected)
            tree.styles.height = "1fr"
            search = app.query_one("#image-search", Input)
            search.value = "code"
            await pilot.pause()
            self.assert_tree_path(app, tree, color, {(0, 1), (1, 1), (2, 1), (4, 2), (5, 2), (6, 2)})
            search.value = "no-such-image"
            await pilot.pause()
            self.assert_tree_path(app, tree, color, set())
            self.assertEqual(len(self.docker.commands), before)

    async def test_image_row_clicks_only_focus_and_show_details(self):
        for view in ("tree", "list"):
            with self.subTest(view=view):
                app = self.make_app()
                async with app.run_test(size=(120, 30)) as pilot:
                    await self.load_images(app, pilot, view=view)
                    widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
                    before = len(self.docker.commands)
                    for image_id in (RUNTIME, FINAL):
                        row = (0 if image_id == RUNTIME else 2) if view == "tree" else widget.get_row_index(image_id) + 1
                        marker_x, _ = self.marker_offset(widget, row)
                        # 名称、列内空白和数值重复点击也不能勾选或取消。
                        for x in (marker_x + 3, marker_x + 3, marker_x + 30, widget.size.width - 20):
                            await pilot.click(widget, offset=(x, row))
                            await pilot.pause()
                            self.assertEqual(app._current_image().image_id, image_id)
                            self.assertIn(image_id, str(app.query_one("#image-diagnostic-detail", Static).content))
                            self.assertEqual(app._selected_image_ids, set())
                    widget.focus()
                    await pilot.press("space")
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL})
                    marker_x, row = self.marker_offset(widget, row)
                    await pilot.click(widget, offset=(marker_x + 3, row), times=2)
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL})
                    self.assertEqual(len(self.docker.commands), before)

    async def test_checkbox_and_adjacent_padding_toggle_once_after_language_and_resize(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            before = len(self.docker.commands)
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    for view in ("tree", "list"):
                        with self.subTest(size=size, language=language, view=view):
                            await pilot.click("#image-view-" + view)
                            await pilot.pause()
                            widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
                            for image_id in (RUNTIME, FINAL):
                                if view == "tree":
                                    node = widget.root.children[0]
                                    if image_id == FINAL:
                                        node = node.children[0].children[0]
                                    widget.move_cursor(node, animate=False)
                                else:
                                    widget.move_cursor(row=widget.get_row_index(image_id), animate=False)
                                await pilot.pause()
                                row = (node.line if view == "tree" else widget.cursor_row + 1) - int(widget.scroll_y)
                                for delta in (-1, 0, 1):
                                    for selected in (True, False):
                                        marker_x, _ = self.marker_offset(widget, row)
                                        self.assertTrue(await pilot.click(widget, offset=(marker_x + delta, row)))
                                        await pilot.pause()
                                        self.assertEqual(app._selected_image_ids, {image_id} if selected else set())
                                        self.assertIn("☑" if selected else "□", widget.render_line(row).text)
                                        if view == "tree":
                                            self.assertTrue(widget.root.children[0].is_expanded,
                                                            "勾选父镜像不能同时折叠子树")
            self.assertEqual(len(self.docker.commands), before)

    async def test_tree_arrow_clicks_only_fold_and_checkbox_click_selects_new_row(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            for expanded in (False, True):
                await pilot.click(tree, offset=(0, 0))
                await pilot.pause()
                self.assertEqual(tree.root.children[0].is_expanded, expanded)
                self.assertEqual(app._selected_image_ids, set())
            for view in ("tree", "list"):
                await pilot.click("#image-view-" + view)
                await pilot.pause()
                widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
                for image_id in (FINAL, RUNTIME):
                    row = (2 if image_id == FINAL else 0) if view == "tree" else widget.get_row_index(image_id) + 1
                    await pilot.click(widget, offset=self.marker_offset(widget, row))
                    await pilot.pause()
                    self.assertEqual(app._current_image().image_id, image_id)
                    self.assertEqual(app._selected_image_ids, {FINAL} if image_id == FINAL else {FINAL, RUNTIME})
                await pilot.click("#image-clear")
                await pilot.pause()

    async def test_default_tree_and_layer_view_preserve_image_selection_without_queries(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree")
            self.assertTrue(app.query("#image-tree"), "默认视图应展示可折叠的真实镜像树")
            tree = app.query_one("#image-tree", Tree)
            self.assertEqual(app.query_one("#image-browser", ContentSwitcher).current, "image-tree-view")
            runtime = tree.root.children[0]
            self.assertEqual(runtime.data.image_id, RUNTIME)
            self.assertEqual(runtime.children[0].children[0].data.image_id, FINAL)
            tree.move_cursor(runtime)
            tree.focus()
            await pilot.pause()
            await pilot.press("left")
            await pilot.pause()
            self.assertFalse(runtime.is_expanded)
            await pilot.press("right", "down", "down", "space")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {FINAL})
            detail = str(app.query_one("#image-detail", Static).content)
            self.assertIn("继承路径", str(app.query_one("#image-metadata-detail", Static).content))
            self.assertIn("10 B", detail)
            before = len(self.docker.commands)
            await pilot.click("#image-view-layers")
            await pilot.pause()
            layers = app.query_one("#image-layer-table", DataTable)
            self.assertEqual(layers.row_count, 4)
            self.assertIn("3", str(layers.get_row_at(0)))
            self.assertTrue(app.query_one("#image-toggle", Button).disabled)
            self.assertIn("acprof-runtime-audio", str(app.query_one("#image-metadata-detail", Static).content))
            await pilot.click("#image-view-list")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {FINAL})
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=table.get_row_index(WEIGHTS))
            await pilot.pause()
            await pilot.click("#image-view-tree")
            await pilot.pause()
            self.assertEqual(tree.cursor_node.data.image_id, WEIGHTS)
            self.assertEqual(len(self.docker.commands), before)

    async def test_tree_search_keeps_ancestors_and_language_resize_keeps_collapse(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            self.assertTrue(app.query("#image-tree"), "搜索镜像时应保留祖先路径")
            tree = app.query_one("#image-tree", Tree)
            app.query_one("#image-search", Input).value = "code"
            await pilot.pause()
            self.assertEqual(len(app._visible_images), 1)
            self.assertEqual(tree.root.children[0].data.image_id, RUNTIME)
            self.assertEqual(tree.root.children[0].children[0].children[0].data.image_id, FINAL)
            tree.root.children[0].collapse()
            for size in ((80, 24), (120, 30)):
                await pilot.resize_terminal(*size)
                app.ui_preferences = replace(app.ui_preferences, language="en")
                app._apply_ui_preferences()
                await pilot.pause()
                self.assertFalse(tree.root.children[0].is_expanded)
                for selector in ("#image-view-tree", "#image-view-list", "#image-view-layers", "#image-delete"):
                    button = app.query_one(selector, Button)
                    self.assertGreater(button.region.height, 0)
                    self.assertLessEqual(button.region.right, size[0], selector)

    async def test_tree_platform_names_and_dotted_versions_survive_language_and_resize(self):
        from acprof.runtime_profiles import ENVIRONMENTS, environment_id

        root = Path(__file__).resolve().parents[1]
        self.docker.images[RUNTIME] = image(
            RUNTIME, ["acprof-platform-cu128:base"], 100, ["os", "deps"],
            labels={"org.acprof.image-kind": "platform", "org.acprof.platform": "cu128"})
        for image_id, profile, layer in ((WEIGHTS, "moss-transformers560", "weights"),
                                         (FINAL, "multimodal-transformers4576", "code")):
            self.docker.images[image_id] = image(
                image_id, ["acprof-runtime-env:" + profile], 400, ["os", "deps", layer],
                labels={"org.acprof.image-kind": "environment", "org.acprof.platform": "cu128",
                        "org.acprof.environment": environment_id(ENVIRONMENTS[profile], root)})
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            before = len(self.docker.commands)
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    with self.subTest(size=size, language=language):
                        app.ui_preferences = replace(app.ui_preferences, language=language)
                        app._apply_ui_preferences()
                        await pilot.pause()
                        tree = app.query_one("#image-tree", Tree)
                        platform = tree.root.children[0]
                        self.assertEqual(platform.label.plain.split("  ")[1], "CUDA 12.8")
                        children = {node.data.image_id: node for node in platform.children}
                        for image_id, expected in ((WEIGHTS, "moss-transformers5.6.0"),
                                                   (FINAL, "multimodal-transformers4.57.6")):
                            node = children[image_id]
                            self.assertIn(expected, node.label.plain)
                            self.assertNotIn("cu128", node.label.plain)
                            self.assertNotIn("CUDA", node.label.plain)
                            tree.move_cursor(node, animate=False)
                            await pilot.pause()
                            row = node.line - int(tree.scroll_y)
                            self.assertIn(expected, tree.render_line(row).text)
            inventory = app._image_inventory
            runtime = next(item for item in inventory.images if item.image_id == WEIGHTS)
            self.assertIn("CUDA 12.8 › moss-transformers5.6.0", str(image_metadata(runtime, inventory)))
            self.assertIn("moss-transformers5.6.0 · CUDA 12.8", str(layer_image_detail(inventory.layers[0], inventory)))
            self.assertEqual({item.image_id for item in filtered_images(inventory, "CUDA 12.8", "all")},
                             {RUNTIME, WEIGHTS, FINAL})
            self.assertEqual([item.image_id for item in filtered_images(inventory, "5.6.0", "all")], [WEIGHTS])
            self.assertEqual([item.image_id for item in filtered_images(inventory, "moss-transformers560", "all")], [WEIGHTS])
            await pilot.click("#image-view-list")
            await pilot.pause()
            table = app.query_one("#image-table", DataTable)
            self.assertIn("moss-transformers5.6.0 · CUDA 12.8", table.get_cell(WEIGHTS, "name").plain)
            self.assertEqual(table.get_cell(WEIGHTS, "parent").plain, "CUDA 12.8")
            self.assertEqual(table.get_cell(WEIGHTS, "tag").plain, "moss-transformers560")
            self.assertEqual(len(self.docker.commands), before)

    async def test_list_header_sorts_numeric_bytes_and_preserves_selection(self):
        for image_id, size in ((RUNTIME, 90), (WEIGHTS, 100), (FINAL, 1000)):
            self.docker.images[image_id]["Size"] = size
        self.docker.layer_sizes.update(deps=70, weights=10, code=900)
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            before = len(self.docker.commands)
            offset = sum(column.get_render_width(table) for column in table.ordered_columns[:2]) + 1
            for expected in ((RUNTIME, WEIGHTS, FINAL), (FINAL, WEIGHTS, RUNTIME)):
                self.assertTrue(await pilot.click("#image-table", offset=(offset, 0)))
                await pilot.pause()
                self.assertEqual(tuple(item.image_id for item in app._visible_images), expected)
                self.assertEqual(app._current_image().image_id, FINAL)
                self.assertEqual(app._selected_image_ids, {FINAL})
            self.assertEqual(len(self.docker.commands), before)

    async def test_header_drag_changes_width_without_sorting_or_selecting(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            before = table.columns["name"].width
            order = tuple(item.image_id for item in app._visible_images)
            focused = app._current_image().image_id
            commands = len(self.docker.commands)
            boundary = sum(column.get_render_width(table) for column in table.ordered_columns[:2]) - 1
            await pilot.mouse_down(table, offset=(boundary, 0))
            await pilot.hover(table, offset=(boundary - 12, 0))
            await pilot.mouse_up(table, offset=(boundary - 12, 0))
            await pilot.pause()
            self.assertEqual(table.columns["name"].width, before - 12)
            self.assertEqual(tuple(item.image_id for item in app._visible_images), order)
            self.assertEqual(app._current_image().image_id, focused)
            self.assertFalse(app._selected_image_ids)
            self.assertIsNone(app.mouse_captured)
            self.assertEqual(len(self.docker.commands), commands)

    async def test_manual_widths_survive_sort_filter_refresh_language_and_resize(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            await drag(pilot, table, header_offset(table, 1), -12, dy=1, release_click=True)
            self.assertEqual(table.columns["name"].width, 64)
            self.assertFalse(app._selected_image_ids, "拖到数据行松手不能勾选镜像")
            table.focus()
            await pilot.press("space")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {FINAL})
            self.assertEqual(table.columns["name"].width, 64)
            await pilot.click(table, offset=(header_offset(table, 1)[0] + 3, 0))
            self.assertEqual(app._image_sort[0], "size")
            self.assertEqual(table.columns["name"].width, 64)
            app.query_one("#image-search", Input).value = "demo/model"
            await pilot.pause()
            self.assertEqual(table.row_count, 2)
            self.assertEqual(table.columns["name"].width, 64)
            app.query_one("#image-search", Input).value = ""
            await pilot.pause()
            await pilot.click("#image-view-layers")
            layers = app.query_one("#image-layer-table", DataTable)
            await drag(pilot, layers, header_offset(layers), -5)
            self.assertEqual(layers.columns["diff"].width, 18)
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertEqual(table.columns["name"].width, 64)
            self.assertEqual(layers.columns["diff"].width, 18)
            await pilot.click("#image-view-tree")
            await pilot.click("#image-view-list")
            self.assertEqual(app._selected_image_ids, {FINAL})
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(table.columns["name"].width, 64)
            self.assertEqual(layers.columns["diff"].width, 18)
            self.assertEqual(len(table.columns), 9)
            self.assertEqual(len(layers.columns), 4)
            self.assertEqual(app._selected_image_ids, {FINAL}, "自动刷新保留有效勾选")
        restarted = self.make_app()
        async with restarted.run_test(size=(120, 30)) as pilot:
            await self.load_images(restarted, pilot)
            self.assertEqual(restarted.query_one("#image-table", DataTable).columns["name"].width, 76)

    async def test_header_drags_in_both_languages_and_all_terminal_sizes(self):
        for size in ((80, 24), (120, 30), (150, 45)):
            app = self.make_app()
            async with app.run_test(size=size) as pilot:
                await self.load_images(app, pilot)
                commands = len(self.docker.commands)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    for view, selector, key, index in (("list", "#image-table", "name", 1),
                                                       ("layers", "#image-layer-table", "diff", 0)):
                        with self.subTest(size=size, language=language, view=view):
                            self.assertTrue(await pilot.click("#image-view-" + view))
                            table = app.query_one(selector, DataTable)
                            width = table.columns[key].width
                            start = header_offset(table, index)
                            await drag(pilot, table, start, -5)
                            self.assertEqual(table.columns[key].width, width - 5)
                            self.assertEqual(header_offset(table, index)[0], start[0] - 5)
                            self.assertEqual(table.row_count, 3 if view == "list" else 4)
                            self.assertIsNone(app.mouse_captured)
                            if view == "list":
                                checkbox_edge = header_offset(table)
                                name_edge = header_offset(table, 1)
                                row_before = table.render_line(1)
                                table.scroll_to(x=6, animate=False, force=True)
                                await pilot.pause()
                                self.assertEqual(header_offset(table), checkbox_edge)
                                self.assertEqual(header_offset(table, 1)[0], name_edge[0] - 6,
                                                 "环境 / 模型列应随数据横向滚动")
                                self.assertEqual(table.render_line(1).crop(0, 3).text,
                                                 row_before.crop(0, 3).text)
                                self.assertEqual(table.render_line(1).crop(3, 23).text,
                                                 row_before.crop(9, 29).text)
                                table.scroll_to(x=0, animate=False, force=True)
                                await pilot.pause()
                self.assertEqual(len(self.docker.commands), commands)

    async def test_measurement_interrupts_drag_and_unlocks_afterwards(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            start = header_offset(table, 1)
            await pilot.mouse_down(table, offset=start)
            self.assertIs(app.mouse_captured, table)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            await pilot.pause()
            self.assertIsNone(app.mouse_captured)
            await pilot.hover(table, offset=(start[0] - 6, 0))
            await pilot.mouse_up(table, offset=(start[0] - 6, 0))
            self.assertEqual(table.columns["name"].width, 76)
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            await pilot.pause()
            await drag(pilot, table, header_offset(table, 1), -6)
            self.assertEqual(table.columns["name"].width, 70)

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
                    self.assertIn("acprof-build-source:", str(app.query_one("#image-metadata-detail", Static).content))
                    self.assertNotIn("HF_TOKEN", str(app.query_one("#image-metadata-detail", Static).content))
                    app.ui_preferences = replace(app.ui_preferences, language="en")
                    app._apply_ui_preferences()
                    await pilot.pause()
                    self.assertEqual(app._selected_image_ids, {FINAL, WEIGHTS})
                    self.assertEqual(app.query_one("#image-search", Input).value, "demo/model")
                    self.assertEqual(app.query_one("#image-delete", Button).label.plain, "Delete")
                    self.assertIn("All tags", str(app.query_one("#image-metadata-detail", Static).content))
                    for selector in ("#image-search", "#image-scope", "#image-toggle", "#image-model",
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

    async def test_version_dots_distinguish_models_in_selection_and_search(self):
        self.docker.images[FINAL] = image(
            FINAL, ["acprof-nlp-qwen--qwen2.5-0.5b:code"], 410,
            ["os", "deps", "weights", "code"], "Qwen/Qwen2.5-0.5B")
        # 没有 MODEL_ID 时，从带点号的标签识别同一模型。
        self.docker.images[WEIGHTS] = image(
            WEIGHTS, ["acprof-weights-nlp-qwen--qwen2.5-0.5b:weights"], 400,
            ["os", "deps", "weights"])
        other = "sha256:" + "d" * 64
        self.docker.images[other] = image(
            other, ["acprof-nlp-qwen--qwen2_5-0_5b:code"], 410,
            ["os", "deps", "other-weights", "code"], "Qwen/Qwen2_5-0_5B")
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            self.assertEqual(table.get_cell(FINAL, "repository").plain, "acprof-nlp-qwen--qwen2.5-0.5b")
            table.move_cursor(row=table.get_row_index(FINAL))
            await pilot.pause()
            self.assertTrue(await pilot.click("#image-model"))
            await pilot.pause()
            with self.subTest(action="选择同模型"):
                self.assertEqual(app._selected_image_ids, {FINAL, WEIGHTS})
            search = app.query_one("#image-search", Input)
            for query, expected in (("Qwen/Qwen2.5-0.5B", {FINAL, WEIGHTS}),
                                    ("qwen--qwen2_5-0_5b", {other})):
                with self.subTest(query=query):
                    search.focus()
                    await pilot.press("ctrl+a", *query)
                    await pilot.pause()
                    self.assertEqual({item.image_id for item in app._visible_images}, expected)

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

    async def test_read_failure_keeps_inventory_and_selection_until_refresh_recovers(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            await pilot.pause()
            with patch("acprof.tui.app.list_images", side_effect=ImageManagementError("无法执行 Docker", "permission denied")):
                app.refresh_images()
                await app.workers.wait_for_complete()
                await pilot.pause()
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 3)
            self.assertEqual(app._selected_image_ids, {FINAL})
            self.assertIn("permission denied", str(app.query_one("#image-status", Static).content))
            self.assertTrue(app.query_one("#image-delete", Button).disabled)
            self.assertFalse(app._is_busy())
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 3)
            self.assertNotIn("permission denied", str(app.query_one("#image-status", Static).content))
            self.assertFalse(app.query_one("#image-delete", Button).disabled)

    async def test_timer_updates_visible_inventory_without_user_action(self):
        app = self.make_app()
        app.IMAGE_REFRESH_INTERVAL = 0.1
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            refreshed = asyncio.Event()
            loop = asyncio.get_running_loop()
            extra = "sha256:" + "d" * 64
            self.docker.images[extra] = image(extra, ["acprof-runtime-new:env"], 120, ["new"])

            def read():
                inventory = list_images()
                loop.call_soon_threadsafe(refreshed.set)
                return inventory

            with patch("acprof.tui.app.list_images", side_effect=read):
                app._image_refresh_timer.reset()
                await asyncio.wait_for(refreshed.wait(), timeout=3)
                await app.workers.wait_for_complete()
                app._image_refresh_timer.pause()
                await pilot.pause()
            self.assertEqual(app.query_one("#image-table", DataTable).row_count, 4)
            self.assertIn(extra, app.query_one("#image-table", DataTable).rows)

    async def test_background_page_and_confirmation_do_not_scan(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            app.action_show_settings()
            await pilot.pause()
            with patch("acprof.tui.app.list_images") as read:
                app.refresh_images()
                await pilot.pause()
                read.assert_not_called()
                self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "settings-tab")
            app.action_show_images()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            with patch("acprof.tui.app.list_images") as read:
                app.refresh_images()
                await pilot.pause()
                read.assert_not_called()
            await pilot.press("escape")
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {FINAL})

    async def test_refresh_keeps_focus_selection_and_scrolled_list(self):
        for number in range(30):
            key = f"sha256:{number:064x}"
            self.docker.images[key] = image(key, [f"acprof-runtime-extra-{number:02}:env"], 200, ["extra"])
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=25)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            selected = set(app._selected_image_ids)
            current = app._current_image().image_id
            table.scroll_to(x=12, y=20, animate=False, force=True)
            await pilot.pause()
            offset = table.scroll_offset
            # 清单在读取期间变化时，搜索/勾选等交互仍可使用。
            self.docker.images[RUNTIME]["Size"] += 1
            with patch.object(app, "_execute_image_refresh") as read:
                app.refresh_images()
                app.refresh_images()
                await pilot.pause()
                read.assert_called_once()
                self.assertIs(app.focused, table)
                self.assertFalse(table.disabled)
                self.assertFalse(app.query_one("#image-search", Input).disabled)
                app._show_images(list_images())
            await pilot.pause()
            self.assertIs(app.focused, table)
            self.assertEqual(app._current_image().image_id, current)
            self.assertEqual(app._selected_image_ids, selected)
            self.assertEqual(table.scroll_offset, offset)

    async def test_refresh_drops_only_selections_with_changed_identity_or_references(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            self.docker.images[FINAL]["RepoTags"].append("acprof-extra:new")
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app._selected_image_ids, {WEIGHTS})
            self.docker.containers["used"] = dict(Image=WEIGHTS, Name="/new-user", State=dict(Status="exited"))
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(app._selected_image_ids)
            app.query_one("#image-table", DataTable).move_cursor(row=0)
            await pilot.click("#image-model")
            self.assertEqual(app._selected_image_ids, {FINAL})
            self.docker.daemon_id = "another-daemon"
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(app._selected_image_ids)

    async def test_refresh_preserves_tree_and_detail_folds(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            tree.focus()
            await pilot.press("left")
            await pilot.pause()
            root = tree.cursor_node.data.image_id
            metadata = app.query_one("#image-metadata", Collapsible)
            metadata.collapsed = False
            tree.scroll_to(x=3, animate=False, force=True)
            await pilot.pause()
            offset = tree.scroll_offset
            self.docker.images[FINAL]["Size"] += 1
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(tree.cursor_node.data.image_id, root)
            self.assertFalse(tree.cursor_node.is_expanded)
            self.assertFalse(metadata.collapsed)
            self.assertEqual(tree.scroll_offset, offset)
            current_node = tree.cursor_node
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIs(tree.cursor_node, current_node, "清单未变时不重建树")

    async def test_auto_refresh_pauses_during_measurement_and_resumes_after_failure(self):
        app = self.make_app()
        app.IMAGE_REFRESH_INTERVAL = 0.1
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app._process_kind = "run"
            app._set_busy(True)
            app._consume_process_line("", ProgressSnapshot(measurement_active=True), False)
            commands = len(self.docker.commands)
            # 等待超过刷新间隔，确认实际计时回调不会读取 Docker。
            await pilot.pause(0.3)
            self.assertEqual(len(self.docker.commands), commands)
            self.assertFalse(app._image_refresh_timer._active.is_set())
            refreshed = asyncio.Event()
            loop = asyncio.get_running_loop()

            def read():
                inventory = list_images()
                loop.call_soon_threadsafe(refreshed.set)
                return inventory

            with patch("acprof.tui.app.list_images", side_effect=read):
                app._process_finished("run", 1, None, "test failure")
                await asyncio.wait_for(refreshed.wait(), timeout=3)
                await app.workers.wait_for_complete()
                app._image_refresh_timer.pause()
                await pilot.pause()
            self.assertFalse(app._latest_snapshot.measurement_active)
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
            self.assertIn("kept-container", str(app.query_one("#image-metadata-detail", Static).content))
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
