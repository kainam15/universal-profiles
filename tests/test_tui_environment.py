"""Connection settings use an isolated project and never send notifications."""
import os
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from textual.widgets import Button, Checkbox, Input, Static, TabbedContent

from acprof.host.env_utils import load_project_env
from acprof.tui.app import AcprofTui
from acprof.tui.commands import RunConfig
from acprof.tui.views import ConfirmActionScreen


class TuiEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for context in (patch('acprof.tui.app.PROJECT_DIR', self.root),
                        patch.dict(os.environ, {'PATH': os.environ.get('PATH', '')}, clear=True)):
            context.start()
            self.addCleanup(context.stop)

    async def test_permission_review_can_cancel_and_failed_sudo_returns_to_form(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            await pilot.click('#open-environment-settings')
            await pilot.pause()
            screen = app.screen
            screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
            await pilot.pause()
            with patch('acprof.tui.environment.build_permission_plan') as build, patch(
                'acprof.tui.environment.execute_permission_plan', side_effect=RuntimeError('sudo cancelled'),
            ) as install, patch('textual.app.App.suspend', return_value=nullcontext()):
                # The real plan contains ten commands, including long Ubuntu ELF paths.
                build.return_value.commands = (
                    ('/usr/sbin/setcap', 'cap_perfmon=ep',
                     '/usr/lib/linux-hwe-7.0-tools-7.0.0-28/perf'),
                ) * 10
                await pilot.click('#configure-environment-permissions')
                await pilot.pause()
                confirmation = app.screen
                self.assertIsInstance(confirmation, ConfirmActionScreen)
                assert isinstance(confirmation, ConfirmActionScreen)
                self.assertIn('cap_perfmon', confirmation.message)
                commands = confirmation.query_one('#confirm-message')
                self.assertGreater(commands.max_scroll_y, 0,
                                   'The complete permission plan must be scrollable')
                commands.focus()
                await pilot.press('end')
                await pilot.wait_for_scheduled_animations()
                self.assertGreater(commands.scroll_y, 0)
                install.assert_not_called()
                await pilot.press('escape')
                await pilot.pause()
                install.assert_not_called()
                await pilot.click('#configure-environment-permissions')
                await pilot.pause()
                confirm = app.screen.query_one('#confirm-yes', Button)
                self.assertLessEqual(confirm.region.bottom, 24)
                self.assertIs(app.get_widget_at(*confirm.region.center)[0], confirm)
                await pilot.click('#confirm-yes')
                await pilot.pause()
                install.assert_called_once()
                self.assertIs(app.screen, screen)
                status = screen.query_one('#environment-status', Static).content
                assert isinstance(status, str)
                self.assertIn('sudo cancelled', status)
                self.assertFalse(screen.query_one('#close-environment-settings', Button).disabled)
            await pilot.press('escape')
            await pilot.pause()
            self.assertFalse(app._is_busy())

    async def test_bilingual_dialog_actions_are_reachable_at_supported_sizes(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(150, 45)) as pilot:
            for size in ((80, 24), (120, 30), (150, 45)):
                for language in ('zh', 'en'):
                    with self.subTest(size=size, language=language):
                        await pilot.resize_terminal(*size)
                        app.ui_preferences = replace(app.ui_preferences, language=language)
                        app._apply_ui_preferences()
                        await pilot.press('f2')
                        await pilot.pause()
                        await pilot.click('#open-environment-settings')
                        await pilot.pause()
                        for widget_id in ('close-environment-settings', 'save-environment-settings'):
                            widget = app.screen.query_one('#' + widget_id, Button)
                            self.assertTrue(widget.region.width and widget.region.height)
                            self.assertLessEqual(widget.region.right, size[0])
                            self.assertLessEqual(widget.region.bottom, size[1])
                        app.screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
                        await pilot.pause()
                        # Check actual hit testing rather than just DOM presence.
                        self.assertTrue(await pilot.click('#permission-tcpdump'))
                        self.assertFalse(app.screen.query_one('#permission-tcpdump', Checkbox).value)
                        self.assertTrue(await pilot.click('#close-environment-settings'))
                        await pilot.pause()

    async def test_settings_can_save_masked_credentials_and_reopen_them(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            self.assertEqual(len(app.query('#open-environment-settings')), 1,
                             'Settings must expose connections and system permissions')
            await pilot.click('#open-environment-settings')
            await pilot.pause()
            token = app.screen.query_one('#env-hf-token', Input)
            self.assertTrue(token.password)
            token.focus()
            await pilot.press(*'hf_testonly')
            app.screen.query_one('#env-hf-endpoint', Input).value = 'https://hub.example'
            self.assertTrue(await pilot.click('#save-environment-settings'))
            await pilot.pause()
            env = {}
            load_project_env(self.root, environ=env)
            self.assertEqual(env['HF_TOKEN'], 'hf_testonly')
            self.assertEqual(os.environ['HF_TOKEN'], 'hf_testonly')
            self.assertEqual(env['HF_ENDPOINT'], 'https://hub.example')
            self.assertFalse((self.root / 'tui.json').exists())
            status = app.screen.query_one('#environment-status', Static).content
            assert isinstance(status, str)
            self.assertNotIn('hf_testonly', status)
            await pilot.press('escape')
            await pilot.pause()
            self.assertFalse(app._is_busy())
            await pilot.click('#open-environment-settings')
            await pilot.pause()
            self.assertEqual(app.screen.query_one('#env-hf-token', Input).value, 'hf_testonly')

    async def test_close_does_not_save_and_busy_run_blocks_configuration(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            self.assertEqual(len(app.query('#open-environment-settings')), 1)
            await pilot.click('#open-environment-settings')
            await pilot.pause()
            app.screen.query_one('#env-hf-token', Input).value = 'hf_unsaved'
            await pilot.press('escape')
            await pilot.pause()
            self.assertFalse((self.root / '.env.local').exists())
            self.assertNotIn('HF_TOKEN', os.environ)
            app._process_kind = 'run'
            app._set_busy(True)
            self.assertTrue(app.query_one('#open-environment-settings', Button).disabled)
            app._process_kind = ''
            app._set_busy(False)


if __name__ == '__main__':
    unittest.main()
