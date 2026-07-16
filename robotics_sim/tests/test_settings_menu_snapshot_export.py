"""
Tests for the top-bar "⋮" application menu (formerly a single gear/wheel
button, now split into menu_button "⋮" + theme_button "☀"/"☾" -- see
theme.py and widgets.TopBar). Covers:

- the gear button is gone; menu_button is what opens the panel/file menu;
- every pre-existing action (Configuration, Navigation Reasoning, Export
  snapshots to Excel, Load .sim, Save .sim) is still present and wired;
- exporting with an empty navigation-debug log shows the existing "No
  snapshots" message and never opens a file dialog (so this test never
  blocks on a real modal).
"""
from __future__ import annotations

import tempfile

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from robotics_sim.app.main_window import MainWindow

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_theme_settings():
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp())
    yield


@pytest.fixture()
def window():
    win = MainWindow()
    win.timer.stop()  # never let a real simulation frame fire during a UI-wiring test
    return win


def test_gear_button_is_replaced_by_menu_and_theme_buttons(window):
    assert not hasattr(window.top_bar, "gear_button")
    assert hasattr(window.top_bar, "menu_button")
    assert hasattr(window.top_bar, "theme_button")
    assert window.top_bar.menu_button.toolTip() == "Open application menu"


def test_menu_button_opens_the_panel_visibility_menu(monkeypatch):
    # QMenu.exec() is a native Qt method invoked on an *instance*
    # (self.panel_visibility_menu.exec(position)); PySide6 does not honor a
    # plain monkeypatch.setattr(QMenu, "exec", ...) override for calls made
    # that way. _show_panel_visibility_menu() is a plain Python method
    # though, so patch *that* on the class before construction, so the
    # clicked.connect(self._show_panel_visibility_menu) call made during
    # build_ui() captures the patched version.
    opened = []
    monkeypatch.setattr(MainWindow, "_show_panel_visibility_menu", lambda self: opened.append(True))

    window = MainWindow()
    window.timer.stop()
    window.top_bar.menu_button.click()

    assert opened == [True]


def test_menu_keeps_every_existing_action(window):
    window._prepare_panel_visibility_menu()
    action_texts = [action.text() for action in window.panel_visibility_menu.actions() if action.text()]

    for expected in (
        "Configuration",
        "Navigation Reasoning",
        "Export snapshots to Excel",
        "Load .sim",
        "Save .sim",
    ):
        assert any(expected in text for text in action_texts), (expected, action_texts)


def test_export_action_triggers_the_real_export_flow(window, monkeypatch):
    """Trigger the real, already-wired QAction (not a re-connected stand-in)
    and observe a real side effect of export_navigation_snapshots() -- with
    an empty log that is the "No snapshots" message box, never a file
    dialog. Confirms the menu action still reaches engine.py's handler."""
    assert len(window.navigation_debug_log) == 0, "a freshly constructed window's log starts empty"
    info_calls = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: info_calls.append(a))
    dialog_calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: dialog_calls.append(a) or ("", ""))

    window.export_snapshots_action.trigger()

    assert info_calls, "expected the existing 'No snapshots' message box"
    assert dialog_calls == [], "must not open a save dialog when there is nothing to export"


def test_load_and_save_sim_actions_remain_wired(window, monkeypatch):
    """Trigger the real, already-wired QActions and confirm they still
    reach QFileDialog -- proof the menu action is connected to the real
    load/save handler, without actually touching the filesystem (both
    handlers return immediately on an empty/cancelled path)."""
    open_calls = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: open_calls.append(a) or ("", ""))
    save_calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: save_calls.append(a) or ("", ""))

    window.load_sim_action.trigger()
    window.save_sim_action.trigger()

    assert open_calls, "Load .sim must still reach QFileDialog.getOpenFileName"
    assert save_calls, "Save .sim must still reach QFileDialog.getSaveFileName"


def test_menu_button_is_legible_and_theme_toggle_does_not_disturb_it(window):
    window._toggle_theme()
    window._prepare_panel_visibility_menu()
    action_texts = [action.text() for action in window.panel_visibility_menu.actions() if action.text()]
    assert any("Configuration" in text for text in action_texts)
    assert any("Export snapshots to Excel" in text for text in action_texts)
