"""
Tests for the theme_button / menu_button split in TopBar and MainWindow's
toggle wiring (_toggle_theme, _apply_theme, _update_theme_button):

- clicking the sun/moon button really does flip the app-wide theme and
  never opens the ⋮ menu, and vice versa;
- toggling theme is a purely cosmetic operation -- it must never reset
  SimulationEngine state, snapshots, history position, or run/pause state;
- the ⋮ menu keeps every one of its existing actions;
- the docked Navigation Reasoning panel and the Configuration panel both
  stay visible (and simply re-themed) across a toggle.

A single MainWindow() is shared across this module (same convention as
test_navigation_panel_controls.py) -- construction is the expensive part
and everything here is read-only wiring/state inspection, never .show()'n.
"""
from __future__ import annotations

import tempfile

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from robotics_sim.app.main_window import MainWindow
from robotics_sim.app.theme import ThemeMode

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_theme_settings():
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp())
    yield


@pytest.fixture()
def window():
    win = MainWindow()
    # MainWindow.__init__ unconditionally starts a real QTimer driving
    # on_simulation_tick(); these tests are pure UI/theme wiring and never
    # want a real simulation frame to fire mid-test (a stray tick against
    # test-only sentinel state would crash and, if the event loop is ever
    # pumped later in the session, keep retrying).
    win.timer.stop()
    return win


# ---------------------------------------------------------------------------
# Toggle behavior: light -> dark -> light, via a real button click.
# ---------------------------------------------------------------------------


def test_theme_toggle_switches_to_dark(window):
    assert window._theme_mode == ThemeMode.LIGHT
    assert window.top_bar.theme_button.toolTip() == "Light mode active — switch to dark mode"

    window.top_bar.theme_button.click()

    assert window._theme_mode == ThemeMode.DARK
    assert window.top_bar.theme_button.toolTip() == "Dark mode active — switch to light mode"


def test_second_toggle_returns_to_light(window):
    window.top_bar.theme_button.click()
    assert window._theme_mode == ThemeMode.DARK

    window.top_bar.theme_button.click()

    assert window._theme_mode == ThemeMode.LIGHT
    assert window.top_bar.theme_button.toolTip() == "Light mode active — switch to dark mode"


# ---------------------------------------------------------------------------
# Toggling theme must be side-effect-free with respect to the simulation.
# ---------------------------------------------------------------------------


def test_theme_toggle_does_not_reset_simulation(window):
    sentinel_robot = object()
    window.robot = sentinel_robot
    window.simulation_time = 42.5
    window.running = True
    window.paused = True
    window.navigation_debug_enabled = True
    event_log = window.navigation_debug_log
    window.canvas._nav_debug_history_position = (3, 7)

    window._toggle_theme()

    assert window.robot is sentinel_robot
    assert window.simulation_time == 42.5
    assert window.running is True
    assert window.paused is True
    assert window.navigation_debug_enabled is True
    assert window.navigation_debug_log is event_log
    assert window.canvas._nav_debug_history_position == (3, 7)


# ---------------------------------------------------------------------------
# The ⋮ menu keeps all of its actions and is unaffected by theming.
# ---------------------------------------------------------------------------


def test_settings_menu_remains_available(window):
    menu_button = window.top_bar.menu_button
    assert menu_button.toolTip() == "Open application menu"

    window._toggle_theme()

    # Same refresh _show_panel_visibility_menu() runs right before exec() --
    # safe to call directly, it never blocks.
    window._prepare_panel_visibility_menu()
    action_texts = [action.text() for action in window.panel_visibility_menu.actions() if action.text()]
    for expected in ("Configuration", "Navigation Reasoning", "Export snapshots to Excel", "Load .sim", "Save .sim"):
        assert any(expected in text for text in action_texts), (expected, action_texts)


def test_theme_and_settings_buttons_are_independent(monkeypatch):
    # _show_panel_visibility_menu() ends in a real, blocking QMenu.exec();
    # QMenu.exec is a native Qt method, and PySide6 does not honor a plain
    # monkeypatch.setattr(QMenu, "exec", ...) override for it (the C++
    # binding resolves the call outside normal Python attribute lookup).
    # _show_panel_visibility_menu() itself is a plain Python method though,
    # so patch *that* on the class -- before construction, so the
    # self.top_bar.menu_button.clicked.connect(self._show_panel_visibility_
    # menu) call made during build_ui() captures the patched version.
    menu_opened = []
    monkeypatch.setattr(MainWindow, "_show_panel_visibility_menu", lambda self: menu_opened.append(True))

    window = MainWindow()
    window.timer.stop()

    # Clicking the theme button must never open the menu ...
    window.top_bar.theme_button.click()
    assert menu_opened == []

    # ... and clicking the menu button must never change the theme.
    mode_before = window._theme_mode
    window.top_bar.menu_button.click()
    assert window._theme_mode == mode_before
    assert menu_opened == [True]


# ---------------------------------------------------------------------------
# Panels stay visible (and simply re-themed) across a toggle.
# ---------------------------------------------------------------------------


def test_navigation_panel_remains_visible_after_theme_change(window):
    window.navigation_reasoning_panel_action.setChecked(True)
    assert window._navigation_reasoning_panel_visible is True
    assert window.side_panel_tabs.indexOf(window.navigation_reasoning_window) >= 0

    window._toggle_theme()

    assert window._navigation_reasoning_panel_visible is True
    assert window.side_panel_tabs.indexOf(window.navigation_reasoning_window) >= 0
    assert window.navigation_reasoning_window._theme_mode == window._theme_mode


def test_configuration_panel_remains_visible_after_theme_change(window):
    window.set_configuration_panel_visible(True)
    assert window._configuration_panel_visible is True
    stack_before = window.config_panel_stack

    window._toggle_theme()

    assert window._configuration_panel_visible is True
    assert window.config_panel_stack is stack_before, "the panel widget must never be torn down/rebuilt by a theme change"
