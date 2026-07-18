"""
Tests for the right-side panel deck's default visibility (Configuration
shown, Navigation Reasoning hidden on a fresh app start) and that a theme
change never alters which panels are open -- only how they are painted.
"""
from __future__ import annotations

import tempfile

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

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


def test_configuration_panel_is_visible_by_default(window):
    assert window._configuration_panel_visible is True
    assert window.configuration_panel_action.isChecked() is True


def test_navigation_reasoning_panel_is_hidden_by_default(window):
    assert window._navigation_reasoning_panel_visible is False
    assert window.navigation_reasoning_panel_action.isChecked() is False
    assert window.side_panel_tabs.indexOf(window.navigation_reasoning_window) < 0


def test_default_panel_visibility_is_unaffected_by_starting_theme():
    """Loading a previously-saved dark preference must not change which
    panels are open on startup -- only _load_saved_theme()/_apply_theme()
    run differently, panel visibility is completely independent state."""
    from robotics_sim.app.theme import THEME_SETTINGS_KEY, open_theme_settings

    settings = open_theme_settings()
    settings.setValue(THEME_SETTINGS_KEY, "dark")
    settings.sync()

    dark_start_window = MainWindow()
    dark_start_window.timer.stop()
    assert dark_start_window._theme_mode.value == "dark"
    assert dark_start_window._configuration_panel_visible is True
    assert dark_start_window._navigation_reasoning_panel_visible is False


def test_theme_toggle_does_not_open_or_close_any_panel(window):
    config_before = window._configuration_panel_visible
    reasoning_before = window._navigation_reasoning_panel_visible

    window._toggle_theme()

    assert window._configuration_panel_visible == config_before
    assert window._navigation_reasoning_panel_visible == reasoning_before
