"""
Tests for the canvas footer action bar (Start/Restart/Speed/Metrics/Console
-- main_window._build_canvas_action_bar()): the buttons exist, keep their
wiring and labels, and stay themed (background/text via the global QSS,
hand-drawn icons refreshed explicitly) across a light/dark toggle without
any of it affecting the simulation itself.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from robotics_sim.app.main_window import MainWindow
from robotics_sim.app.theme import ThemeMode, theme_colors

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


def test_action_bar_buttons_exist_with_expected_labels(window):
    assert window.start_button.text() == "Start"
    assert window.reset_button.text() == "Restart"
    assert window.speed_button.text().startswith("Speed")
    assert window.metrics_button.text() == "Metrics"
    assert window.console_button.text() == "Console"

    assert window.canvas_action_bar.objectName() == "canvasActionBar"
    for button in (window.start_button, window.reset_button, window.speed_button, window.metrics_button, window.console_button):
        assert button.parentWidget() is window.canvas_action_bar


def test_action_bar_buttons_stay_wired_after_theme_toggle(window, monkeypatch):
    start_calls = []
    reset_calls = []
    monkeypatch.setattr(window, "handle_start_pause_button", lambda: start_calls.append(True))
    monkeypatch.setattr(window, "restart_simulation", lambda: reset_calls.append(True))
    window.start_button.clicked.disconnect()
    window.start_button.clicked.connect(window.handle_start_pause_button)
    window.reset_button.clicked.disconnect()
    window.reset_button.clicked.connect(window.restart_simulation)

    window._toggle_theme()
    window.start_button.click()
    window.reset_button.click()

    assert start_calls == [True]
    assert reset_calls == [True]


def test_restart_replaces_stale_frontier_bfs_overlay_immediately(window):
    stale = {
        "resolution": 1.0,
        "bounds": (-1.0, 1.0, -1.0, 1.0),
        "grid": np.zeros((2, 2), dtype=np.int8),
        "frontier_cells": ((0, 0),),
        "bfs_steps": np.zeros((2, 2), dtype=np.int32),
    }
    window.canvas.set_grid_overlay_snapshot(stale)
    window.canvas.set_frontier_reasoning_decision({"robot": (0, 0), "frontier": (1, 1)})

    window.restart_simulation()

    fresh = window.canvas._grid_overlay_snapshot
    assert fresh is not stale
    assert fresh["frontier_cells"] == ()
    assert np.all(fresh["bfs_steps"] == -1)
    assert window.canvas.frontier_reasoning_decision is None


def test_action_bar_icons_are_regenerated_per_theme(window):
    light_icon = window.reset_button.icon()
    assert not light_icon.isNull()

    window._toggle_theme()
    dark_icon = window.reset_button.icon()
    assert not dark_icon.isNull()

    light_pixmap = light_icon.pixmap(16, 16).toImage()
    dark_pixmap = dark_icon.pixmap(16, 16).toImage()
    assert light_pixmap != dark_pixmap, "the icon bitmap must be redrawn in the new theme's text color"


def test_action_bar_theme_toggle_does_not_touch_simulation_state(window):
    window.simulation_time = 12.0
    window.running = True
    sentinel = object()
    window.robot = sentinel

    window._toggle_theme()

    assert window.simulation_time == 12.0
    assert window.running is True
    assert window.robot is sentinel


def test_start_button_keeps_hardcoded_white_icon_both_themes(window):
    """canvasStartButton keeps its maroon background regardless of theme
    (brand-locked, like the Resume button in the navigation snapshot bar)
    -- so its icon must stay drawn in white in both modes."""
    from robotics_sim.app.widgets import make_icon

    reference = make_icon("play", "white").pixmap(18, 18).toImage()

    assert window.start_button.icon().pixmap(18, 18).toImage() == reference
    window._toggle_theme()
    assert window.start_button.icon().pixmap(18, 18).toImage() == reference
