"""
Tests for the centralized theme contract (robotics_sim/app/theme.py) and for
the architectural guarantee that lives alongside it: simulated-world
semantic colors (robot, FoV, routes, obstacles-as-data, hazards) never
change with theme, and SimulationCanvas.set_theme_mode() invalidates only
the pixmap caches that actually bake in theme colors.

Every test here redirects QSettings to a fresh temp directory before it
runs (see _isolated_theme_settings below) so nothing ever touches the real
user's registry/config file -- see theme.open_theme_settings()'s docstring
for why the explicit (IniFormat, UserScope) constructor is required for
that redirection to actually take effect on Windows.
"""
from __future__ import annotations

import inspect
import tempfile

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.theme import (
    THEME_SETTINGS_KEY,
    DEFAULT_THEME_MODE,
    ThemeMode,
    open_theme_settings,
    parse_theme_mode,
    theme_colors,
)
from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.app.main_window import MainWindow
from robotics_sim.simulation import config as sim_config

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_theme_settings():
    """Give every test in this file its own throwaway settings directory so
    "missing key" tests never see a value left behind by another test."""
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp())
    yield


# ---------------------------------------------------------------------------
# Default resolution and persistence rules (theme.parse_theme_mode /
# open_theme_settings): missing key -> light, invalid value -> light, a
# valid saved value round-trips.
# ---------------------------------------------------------------------------


def _new_window() -> MainWindow:
    """MainWindow.__init__ unconditionally starts a real QTimer driving
    on_simulation_tick(); stop it immediately -- these are pure theme/
    persistence tests and must never let a real simulation frame fire."""
    window = MainWindow()
    window.timer.stop()
    return window


def test_default_theme_is_light():
    assert DEFAULT_THEME_MODE == ThemeMode.LIGHT

    window = _new_window()
    assert window._theme_mode == ThemeMode.LIGHT
    assert window.top_bar.theme_button.toolTip() == "Light mode active — switch to dark mode"


def test_invalid_saved_theme_falls_back_to_light():
    assert parse_theme_mode(None) == ThemeMode.LIGHT
    assert parse_theme_mode("not-a-real-theme") == ThemeMode.LIGHT
    assert parse_theme_mode("") == ThemeMode.LIGHT

    settings = open_theme_settings()
    settings.setValue(THEME_SETTINGS_KEY, "purple-haze")
    settings.sync()

    window = _new_window()
    assert window._theme_mode == ThemeMode.LIGHT


def test_theme_preference_persists():
    window = _new_window()
    window._toggle_theme()
    assert window._theme_mode == ThemeMode.DARK

    # A fresh MainWindow (a stand-in for "close and reopen the app") reads
    # the same QSettings store back and must resume in dark.
    reopened = _new_window()
    assert reopened._theme_mode == ThemeMode.DARK


# ---------------------------------------------------------------------------
# Simulated-world semantic colors must never depend on the active theme.
# ---------------------------------------------------------------------------


def test_canvas_semantic_colors_are_not_theme_dependent():
    before = (
        sim_config.BLUE,
        sim_config.GREEN,
        sim_config.ORANGE,
        sim_config.RED,
        sim_config.MAROON,
    )

    canvas = SimulationCanvas()
    canvas.set_theme_mode(ThemeMode.DARK)
    canvas.set_theme_mode(ThemeMode.LIGHT)

    after = (
        sim_config.BLUE,
        sim_config.GREEN,
        sim_config.ORANGE,
        sim_config.RED,
        sim_config.MAROON,
    )
    assert after == before, "config.py's semantic palette must never be mutated by theme switching"

    # robot_color() takes no theme argument at all -- confirm the semantic
    # draw methods that call it/BLUE/GREEN/ORANGE/RED never touch theme.py.
    semantic_methods = (
        SimulationCanvas.draw_sensor_range,
        SimulationCanvas.draw_planned_route,
        SimulationCanvas.draw_multi_planned_routes,
        SimulationCanvas.draw_goal_and_robot,
        SimulationCanvas.draw_fires,
        SimulationCanvas.draw_discovered_hazard,
    )
    for method in semantic_methods:
        source = inspect.getsource(method)
        assert "theme_colors(" not in source, (
            f"{method.__name__} must keep reading simulated-world colors "
            "directly from config.py, not from theme.py"
        )


def test_canvas_theme_change_invalidates_only_theme_cache():
    canvas = SimulationCanvas()

    # Populate theme-dependent pixmap caches with dummy pixmaps ...
    canvas._static_plot_cache = QPixmap(4, 4)
    canvas._obstacles_cache = QPixmap(4, 4)
    canvas._explored_area_cache = QPixmap(4, 4)

    # ... and populate simulated-world state that must survive untouched.
    canvas.explored_area_polygons = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]
    canvas.planned_path_points = [(0.0, 0.0), (2.0, 2.0)]
    canvas.known_obstacles = [(0.0, 0.0, 1.0, 1.0)]
    seed_mask_marker = object()
    canvas._explored_area_seed_mask = seed_mask_marker

    canvas.set_theme_mode(ThemeMode.DARK)

    assert canvas._static_plot_cache is None
    assert canvas._obstacles_cache is None
    assert canvas._explored_area_cache is None

    assert canvas.explored_area_polygons == [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]
    assert canvas.planned_path_points == [(0.0, 0.0), (2.0, 2.0)]
    assert canvas.known_obstacles == [(0.0, 0.0, 1.0, 1.0)]
    assert canvas._explored_area_seed_mask is seed_mask_marker

    # Setting the same mode again is a no-op -- no redundant invalidation.
    canvas._static_plot_cache = QPixmap(4, 4)
    canvas.set_theme_mode(ThemeMode.DARK)
    assert canvas._static_plot_cache is not None


def test_light_and_dark_palettes_define_distinct_colors_for_every_field():
    light = theme_colors(ThemeMode.LIGHT)
    dark = theme_colors(ThemeMode.DARK)
    for field in light.__dataclass_fields__:
        assert getattr(light, field) != getattr(dark, field), field
