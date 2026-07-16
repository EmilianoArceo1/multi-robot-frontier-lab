"""Centralized light/dark application theme contract.

This module owns UI *chrome* only: window/panel/card/border/text/input/menu
colors used by the app's own widgets (top bar, side panels, buttons, forms,
tabs, menus, tooltips, scrollbars). It intentionally does not know anything
about the simulated world.

The simulated-world semantic palette -- robot colors, FoV, occupancy grid,
explored area, obstacles, routes, waypoints, frontiers, goal, hazard
heatmap, fire, collision/safety indicators -- lives entirely in
robotics_sim.simulation.config (MAROON/BLUE/GREEN/ORANGE/RED/GRID/
OBSTACLE_*/ROBOT_COLOR_HEXES/robot_color()) and is deliberately untouched by
theme switching: SimulationCanvas keeps reading those module constants
directly for every semantic draw call. Only a handful of canvas *chrome*
draw calls (card background, header/footer, plot backdrop, borders) read
from this module instead -- see SimulationCanvas.set_theme_mode()'s
docstring for the exact split.

No widget should ever write `if dark_mode: color = ...` -- the pattern is
always "ask this module for the current ThemeColors, use its fields." Any
widget with its own cached QPainter output (SimulationCanvas, and nothing
else in this app) additionally implements a small `set_theme_mode(mode)`
that stores the mode, invalidates only the caches whose pixels depend on
theme colors, and repaints -- never a full data/state rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication


class ThemeMode(str, Enum):
    LIGHT = "light"
    DARK = "dark"


# QSettings identity (no prior QSettings usage exists anywhere in this app
# to match) and the single stable key the chosen theme is persisted under.
# Only the theme *name* is ever written here -- never a stylesheet or a
# resolved ThemeColors -- so a future palette tweak in this module still
# applies correctly to an already-saved "dark"/"light" preference.
SETTINGS_ORGANIZATION = "Robotics Simulation Lab"
SETTINGS_APPLICATION = "Robotics Simulation Lab"
THEME_SETTINGS_KEY = "appearance/theme"
DEFAULT_THEME_MODE = ThemeMode.LIGHT


def parse_theme_mode(value: object) -> ThemeMode:
    """Parse a raw QSettings value into a valid ThemeMode.

    Missing or invalid values fall back to DEFAULT_THEME_MODE (light) --
    this is the one place that rule is encoded, so both MainWindow's
    startup load and any test exercising the fallback go through the same
    logic.
    """
    if value is None:
        return DEFAULT_THEME_MODE
    try:
        return ThemeMode(str(value))
    except ValueError:
        return DEFAULT_THEME_MODE


def open_theme_settings() -> QSettings:
    """Open the QSettings store the theme preference lives in.

    Always the explicit (IniFormat, UserScope) constructor -- never the
    ambiguous 2-arg QSettings(organization, application) form, which
    resolves to the Windows registry on Windows and ignores
    QSettings.setPath()/setDefaultFormat() redirection. Using IniFormat
    explicitly means tests can redirect storage to a temp directory via
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tmp_dir)
    and never touch a real user's registry or config file.
    """
    return QSettings(QSettings.IniFormat, QSettings.UserScope, SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


@dataclass(frozen=True)
class ThemeColors:
    """One resolved palette. Every field is a CSS-legal color string (hex),
    safe to interpolate directly into a Qt stylesheet or QColor(...)."""

    app_background: str
    panel_background: str
    card_background: str
    elevated_background: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_disabled: str
    accent: str
    accent_hover: str
    destructive: str
    success: str
    warning: str
    input_background: str
    menu_background: str
    scrollbar: str


# Light is the original, already-shipped palette (see robotics_sim.
# simulation.config's BG/CARD/PANEL_CARD/TEXT*/BORDER* -- kept numerically
# identical here so switching to LIGHT never visibly changes anything for
# users who never open the theme menu).
LIGHT_THEME = ThemeColors(
    app_background="#F4F5F7",
    panel_background="#FDFDFC",
    card_background="#FFFFFF",
    elevated_background="#F2F3F6",
    border="#DADFE7",
    border_strong="#B7BFCC",
    text_primary="#22252A",
    text_secondary="#777B84",
    text_disabled="#A5A9B2",
    accent="#500000",
    accent_hover="#3A0000",
    destructive="#B42318",
    success="#219653",
    warning="#B0530A",
    input_background="#FFFFFF",
    menu_background="#FFFFFF",
    scrollbar="#C7CCD6",
)

# Dark: every background/text pair below was chosen to keep readable
# contrast without ever pairing pure black (#000000) with dark grey, and
# without pure white (#FFFFFF) body text -- see the module docstring in
# navigation_reasoning_window.py for why that matters for the badges
# specifically.
DARK_THEME = ThemeColors(
    app_background="#16181C",
    panel_background="#1D2025",
    card_background="#23262C",
    elevated_background="#2B2F36",
    border="#383D45",
    border_strong="#4B515A",
    text_primary="#C9CDD3",
    text_secondary="#9BA0AA",
    text_disabled="#6B7078",
    accent="#D98488",
    accent_hover="#E8A0A3",
    destructive="#F1867E",
    success="#5FC98A",
    warning="#EFAE5C",
    input_background="#282C33",
    menu_background="#22262C",
    scrollbar="#454B54",
)

_PALETTES: dict[ThemeMode, ThemeColors] = {
    ThemeMode.LIGHT: LIGHT_THEME,
    ThemeMode.DARK: DARK_THEME,
}


def theme_colors(mode: ThemeMode | str) -> ThemeColors:
    """Resolve a ThemeMode (or its raw string value) to a ThemeColors."""
    return _PALETTES[ThemeMode(mode)]


def with_alpha(hex_color: str, alpha: int) -> str:
    """Return `hex_color` as an `rgba(...)` CSS string with the given alpha
    (0-255). Used to derive translucent tints (status badge backgrounds,
    hover washes) from a single solid theme token instead of hand-picking a
    second hardcoded color per state."""
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {max(0, min(255, int(alpha)))})"


def dropdown_popup_stylesheet(mode: ThemeMode | str) -> str:
    """QSS for a QComboBox popup's own QListView.

    Applied directly to `combo.view()` (not just relied on via the global
    `QComboBox QAbstractItemView` selector in build_application_stylesheet())
    because some Windows styles ignore part of the app-level QSS for a
    popup's own view and can otherwise show unreadable text -- see
    config_panel.labeled_combo()'s docstring. MainWindow._apply_theme() re-
    applies this to every combo box tagged themedDropdownPopup=True
    whenever the theme changes, since a popup view's stylesheet is not
    itself reachable by the app-level cascade the same way ordinary child
    widgets are.
    """
    c = theme_colors(mode)
    return f"""
        QListView {{
            background-color: {c.menu_background};
            color: {c.text_primary};
            border: 1px solid {c.border};
            border-radius: 6px;
            padding: 4px;
            outline: 0px;
            selection-background-color: {with_alpha(c.accent, 46)};
            selection-color: {c.accent};
        }}

        QListView::item {{
            min-height: 28px;
            padding: 6px 8px;
            color: {c.text_primary};
            background-color: {c.menu_background};
        }}

        QListView::item:selected {{
            color: {c.accent};
            background-color: {with_alpha(c.accent, 46)};
        }}

        QListView::item:hover {{
            color: {c.accent};
            background-color: {with_alpha(c.accent, 28)};
        }}
    """


def build_application_stylesheet(mode: ThemeMode | str) -> str:
    """Return the full application-wide Qt stylesheet for `mode`.

    This is the single source of QSS for the app's own chrome -- top bar,
    side panels, tabs, forms, buttons, menus, tooltips, scrollbars. Widgets
    that paint themselves (SimulationCanvas's chrome layers, ToggleSwitch,
    NavigationReasoningWindow's per-instance stylesheet) read ThemeColors
    directly instead of relying on QSS selectors reaching into them.
    """
    c = theme_colors(mode)
    return f"""
    QWidget#root {{
        background: {c.app_background};
        font-family: "Segoe UI";
        color: {c.text_primary};
    }}

    QWidget#body {{
        background: {c.app_background};
    }}

    QFrame#topBar {{
        background: qlineargradient(
            x1: 0, y1: 0,
            x2: 1, y2: 0,
            stop: 0 #3A0000,
            stop: 1 #500000
        );
    }}

    QLabel#topTitle {{
        color: white;
        font-size: 14px;
        font-weight: 900;
        background: transparent;
    }}

    QLabel#statusReady,
    QLabel#statusRunning {{
        color: #6FCF97;
        font-size: 12px;
        font-weight: 800;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 5px 13px;
    }}

    QLabel#statusPaused {{
        color: #F0A868;
        font-size: 12px;
        font-weight: 800;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 5px 13px;
    }}

    QPushButton#topIconButton,
    QPushButton#windowButton,
    QPushButton#closeButton {{
        background: transparent;
        border: none;
        border-radius: 6px;
    }}

    QPushButton#topIconButton:hover,
    QPushButton#windowButton:hover {{
        background: rgba(255,255,255,0.12);
    }}

    QPushButton#closeButton:hover {{
        background: #B42318;
    }}

    QComboBox#topModeSelector {{
        background: rgba(255,255,255,0.08);
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 6px;
        padding-left: 9px;
        font-size: 11px;
        font-weight: 800;
        min-height: 28px;
    }}

    QComboBox#topModeSelector::drop-down {{
        width: 22px;
        border: none;
    }}

    QPushButton#modeSegmentButton {{
        background: rgba(255,255,255,0.08);
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 7px;
        font-size: 11px;
        font-weight: 900;
    }}

    QPushButton#modeSegmentButton:hover {{
        background: rgba(255,255,255,0.14);
    }}

    QPushButton#modeSegmentButton:checked {{
        background: #FFFFFF;
        color: #500000;
        border: 1px solid #FFFFFF;
    }}

    QPushButton#modeSegmentButton:disabled {{
        color: rgba(255,255,255,0.45);
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }}

    QFrame#sidePanel,
    QWidget#sidePanelContainer {{
        background: {c.card_background};
        border: 1px solid {c.border};
        border-radius: 14px;
    }}

    QTabWidget#sidePanelTabs {{
        background: transparent;
        border: none;
    }}

    QTabWidget#sidePanelTabs::pane {{
        background: {c.card_background};
        border: none;
        border-radius: 12px;
        top: -1px;
    }}

    QTabWidget#sidePanelTabs QTabBar::tab {{
        background: {c.elevated_background};
        color: {c.text_secondary};
        border: 1px solid {c.border};
        border-bottom: none;
        padding: 8px 12px;
        min-width: 120px;
        font-size: 10px;
        font-weight: 850;
    }}

    QTabWidget#sidePanelTabs QTabBar::tab:first {{
        border-top-left-radius: 10px;
    }}

    QTabWidget#sidePanelTabs QTabBar::tab:last {{
        border-top-right-radius: 10px;
    }}

    QTabWidget#sidePanelTabs QTabBar::tab:selected {{
        background: {c.card_background};
        color: {c.accent};
        border-color: {c.border};
    }}

    QTabWidget#sidePanelTabs QTabBar::tab:hover:!selected {{
        background: {c.border};
        color: {c.text_primary};
    }}

    QScrollArea#configScroll,
    QScrollArea#navigationReasoningScroll {{
        background: transparent;
        border: none;
    }}

    QWidget#scrollContent {{
        background: transparent;
    }}

    QFrame#sectionCard {{
        background: {c.panel_background};
        border: 1px solid {c.border};
        border-radius: 9px;
    }}

    QFrame#actionPanelBottom {{
        background: {c.card_background};
        border-top: 1px solid {c.border};
        border-bottom-left-radius: 14px;
        border-bottom-right-radius: 14px;
    }}

    QLabel#sectionTitle {{
        color: {c.accent};
        font-size: 13px;
        font-weight: 900;
    }}

    QLabel#fieldLabel {{
        color: {c.text_secondary};
        font-size: 10px;
        font-weight: 700;
    }}

    QLabel#subsectionLabel {{
        color: {c.text_secondary};
        font-size: 10px;
        font-weight: 900;
        padding-top: 3px;
    }}

    QLabel {{
        color: {c.text_primary};
    }}

    QPushButton#stepperButton {{
        background: {c.elevated_background};
        color: {c.accent};
        border: 1px solid {c.border};
        border-radius: 5px;
        min-height: 28px;
        font-size: 13px;
        font-weight: 900;
    }}

    QPushButton#stepperButton:hover {{
        background: {c.border};
        border: 1px solid {c.accent};
    }}

    QLineEdit#numericInput,
    QLineEdit#smallNumericInput {{
        background: {c.input_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 5px;
        min-height: 28px;
        font-size: 11px;
        font-weight: 900;
        padding-left: 4px;
        padding-right: 4px;
    }}

    QLineEdit#numericInput:focus,
    QLineEdit#smallNumericInput:focus {{
        border: 2px solid {c.accent};
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background: {c.input_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 5px;
        selection-background-color: {c.accent};
        selection-color: #FFFFFF;
    }}

    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
    QComboBox:disabled, QPushButton:disabled {{
        color: {c.text_disabled};
    }}

    QComboBox {{
        background-color: {c.input_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 7px;
        min-height: 34px;
        padding-left: 10px;
        padding-right: 28px;
        font-size: 12px;
        font-weight: 800;
        selection-background-color: {with_alpha(c.accent, 46)};
        selection-color: {c.accent};
    }}

    QComboBox:hover {{
        background-color: {c.elevated_background};
        border: 1px solid {c.border_strong};
    }}

    QComboBox:focus {{
        background-color: {c.input_background};
        border: 2px solid {c.accent};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 28px;
        border-left: 1px solid {c.border};
        border-top-right-radius: 7px;
        border-bottom-right-radius: 7px;
        background-color: transparent;
    }}

    QComboBox::down-arrow {{
        image: none;
        width: 0px;
        height: 0px;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {c.text_secondary};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c.menu_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 6px;
        padding: 4px;
        outline: 0px;
        selection-background-color: {with_alpha(c.accent, 46)};
        selection-color: {c.accent};
    }}

    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 6px 8px;
        color: {c.text_primary};
        background-color: {c.menu_background};
    }}

    QComboBox QAbstractItemView::item:selected {{
        color: {c.accent};
        background-color: {with_alpha(c.accent, 46)};
    }}

    QComboBox QAbstractItemView::item:hover {{
        color: {c.accent};
        background-color: {with_alpha(c.accent, 28)};
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {c.border};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: {c.accent};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {c.accent};
        border: 2px solid {c.card_background};
        width: 13px;
        height: 13px;
        margin: -5px 0;
        border-radius: 7px;
    }}

    QCheckBox {{
        color: {c.text_primary};
        font-size: 11px;
        font-weight: 700;
    }}

    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 3px;
        border: 1.5px solid {c.accent};
        background: {c.input_background};
    }}

    QCheckBox::indicator:checked {{
        background: {c.accent};
    }}

    QFrame#canvasActionBar {{
        background: {with_alpha(c.card_background, 245)};
        border: 1px solid {c.border};
        border-radius: 9px;
    }}

    QPushButton#canvasStartButton {{
        background: #500000;
        color: white;
        border: none;
        border-radius: 7px;
        min-height: 30px;
        font-size: 11px;
        font-weight: 900;
    }}

    QPushButton#canvasStartButton:hover {{
        background: #6A0000;
    }}

    QPushButton#canvasActionButton {{
        background: {c.card_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 7px;
        min-height: 30px;
        font-size: 10px;
        font-weight: 800;
    }}

    QPushButton#canvasActionButton:hover {{
        background: {c.elevated_background};
        border-color: {c.border_strong};
    }}

    QPushButton#startButton {{
        background: #500000;
        color: white;
        border: none;
        border-radius: 7px;
        min-height: 40px;
        font-size: 14px;
        font-weight: 900;
    }}

    QPushButton#startButton:hover {{
        background: #6A0000;
    }}

    QPushButton#secondaryButton {{
        background: {c.card_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 6px;
        min-height: 32px;
        font-size: 11px;
        font-weight: 800;
    }}

    QPushButton#secondaryButton:hover {{
        background: {c.elevated_background};
    }}

    QPushButton#secondaryButton:checked {{
        background: {with_alpha(c.accent, 40)};
        color: {c.accent};
        border: 1px solid {c.accent};
    }}

    QTableWidget {{
        background: #1F1F1F;
        color: #FFFFFF;
        gridline-color: #3D3D3D;
        border: 1px solid #4B4B4B;
        border-radius: 8px;
        font-size: 12px;
    }}

    QTableWidget::item {{
        padding: 7px 8px;
    }}

    QTableWidget::item:selected {{
        background: #0B79D0;
        color: #FFFFFF;
    }}

    QHeaderView::section {{
        background: #343434;
        color: #FFFFFF;
        border: none;
        border-bottom: 1px solid #525252;
        padding: 8px;
        font-size: 12px;
        font-weight: 900;
    }}

    QLabel#metricsMessageBox {{
        background: #252525;
        color: #FFFFFF;
        border: 1px solid #4B4B4B;
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
    }}

    QPlainTextEdit#consoleText {{
        background: #171717;
        color: #F3F4F6;
        border: 1px solid #4B4B4B;
        border-radius: 8px;
        padding: 10px;
        font-family: Consolas, "Cascadia Mono", monospace;
        font-size: 11px;
    }}

    QMenu {{
        background-color: {c.menu_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 8px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 7px 22px 7px 12px;
        border-radius: 5px;
        color: {c.text_primary};
    }}

    QMenu::item:selected {{
        background-color: {with_alpha(c.accent, 40)};
        color: {c.accent};
    }}

    QMenu::item:disabled {{
        color: {c.text_disabled};
    }}

    QMenu::separator {{
        height: 1px;
        background: {c.border};
        margin: 4px 8px;
    }}

    QMenu::indicator {{
        width: 13px;
        height: 13px;
    }}

    QToolTip {{
        background-color: {c.menu_background};
        color: {c.text_primary};
        border: 1px solid {c.border};
        border-radius: 4px;
        padding: 4px 6px;
    }}

    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 5px;
    }}

    QScrollBar::handle:vertical {{
        background: {c.scrollbar};
        border-radius: 2px;
    }}

    QScrollBar:horizontal {{
        height: 0px;
    }}

    QPushButton#configPanelCloseButton {{
        border: none;
        background: {with_alpha(c.card_background, 200)};
        color: {c.text_secondary};
        border-radius: 5px;
        font-size: 18px;
        font-weight: 600;
    }}

    QPushButton#configPanelCloseButton:hover {{
        background: {c.card_background};
        color: {c.text_primary};
    }}

    QLabel#editorStatusLabel {{
        font-size: 11px;
        color: {c.text_secondary};
        line-height: 1.35;
    }}
    """


def apply_application_theme(app: QApplication, mode: ThemeMode | str) -> None:
    """Apply `mode`'s stylesheet to the whole application (every top-level
    widget, including popups/menus/tooltips that don't inherit a specific
    QMainWindow's own setStyleSheet())."""
    app.setStyleSheet(build_application_stylesheet(mode))
