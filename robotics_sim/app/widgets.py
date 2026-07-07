"""
Qt-only reusable widgets.

These classes are visual/UX plumbing: icons, section cards, numeric inputs,
toggle switches, top bar, and the movable metrics table. They should not
decide paths, frontiers, collisions, or robot behavior.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QPixmap,
    QIcon,
    QDoubleValidator,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import *

def make_icon(icon_type: str, color: str = TEXT) -> QIcon:
    pixmap = QPixmap(28, 28)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    c = QColor(color)
    painter.setPen(QPen(c, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)

    if icon_type == "play":
        painter.setBrush(c)
        painter.setPen(Qt.NoPen)
        painter.drawPolygon([
            QPointF(10, 7),
            QPointF(10, 21),
            QPointF(21, 14),
        ])

    elif icon_type == "pause":
        painter.setBrush(c)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(8, 7, 4.5, 14), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(15.5, 7, 4.5, 14), 1.5, 1.5)

    elif icon_type == "reset":
        painter.drawArc(QRectF(6, 6, 16, 16), 35 * 16, 280 * 16)
        painter.setBrush(c)
        painter.drawPolygon([
            QPointF(18, 4),
            QPointF(23, 8),
            QPointF(17, 10),
        ])

    elif icon_type == "save":
        painter.drawRoundedRect(QRectF(6, 5, 16, 18), 2, 2)
        painter.drawRect(QRectF(10, 7, 8, 5))
        painter.drawLine(QPointF(10, 19), QPointF(18, 19))

    elif icon_type == "gear":
        painter.drawEllipse(QRectF(9, 9, 10, 10))
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = 14 + 8 * math.cos(rad)
            y1 = 14 + 8 * math.sin(rad)
            x2 = 14 + 12 * math.cos(rad)
            y2 = 14 + 12 * math.sin(rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    elif icon_type == "minimize":
        painter.drawLine(QPointF(8, 15), QPointF(20, 15))

    elif icon_type == "maximize":
        painter.drawRect(QRectF(8, 8, 12, 12))

    elif icon_type == "close":
        painter.drawLine(QPointF(9, 9), QPointF(19, 19))
        painter.drawLine(QPointF(19, 9), QPointF(9, 19))

    elif icon_type == "console":
        painter.drawRoundedRect(QRectF(5, 6, 18, 16), 2, 2)
        painter.drawLine(QPointF(8, 11), QPointF(12, 14))
        painter.drawLine(QPointF(12, 14), QPointF(8, 17))
        painter.drawLine(QPointF(14, 17), QPointF(20, 17))

    elif icon_type == "single_robot":
        painter.drawRoundedRect(QRectF(6, 9, 16, 10), 3, 3)
        painter.drawLine(QPointF(14, 9), QPointF(14, 5))
        painter.setBrush(c)
        painter.drawEllipse(QRectF(10, 13, 2.5, 2.5))
        painter.drawEllipse(QRectF(16, 13, 2.5, 2.5))

    elif icon_type == "multi_robot":
        painter.drawRoundedRect(QRectF(5, 6, 10, 8), 2, 2)
        painter.drawRoundedRect(QRectF(14, 14, 10, 8), 2, 2)
        painter.drawLine(QPointF(15, 10), QPointF(18, 14))

    painter.end()
    return QIcon(pixmap)


# ============================================================
# VECTOR ICONS
# ============================================================

class VectorIcon(QWidget):
    def __init__(self, icon_type: str):
        super().__init__()
        self.icon_type = icon_type
        self.setFixedSize(26, 26)

    def set_icon_type(self, icon_type: str):
        self.icon_type = icon_type
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        color = QColor(MAROON_SOFT)
        painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        if self.icon_type == "robot":
            painter.drawRoundedRect(4, 8, 18, 12, 3, 3)
            painter.drawLine(13, 8, 13, 4)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(8, 12, 2.7, 2.7))
            painter.drawEllipse(QRectF(16, 12, 2.7, 2.7))

        elif self.icon_type == "dynamics":
            painter.drawArc(4, 5, 18, 18, -20 * 16, 220 * 16)
            painter.drawLine(13, 14, 18, 8)
            painter.drawEllipse(QRectF(12, 13, 2, 2))

        elif self.icon_type == "goal":
            painter.drawEllipse(4, 4, 18, 18)
            painter.drawEllipse(8, 8, 10, 10)
            painter.drawEllipse(12, 12, 2, 2)

        elif self.icon_type == "options":
            painter.drawLine(4, 8, 22, 8)
            painter.drawLine(4, 16, 22, 16)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(8, 5, 4, 6), 1, 1)
            painter.drawRoundedRect(QRectF(16, 13, 4, 6), 1, 1)

        elif self.icon_type == "single_robot":
            painter.drawRoundedRect(QRectF(5, 9, 16, 10), 3, 3)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(9, 13, 2.7, 2.7))
            painter.drawEllipse(QRectF(16, 13, 2.7, 2.7))

        elif self.icon_type == "multi_robot":
            painter.drawRoundedRect(QRectF(4, 6, 10, 8), 2, 2)
            painter.drawRoundedRect(QRectF(13, 14, 10, 8), 2, 2)
            painter.drawLine(QPointF(14, 10), QPointF(18, 14))


# ============================================================
# CUSTOM TOP BAR
# ============================================================

class TopBar(QFrame):
    def __init__(self, window: QMainWindow):
        super().__init__()
        self.window = window
        self.drag_position = None

        self.setObjectName("topBar")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 14, 0)
        layout.setSpacing(12)

        self.lab_icon = QLabel()
        self.lab_icon.setPixmap(make_icon("single_robot", "white").pixmap(24, 24))

        self.title = QLabel("Robotics Simulation Lab")
        self.title.setObjectName("topTitle")

        self.status = QLabel("●  Ready to run")
        self.status.setObjectName("statusReady")

        self.mode_icon = VectorIcon("single_robot")

        self.mode_selector = QComboBox()
        self.mode_selector.setObjectName("topModeSelector")
        self.mode_selector.addItems(["Single Robot Mode", "Multiple Robot Mode"])
        self.mode_selector.view().setStyleSheet(f"""
            QListView {{
                background-color: #FFFFFF;
                color: #111827;
                border: 1px solid {BORDER};
                selection-background-color: #F4EAEA;
                selection-color: {MAROON};
            }}
            QListView::item {{
                min-height: 28px;
                padding: 6px 8px;
                color: #111827;
                background-color: #FFFFFF;
            }}
            QListView::item:selected {{
                color: {MAROON};
                background-color: #F4EAEA;
            }}
        """)
        self.mode_selector.setFixedWidth(172)
        self.mode_selector.currentTextChanged.connect(self.update_mode_icon)
        self.mode_selector.setVisible(False)

        self.single_mode_button = QPushButton("Single")
        self.single_mode_button.setObjectName("modeSegmentButton")
        self.single_mode_button.setCheckable(True)
        self.single_mode_button.setFixedSize(74, 30)
        self.single_mode_button.clicked.connect(lambda: self.set_agent_mode("Single Robot Mode"))

        self.multi_mode_button = QPushButton("Multiple")
        self.multi_mode_button.setObjectName("modeSegmentButton")
        self.multi_mode_button.setCheckable(True)
        self.multi_mode_button.setFixedSize(86, 30)
        self.multi_mode_button.clicked.connect(lambda: self.set_agent_mode("Multiple Robot Mode"))

        self.editor_button = QPushButton("Editor")
        self.editor_button.setObjectName("modeSegmentButton")
        self.editor_button.setCheckable(True)
        self.editor_button.setFixedSize(78, 30)
        self.editor_button.clicked.connect(self.window.toggle_editor_mode_from_button)
        self.update_mode_icon(self.mode_selector.currentText())

        self.gear_button = QPushButton()
        self.gear_button.setObjectName("topIconButton")
        self.gear_button.setIcon(make_icon("gear", "white"))
        self.gear_button.setIconSize(QSize(20, 20))
        self.gear_button.setFixedSize(34, 32)

        self.min_button = QPushButton()
        self.min_button.setObjectName("windowButton")
        self.min_button.setIcon(make_icon("minimize", "white"))
        self.min_button.setIconSize(QSize(18, 18))
        self.min_button.setFixedSize(34, 32)
        self.min_button.clicked.connect(self.window.showMinimized)

        self.max_button = QPushButton()
        self.max_button.setObjectName("windowButton")
        self.max_button.setIcon(make_icon("maximize", "white"))
        self.max_button.setIconSize(QSize(18, 18))
        self.max_button.setFixedSize(34, 32)
        self.max_button.clicked.connect(self.toggle_max_restore)

        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.setIcon(make_icon("close", "white"))
        self.close_button.setIconSize(QSize(18, 18))
        self.close_button.setFixedSize(34, 32)
        self.close_button.clicked.connect(self.window.close)

        layout.addWidget(self.lab_icon)
        layout.addWidget(self.title)
        layout.addStretch()
        layout.addWidget(self.status)
        layout.addSpacing(44)
        layout.addWidget(self.mode_icon)
        layout.addWidget(self.single_mode_button)
        layout.addWidget(self.multi_mode_button)
        layout.addWidget(self.editor_button)
        layout.addSpacing(12)
        layout.addWidget(self.gear_button)
        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)

    def set_agent_mode(self, mode: str):
        if mode not in ("Single Robot Mode", "Multiple Robot Mode"):
            return
        if self.mode_selector.currentText() != mode:
            self.mode_selector.setCurrentText(mode)
        else:
            self.update_mode_icon(mode)

    def update_mode_icon(self, text: str):
        is_multiple = "Multiple" in text
        if is_multiple:
            self.mode_icon.set_icon_type("multi_robot")
        else:
            self.mode_icon.set_icon_type("single_robot")

        if hasattr(self, "single_mode_button"):
            self.single_mode_button.blockSignals(True)
            self.multi_mode_button.blockSignals(True)
            self.single_mode_button.setChecked(not is_multiple)
            self.multi_mode_button.setChecked(is_multiple)
            self.single_mode_button.blockSignals(False)
            self.multi_mode_button.blockSignals(False)

        if hasattr(self, "editor_button"):
            self.editor_button.blockSignals(True)
            self.editor_button.setChecked(getattr(self.window, "editor_mode", False))
            self.editor_button.blockSignals(False)

    def set_status(self, state: str):
        if state == "running":
            self.status.setText("●  Running")
            self.status.setObjectName("statusRunning")
        elif state == "paused":
            self.status.setText("●  Paused")
            self.status.setObjectName("statusPaused")
        else:
            self.status.setText("●  Ready to run")
            self.status.setObjectName("statusReady")

        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def toggle_max_restore(self):
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()

    def multi_robot_screen_positions(self) -> list[tuple[int, float, float, RobotStartConfig]]:
        if "Multiple" not in self.config.agent_mode:
            return []

        robots = normalized_robot_start_configs(self.config)
        positions: list[tuple[int, float, float, RobotStartConfig]] = []
        for index, robot_cfg in enumerate(robots):
            sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
            positions.append((index, sx, sy, robot_cfg))
        return positions

    def robot_index_at_screen_position(self, sx: float, sy: float) -> tuple[int, RobotStartConfig] | None:
        if self.robot is not None:
            return None

        px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)
        body_px = max(7.0, float(self.config.body_radius) * px_per_meter)
        hit_radius = max(13.0, body_px + 5.0)

        # Reverse order so the visually topmost/highest-index robot is easier to pick.
        for index, rx, ry, robot_cfg in reversed(self.multi_robot_screen_positions()):
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return index, robot_cfg

        # Single-robot preview is draggable too. Use index -1 to indicate the
        # global single robot initial pose.
        if "Multiple" not in self.config.agent_mode:
            rx, ry = self.world_to_screen(self.config.x, self.config.y)
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return -1, RobotStartConfig(
                    float(self.config.x),
                    float(self.config.y),
                    float(self.config.theta),
                    float(self.config.v),
                )

        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if (
            event.buttons() & Qt.LeftButton
            and self.drag_position is not None
            and not self.window.isMaximized()
        ):
            self.window.move(event.globalPosition().toPoint() - self.drag_position)

    def mouseReleaseEvent(self, event):
        self.drag_position = None


# ============================================================
# HERO HEADER
# ============================================================

class HeroHeader(QWidget):
    """
    Visual header for the side panel.

    This widget only draws the TAMU image/background and header labels.
    It does not contain simulation logic.
    """

    def __init__(self, image_path: str | None = None):
        super().__init__()
        self.setFixedHeight(142)

        self.image_path = image_path
        self.pixmap = QPixmap(image_path) if image_path else QPixmap()

        if image_path:
            print(
                f"HEADER IMAGE LOAD: {image_path} | isNull={self.pixmap.isNull()}",
                flush=True,
            )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(self.rect())

        rounded_path = QPainterPath()
        rounded_path.addRoundedRect(rect, 14, 14)

        # ----------------------------------------------------
        # 1. Background layer
        # ----------------------------------------------------
        painter.fillPath(rounded_path, QColor(MAROON))

        # ----------------------------------------------------
        # 2. Image layer, clipped to rounded shape
        # ----------------------------------------------------
        painter.save()
        painter.setClipPath(rounded_path)

        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )

            crop_x = max(0, (scaled.width() - self.width()) // 2)
            crop_y = max(0, (scaled.height() - self.height()) // 2)

            # Draw the image as a subtle background layer so the maroon
            # identity remains dominant.
            painter.setOpacity(0.42)
            painter.drawPixmap(
                0,
                0,
                scaled,
                crop_x,
                crop_y,
                self.width(),
                self.height(),
            )

            painter.setOpacity(1.0)

            # Maroon overlay improves text readability and keeps the header
            # visually aligned with the TAMU theme.
            painter.fillPath(rounded_path, QColor(80, 0, 0, 185))

        else:
            painter.setOpacity(1.0)
            painter.setPen(QPen(QColor(130, 40, 45, 140), 2))

            for i in range(9):
                x1 = self.width() - 250 + i * 34
                painter.drawLine(x1, self.height(), x1 + 135, 18)

        painter.restore()

        # ----------------------------------------------------
        # 3. Text layer.
        # Important:
        # Draw text after painter.restore(), so it is not clipped
        # or affected by image opacity.
        # ----------------------------------------------------
        painter.save()
        painter.setClipping(False)
        painter.setOpacity(1.0)

        title_font = QFont("Segoe UI", 20, QFont.Bold)
        subtitle_font = QFont("Segoe UI", 10, QFont.Bold)
        badge_font = QFont("Segoe UI", 8, QFont.Bold)

        painter.setFont(title_font)
        painter.setPen(QPen(QColor("#FFFFFF")))

        painter.drawText(
            QRectF(22, 15, self.width() - 44, 30),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Texas A&M Robotics",
        )

        painter.drawText(
            QRectF(22, 45, self.width() - 44, 30),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Simulation Studio",
        )

        painter.setFont(subtitle_font)
        painter.setPen(QPen(QColor(245, 216, 216)))

        painter.drawText(
            QRectF(23, 80, self.width() - 46, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Pre-Simulation Configuration",
        )

        # ----------------------------------------------------
        # 4. Badge layer
        # ----------------------------------------------------
        badge_rect = QRectF(22, 108, 150, 24)

        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 12, 12)

        painter.fillPath(badge_path, QColor(55, 87, 45, 235))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(GREEN))
        painter.drawEllipse(
            QRectF(
                badge_rect.left() + 8,
                badge_rect.top() + 6,
                12,
                12,
            )
        )

        painter.setPen(QPen(QColor("white"), 1.7, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(
            badge_rect.left() + 11,
            badge_rect.top() + 12,
            badge_rect.left() + 13,
            badge_rect.top() + 15,
        )
        painter.drawLine(
            badge_rect.left() + 13,
            badge_rect.top() + 15,
            badge_rect.left() + 18,
            badge_rect.top() + 9,
        )

        painter.setFont(badge_font)
        painter.setPen(QPen(QColor("white")))

        painter.drawText(
            badge_rect.adjusted(27, 0, 0, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            "Validated parameters",
        )

        painter.restore()

# ============================================================
# SECTION CARD
# ============================================================

class SectionCard(QFrame):
    def __init__(self, icon_type: str, title: str):
        super().__init__()
        self.setObjectName("sectionCard")

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(16, 14, 16, 16)
        self.root.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        icon = VectorIcon(icon_type)

        label = QLabel(title)
        label.setObjectName("sectionTitle")

        title_row.addWidget(icon)
        title_row.addWidget(label)
        title_row.addStretch()

        self.root.addLayout(title_row)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(
            f"background-color: {BORDER_SOFT}; max-height: 1px; border: none;"
        )

        self.root.addWidget(divider)


# ============================================================
# NUMERIC CONTROLS: NEW APPROACH
# ============================================================

class NumericStepper(QWidget):
    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int = 2,
    ):
        super().__init__()

        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.decimals = decimals
        self._value = float(value)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setObjectName("fieldLabel")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.minus_button = QPushButton("−")
        self.minus_button.setObjectName("stepperButton")
        self.minus_button.setFixedWidth(24)

        self.input = QLineEdit()
        self.input.setObjectName("numericInput")
        self.input.setAlignment(Qt.AlignCenter)
        self.input.setValidator(QDoubleValidator(self.minimum, self.maximum, self.decimals, self))
        self.input.setText(self.format_value(self._value))

        self.plus_button = QPushButton("+")
        self.plus_button.setObjectName("stepperButton")
        self.plus_button.setFixedWidth(24)

        row.addWidget(self.minus_button)
        row.addWidget(self.input, 1)
        row.addWidget(self.plus_button)

        layout.addWidget(self.label)
        layout.addLayout(row)

        self.minus_button.clicked.connect(lambda: self.setValue(self.value() - self.step))
        self.plus_button.clicked.connect(lambda: self.setValue(self.value() + self.step))
        self.input.editingFinished.connect(self.commit_text)

    def format_value(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"

    def commit_text(self):
        try:
            value = float(self.input.text())
        except ValueError:
            value = self._value

        self.setValue(value)

    def value(self) -> float:
        self.commit_text_silent()
        return self._value

    def commit_text_silent(self):
        try:
            value = float(self.input.text())
        except ValueError:
            value = self._value

        value = clamp(value, self.minimum, self.maximum)
        self._value = value
        self.input.setText(self.format_value(value))

    def setValue(self, value: float):
        value = clamp(float(value), self.minimum, self.maximum)

        if abs(value - self._value) > 1e-12:
            self._value = value
            self.input.setText(self.format_value(value))
            self.valueChanged.emit(self._value)
        else:
            self.input.setText(self.format_value(value))


class SliderValueRow(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label: str, value: float, minimum: float, maximum: float):
        super().__init__()

        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self._value = float(value)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.label = QLabel(label)
        self.label.setObjectName("fieldLabel")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self.value_to_slider(self._value))

        self.input = QLineEdit()
        self.input.setObjectName("smallNumericInput")
        self.input.setAlignment(Qt.AlignCenter)
        self.input.setValidator(QDoubleValidator(self.minimum, self.maximum, 2, self))
        self.input.setText(f"{self._value:.2f}")
        self.input.setFixedWidth(58)

        row.addWidget(self.slider, 1)
        row.addWidget(self.input)

        layout.addWidget(self.label)
        layout.addLayout(row)

        self.slider.valueChanged.connect(self.on_slider_changed)
        self.input.editingFinished.connect(self.on_text_changed)

    def value_to_slider(self, value: float) -> int:
        ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return int(clamp(ratio, 0.0, 1.0) * 1000)

    def slider_to_value(self, slider_value: int) -> float:
        return self.minimum + (slider_value / 1000.0) * (self.maximum - self.minimum)

    def _value_from_text(self) -> float:
        try:
            value = float(self.input.text())
        except ValueError:
            value = self._value

        return clamp(value, self.minimum, self.maximum)

    def _sync_widgets_silent(self, value: float) -> None:
        """
        Keep the slider and text box visually synchronized without emitting
        valueChanged.

        This is important because read_config() calls value(), and reading
        configuration must not trigger update_preview() recursively.
        """
        blocked_slider = self.slider.blockSignals(True)
        self.slider.setValue(self.value_to_slider(value))
        self.slider.blockSignals(blocked_slider)

        blocked_input = self.input.blockSignals(True)
        self.input.setText(f"{value:.2f}")
        self.input.blockSignals(blocked_input)

    def on_slider_changed(self, slider_value: int):
        """
        User moved the slider, so this is a real value change.
        It is allowed to emit valueChanged.
        """
        value = self.slider_to_value(slider_value)
        changed = abs(value - self._value) > 1e-12

        self._value = value

        blocked_input = self.input.blockSignals(True)
        self.input.setText(f"{self._value:.2f}")
        self.input.blockSignals(blocked_input)

        if changed:
            self.valueChanged.emit(self._value)

    def on_text_changed(self):
        """
        User finished editing the text box, so this is a real value change.
        It is allowed to emit valueChanged only if the value actually changed.
        """
        self.setValue(self._value_from_text())

    def value(self) -> float:
        """
        Read the current value.

        Reading must be silent:
            no valueChanged emission
            no update_preview recursion
        """
        value = self._value_from_text()
        self._value = value
        self._sync_widgets_silent(value)
        return self._value

    def setValue(self, value: float):
        """
        Programmatically set the value.

        This emits valueChanged only when the value actually changes.
        """
        value = clamp(float(value), self.minimum, self.maximum)
        changed = abs(value - self._value) > 1e-12

        self._value = value
        self._sync_widgets_silent(value)

        if changed:
            self.valueChanged.emit(self._value)

class ToggleSwitch(QPushButton):
    """
    Small boolean switch used instead of On/Off combo boxes.

    The switch is intentionally visual-only; it still exposes the normal
    QPushButton checked/toggled API, so configuration code can read isChecked().
    """

    def __init__(self, checked: bool = False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(bool(checked))
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(48, 26)
        self.toggled.connect(self.update)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        checked = self.isChecked()
        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)

        if checked:
            track_color = QColor(MAROON)
            knob_color = QColor("white")
            text_color = QColor("white")
            label = "ON"
        else:
            track_color = QColor(218, 223, 231)
            knob_color = QColor("white")
            text_color = QColor(90, 96, 106)
            label = "OFF"

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(rect, 13, 13)

        knob_diameter = 20
        knob_x = self.width() - knob_diameter - 4 if checked else 4
        knob_y = (self.height() - knob_diameter) / 2

        painter.setBrush(QBrush(knob_color))
        painter.drawEllipse(QRectF(knob_x, knob_y, knob_diameter, knob_diameter))

        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        painter.setPen(QPen(text_color))

        if checked:
            text_rect = QRectF(5, 0, 22, self.height())
        else:
            text_rect = QRectF(20, 0, 25, self.height())

        painter.drawText(text_rect, Qt.AlignCenter, label)




class SimulationMetricsWindow(QWidget):
    """
    Live metrics dashboard.

    The table is kept for compact scalar metrics. Long reasoning/status text is
    shown below the table so it is no longer squeezed into a one-line cell.
    """

    def __init__(self, owner):
        super().__init__(None)
        self.owner = owner
        self.setWindowTitle("Simulation Metrics")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
        )
        self.setStyleSheet(owner.stylesheet())
        self.resize(760, 640)
        self.setMinimumSize(640, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Simulation Metrics")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.ElideNone)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(180)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 300)
        layout.addWidget(self.table, 1)

        message_title = QLabel("Latest decision / status")
        message_title.setObjectName("fieldLabel")
        layout.addWidget(message_title)

        self.message_box = QLabel("--")
        self.message_box.setObjectName("metricsMessageBox")
        self.message_box.setWordWrap(True)
        self.message_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.message_box.setMinimumHeight(58)
        self.message_box.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.message_box, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.table.setColumnWidth(0, max(260, min(360, int(self.width() * 0.42))))

    def refresh(self) -> None:
        raw_metrics = self.owner.get_metrics_snapshot()
        metrics = [
            (name, value)
            for name, value in raw_metrics
            if str(name) not in {"Last goal-selection reason", "Last decision / status"}
        ]

        self.table.setRowCount(len(metrics))
        for row, (name, value) in enumerate(metrics):
            name_item = QTableWidgetItem(str(name))
            value_item = QTableWidgetItem(str(value))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)

        latest = "--"
        if hasattr(self.owner, "latest_decision_message"):
            latest = self.owner.latest_decision_message()
        elif hasattr(self.owner, "last_goal_selection_reason"):
            latest = str(self.owner.last_goal_selection_reason)
        self.message_box.setText(str(latest) if latest else "--")
        self.table.resizeRowsToContents()

    def closeEvent(self, event):
        self.timer.stop()
        if getattr(self.owner, "metrics_window", None) is self:
            self.owner.metrics_window = None
        super().closeEvent(event)


class SimulationConsoleWindow(QWidget):
    """Movable window with the full simulator status log.

    The console keeps the raw chronological log, but also provides filters and
    a compact diagnostics copy so debugging does not depend on manually reading
    hundreds of movement lines.
    """

    def __init__(self, owner):
        super().__init__(None)
        self.owner = owner
        self.setWindowTitle("Simulation Console")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
        )
        self.setStyleSheet(owner.stylesheet())
        self.resize(980, 520)
        self.setMinimumSize(760, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("Simulation Console")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()

        self.filter_combo = QComboBox()
        self.filter_combo.setObjectName("compactCombo")
        self.filter_combo.addItems([
            "All",
            "Config",
            "Decisions / routes",
            "Safety / warnings",
            "Mapping",
            "Movement",
        ])
        self.filter_combo.setFixedWidth(170)
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        title_row.addWidget(self.filter_combo)

        self.copy_selected_button = QPushButton("Copy selected")
        self.copy_selected_button.setObjectName("secondaryButton")
        self.copy_selected_button.clicked.connect(self.copy_selected)
        title_row.addWidget(self.copy_selected_button)

        self.copy_visible_button = QPushButton("Copy visible")
        self.copy_visible_button.setObjectName("secondaryButton")
        self.copy_visible_button.clicked.connect(self.copy_visible)
        title_row.addWidget(self.copy_visible_button)

        self.copy_all_button = QPushButton("Copy all")
        self.copy_all_button.setObjectName("secondaryButton")
        self.copy_all_button.clicked.connect(self.copy_all)
        title_row.addWidget(self.copy_all_button)

        self.copy_diagnostics_button = QPushButton("Copy diagnostics")
        self.copy_diagnostics_button.setObjectName("secondaryButton")
        self.copy_diagnostics_button.clicked.connect(self.copy_diagnostics)
        title_row.addWidget(self.copy_diagnostics_button)

        self.pause_button = QPushButton("Pause log")
        self.pause_button.setObjectName("secondaryButton")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.on_pause_toggled)
        title_row.addWidget(self.pause_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_console)
        title_row.addWidget(self.clear_button)
        layout.addLayout(title_row)

        self.console = QPlainTextEdit()
        self.console.setObjectName("consoleText")
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.console.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(self.console, 1)

        hint = QLabel(
            "Use filters for reading. Use Copy diagnostics for a compact bug report; Copy all still copies the raw full log."
        )
        hint.setObjectName("fieldHelp")
        layout.addWidget(hint)

        self._last_rendered_text = ""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()

    def _lines(self) -> list[str]:
        if hasattr(self.owner, "get_console_lines"):
            return list(self.owner.get_console_lines())
        canvas = getattr(self.owner, "canvas", None)
        if canvas is not None and hasattr(canvas, "status_history_lines"):
            return list(canvas.status_history_lines())
        message = getattr(canvas, "status_message", "") if canvas is not None else ""
        return [str(message)] if message else []

    def _matches_filter(self, line: str, filter_name: str) -> bool:
        text = str(line)
        low = text.lower()
        if filter_name == "All":
            return True
        if filter_name == "Config":
            return (
                "=== simulation started ===" in low
                or low.startswith("[") and any(key in low for key in [
                    "mode:", "planner:", "path simplifier:", "exploration planner:",
                    "multi-robot coordinator:", "vision model:", "sensor range:",
                    "grid resolution:", "goal g:", "robot count:", "same robot",
                    "r1 start:", "r2 start:", "r3 start:", "safety radius",
                    "ipp λ", "obstacles in scenario",
                ])
            )
        if filter_name == "Decisions / routes":
            return any(key in low for key in [
                "route assigned", "frontier assigned", "holding position",
                "no valid frontier", "exploration target reached", "goal adjusted",
                "path found", "replanned", "keeping assigned frontier",
                "state=active", "state=hold_no_frontier", "state=stuck_safety", "state=escape_local",
            ])
        if filter_name == "Safety / warnings":
            return any(key in low for key in [
                "collision", "robot obstacle", "safety", "blocked", "start cell",
                "predicted collision", "crosses", "ignored", "failed", "unavailable",
                "no valid", "deadlock", "state=stuck_safety",
            ])
        if filter_name == "Mapping":
            return any(key in low for key in [
                "mapping", "mapped", "obstacle boundary", "boundary sample", "belief",
                "occupied belief", "new boundary",
            ])
        if filter_name == "Movement":
            return " move @ " in low or " pos=(" in low
        return True

    def _filtered_lines(self) -> list[str]:
        filter_name = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"
        return [line for line in self._lines() if self._matches_filter(line, filter_name)]

    def refresh(self) -> None:
        if self.pause_button.isChecked():
            return

        text = "\n".join(self._filtered_lines())
        if text == self._last_rendered_text:
            return

        # Do not destroy an active text selection while the user is copying from
        # the console. The user can press Pause log for longer inspections.
        cursor = self.console.textCursor()
        if self.console.hasFocus() and cursor.hasSelection():
            return

        previous_scroll = self.console.verticalScrollBar().value()
        at_bottom = previous_scroll >= self.console.verticalScrollBar().maximum() - 4
        self.console.setPlainText(text)
        self._last_rendered_text = text
        if at_bottom:
            self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def on_filter_changed(self, *_):
        self._last_rendered_text = ""
        self.refresh()

    def copy_selected(self) -> None:
        selected = self.console.textCursor().selectedText().replace("\u2029", "\n")
        if selected:
            QApplication.clipboard().setText(selected)
        else:
            self.copy_visible()

    def copy_visible(self) -> None:
        QApplication.clipboard().setText("\n".join(self._filtered_lines()))

    def copy_all(self) -> None:
        QApplication.clipboard().setText("\n".join(self._lines()))

    def copy_diagnostics(self) -> None:
        QApplication.clipboard().setText(self.build_diagnostics_summary())

    def build_diagnostics_summary(self) -> str:
        lines = self._lines()
        lower_lines = [line.lower() for line in lines]

        def count_contains(*needles: str) -> int:
            return sum(1 for low in lower_lines if any(n in low for n in needles))

        def robot_count(robot_label: str, *needles: str) -> int:
            prefix = f"] {robot_label.lower()}"
            return sum(
                1 for line, low in zip(lines, lower_lines)
                if prefix in low and any(n in low for n in needles)
            )

        config_lines = []
        in_config = False
        for line in lines:
            low = line.lower()
            if "=== simulation started ===" in low:
                in_config = True
            if in_config:
                if " route assigned" in low or " move @ " in low:
                    break
                config_lines.append(line)

        robots = []
        for label in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
            routes = robot_count(label, "route assigned")
            moves = robot_count(label, "move @")
            holds = robot_count(label, "holding position", "no valid frontier", "state=hold_no_frontier")
            stuck = robot_count(label, "state=stuck_safety")
            active = robot_count(label, "state=active")
            safety = robot_count(label, "robot obstacle", "predicted collision", "blocked", "collision")
            if routes or moves or holds or stuck or active or safety:
                robots.append(
                    f"{label}: routes={routes}, moves={moves}, state-active={active}, "
                    f"holds/no-frontier={holds}, stuck-safety={stuck}, safety-events={safety}"
                )

        important = [
            line for line in lines
            if self._matches_filter(line, "Safety / warnings") or self._matches_filter(line, "Decisions / routes")
        ]
        tail = important[-80:]

        summary = [
            "=== DIAGNOSTIC SUMMARY ===",
            f"Total console lines: {len(lines)}",
            f"Route assignments: {count_contains('route assigned')}",
            f"Movement logs: {count_contains(' move @ ')}",
            f"Mapping events: {count_contains('mapping', 'mapped', 'boundary sample')}",
            f"Safety / warning events: {count_contains('collision', 'robot obstacle', 'predicted collision', 'blocked', 'start cell', 'no valid frontier')}",
            f"Route-state transitions: {count_contains('state=active', 'state=hold_no_frontier', 'state=stuck_safety', 'state=escape_local')}",
            f"Collision-after-update events: {count_contains('collision:')}",
            "",
            "--- RUN CONFIG ---",
        ]
        summary.extend(config_lines[-40:] if config_lines else ["No simulation-start config block found."])
        summary.extend(["", "--- PER-ROBOT COUNTS ---"])
        summary.extend(robots if robots else ["No per-robot activity found."])
        summary.extend(["", "--- LAST IMPORTANT EVENTS ---"])
        summary.extend(tail if tail else ["No important events found."])
        return "\n".join(summary)

    def on_pause_toggled(self, checked: bool) -> None:
        self.pause_button.setText("Resume log" if checked else "Pause log")
        if not checked:
            self.refresh()

    def clear_console(self) -> None:
        if hasattr(self.owner, "clear_console_messages"):
            self.owner.clear_console_messages()
        else:
            canvas = getattr(self.owner, "canvas", None)
            if canvas is not None and hasattr(canvas, "clear_status_history"):
                canvas.clear_status_history()
        self._last_rendered_text = ""
        self.refresh()

    def closeEvent(self, event):
        self.timer.stop()
        if getattr(self.owner, "console_window", None) is self:
            self.owner.console_window = None
        super().closeEvent(event)

# ============================================================
# MAIN WINDOW
# ============================================================
