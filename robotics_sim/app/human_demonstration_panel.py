"""Presentation-only Human Demonstration side panel.

This widget never touches the filesystem, never computes a frontier
candidate, and never calls a planner -- it only renders whatever state
main_window.py pushes into it (via the ``set_*`` methods) and emits
signals when the user interacts with a control. All actual behavior
(loading a .sim file, freezing a candidate pool, writing an episode
folder, ...) lives in ``robotics_sim.simulation.human_demonstration_
runtime.HumanDemonstrationRuntime``; MainWindow is the only thing that
connects this panel's signals to that runtime.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HumanDemonstrationPanel(QFrame):
    """Emits one signal per user action; renders whatever the host tells
    it to via the ``set_*`` methods. Holds no domain state of its own
    beyond what is currently displayed."""

    collectorSelected = Signal(str)
    mapSelected = Signal(str)
    episodeSelected = Signal(int)
    previousEpisodeRequested = Signal()
    nextEpisodeRequested = Signal()
    loadEpisodeRequested = Signal()
    finishEpisodeRequested = Signal()
    abortEpisodeRequested = Signal()
    closeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("humanDemonstrationPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Human Demonstration")
        title.setObjectName("humanDemonstrationTitle")
        header.addWidget(title, 1)
        self.close_button = QPushButton("✕")
        self.close_button.setObjectName("humanDemonstrationCloseButton")
        self.close_button.setFixedWidth(28)
        self.close_button.clicked.connect(self.closeRequested.emit)
        header.addWidget(self.close_button, 0)
        root.addLayout(header)

        # -- Collector / map / episode selection ------------------------
        self.collector_combo = QComboBox()
        self.collector_combo.setObjectName("humanDemoCollectorCombo")
        self.collector_combo.currentTextChanged.connect(self._on_collector_changed)
        root.addWidget(self._labeled("Collector", self.collector_combo))

        self.map_combo = QComboBox()
        self.map_combo.setObjectName("humanDemoMapCombo")
        self.map_combo.currentTextChanged.connect(self._on_map_changed)
        root.addWidget(self._labeled("Map", self.map_combo))

        episode_row = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previousEpisodeRequested.emit)
        self.episode_combo = QComboBox()
        self.episode_combo.setObjectName("humanDemoEpisodeCombo")
        self.episode_combo.currentTextChanged.connect(self._on_episode_changed)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.nextEpisodeRequested.emit)
        episode_row.addWidget(self.previous_button)
        episode_row.addWidget(self.episode_combo, 1)
        episode_row.addWidget(self.next_button)
        root.addLayout(episode_row)

        # -- Read-only status labels --------------------------------------
        self.episode_position_label = QLabel("Episode 0 of 0")
        self.episode_position_label.setObjectName("humanDemoEpisodePositionLabel")
        root.addWidget(self.episode_position_label)

        self.scenario_seed_label = QLabel("Scenario: -- / Seed: --")
        self.scenario_seed_label.setObjectName("humanDemoScenarioSeedLabel")
        root.addWidget(self.scenario_seed_label)

        self.recorded_progress_label = QLabel("Recorded 0 of 0")
        self.recorded_progress_label.setObjectName("humanDemoRecordedProgressLabel")
        root.addWidget(self.recorded_progress_label)

        self.accepted_progress_label = QLabel("Accepted 0 of 0")
        self.accepted_progress_label.setObjectName("humanDemoAcceptedProgressLabel")
        root.addWidget(self.accepted_progress_label)

        self.map_complete_label = QLabel("")
        self.map_complete_label.setObjectName("humanDemoMapCompleteLabel")
        self.map_complete_label.setVisible(False)
        root.addWidget(self.map_complete_label)

        # -- Load / Finish / Abort ----------------------------------------
        self.load_button = QPushButton("Load Episode")
        self.load_button.setObjectName("humanDemoLoadButton")
        self.load_button.clicked.connect(self.loadEpisodeRequested.emit)
        root.addWidget(self.load_button)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setObjectName("humanDemoStatusLabel")
        root.addWidget(self.status_label)

        self.fires_loaded_label = QLabel("Fires loaded: --")
        self.fires_loaded_label.setObjectName("humanDemoFiresLoadedLabel")
        root.addWidget(self.fires_loaded_label)

        self.selected_robot_label = QLabel("Selected robot: --")
        self.selected_robot_label.setObjectName("humanDemoSelectedRobotLabel")
        root.addWidget(self.selected_robot_label)

        self.pending_robots_label = QLabel("Pending robots: --")
        self.pending_robots_label.setObjectName("humanDemoPendingRobotsLabel")
        root.addWidget(self.pending_robots_label)

        finish_abort_row = QHBoxLayout()
        self.finish_button = QPushButton("Finish Episode && Save")
        self.finish_button.setObjectName("humanDemoFinishButton")
        self.finish_button.clicked.connect(self.finishEpisodeRequested.emit)
        self.abort_button = QPushButton("Abort Episode")
        self.abort_button.setObjectName("humanDemoAbortButton")
        self.abort_button.clicked.connect(self.abortEpisodeRequested.emit)
        finish_abort_row.addWidget(self.finish_button)
        finish_abort_row.addWidget(self.abort_button)
        root.addLayout(finish_abort_row)

        self.last_saved_path_label = QLabel("")
        self.last_saved_path_label.setObjectName("humanDemoLastSavedPathLabel")
        self.last_saved_path_label.setWordWrap(True)
        root.addWidget(self.last_saved_path_label)

        root.addStretch()

        self._set_episode_controls_enabled(True)
        self.set_finish_enabled(False)
        self.set_abort_enabled(False)

    @staticmethod
    def _labeled(text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        label = QLabel(text)
        label.setObjectName("humanDemoFieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return box

    # -- internal combo change relays -----------------------------------

    def _on_collector_changed(self, text: str) -> None:
        if text:
            self.collectorSelected.emit(text)

    def _on_map_changed(self, text: str) -> None:
        if text:
            self.mapSelected.emit(text)

    def _on_episode_changed(self, text: str) -> None:
        if text:
            try:
                self.episodeSelected.emit(int(text))
            except ValueError:
                pass

    # -- host -> panel rendering (no logic, no filesystem, no planner) --

    def set_collector_options(self, collector_ids: list[str], *, current: str | None = None) -> None:
        self.collector_combo.blockSignals(True)
        self.collector_combo.clear()
        self.collector_combo.addItems(list(collector_ids))
        if current is not None:
            self.collector_combo.setCurrentText(current)
        self.collector_combo.blockSignals(False)

    def set_map_options(self, map_ids: list[str], *, current: str | None = None) -> None:
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        self.map_combo.addItems(list(map_ids))
        if current is not None:
            self.map_combo.setCurrentText(current)
        self.map_combo.blockSignals(False)

    def set_episode_options(self, episode_numbers: list[int], *, current: int | None = None) -> None:
        self.episode_combo.blockSignals(True)
        self.episode_combo.clear()
        self.episode_combo.addItems([str(n) for n in episode_numbers])
        if current is not None:
            self.episode_combo.setCurrentText(str(current))
        self.episode_combo.blockSignals(False)

    def set_scenario_and_seed(self, scenario_id: str, seed: int) -> None:
        self.scenario_seed_label.setText(f"Scenario: {scenario_id} / Seed: {seed}")

    def set_episode_position_text(self, text: str) -> None:
        self.episode_position_label.setText(text)

    def set_recorded_progress_text(self, text: str) -> None:
        self.recorded_progress_label.setText(text)

    def set_accepted_progress_text(self, text: str) -> None:
        self.accepted_progress_label.setText(text)

    def set_map_complete_text(self, text: str | None) -> None:
        self.map_complete_label.setVisible(text is not None)
        self.map_complete_label.setText(text or "")

    def set_status_text(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def set_fires_loaded_count(self, count: int | None) -> None:
        self.fires_loaded_label.setText(f"Fires loaded: {count if count is not None else '--'}")

    def set_selected_robot_text(self, text: str | None) -> None:
        self.selected_robot_label.setText(f"Selected robot: {text or '--'}")

    def set_pending_robots_text(self, text: str | None) -> None:
        self.pending_robots_label.setText(f"Pending robots: {text or '--'}")

    def set_last_saved_path(self, path: str | None) -> None:
        self.last_saved_path_label.setText(f"Saved: {path}" if path else "")

    def _set_episode_controls_enabled(self, enabled: bool) -> None:
        """Collector/map/episode selectors and Load are only editable
        while no episode is active (Fase 3: bloquear cambios durante un
        episodio activo -- Finish/Abort is the only way out)."""

        self.collector_combo.setEnabled(enabled)
        self.map_combo.setEnabled(enabled)
        self.episode_combo.setEnabled(enabled)
        self.previous_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

    def set_episode_active(self, active: bool) -> None:
        self._set_episode_controls_enabled(not active)
        self.load_button.setEnabled(not active)

    def set_load_enabled(self, enabled: bool) -> None:
        self.load_button.setEnabled(enabled)

    def set_finish_enabled(self, enabled: bool) -> None:
        self.finish_button.setEnabled(enabled)

    def set_abort_enabled(self, enabled: bool) -> None:
        self.abort_button.setEnabled(enabled)
