"""
Simulation canvas and rendering logic.

This module draws the current simulator snapshot: grid, obstacles, mapped
points, explored area, robots, FoV/LiDAR, routes, and frontiers. Runtime
controls are real child widgets registered by MainWindow.
It emits interaction events, but it does not choose frontiers or compute routes.
"""

from __future__ import annotations

import math
import os
import time
import zlib
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QPixmap,
    QImage,
)
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget

from robotics_sim.simulation.config import *
from robotics_sim.simulation.navigation_modes import is_goal_seeking_planner
from robotics_sim.app.theme import ThemeMode, theme_colors
from robotics_sim.app.map_editor import (
    MIN_EDITOR_OBSTACLE_SIZE,
    connected_obstacle_indices,
    find_obstacle_group_at,
    remove_obstacle_at,
)
from robotics_sim.app.render_perf import (
    PerfGuiWarningGate,
    RenderDetailLogger,
    RenderPerfMonitor,
    format_gui_perf_warning,
)

# Soft cap on how many occupancy cells the grid overlay will color-fill in a
# single cache rebuild. Above this, per-cell coloring is skipped for that
# rebuild (grid lines are still drawn) so a small grid_resolution over a
# large visible area can never freeze the UI trying to draw every cell.
MAX_GRID_OVERLAY_CELLS = 20000

# Radii (px) of planned-route markers: the small numbered waypoint dots,
# the active/current waypoint (larger so it still stands out, but not so
# large it dominates the route), and the S (start) / F,G (frontier or
# final-goal endpoint) markers. Purely visual -- none of these affect
# waypoint coordinates, route geometry, or planning in any way.
WAYPOINT_MARKER_RADIUS = 4
MULTI_ROBOT_WAYPOINT_MARKER_RADIUS = 3
ACTIVE_WAYPOINT_MARKER_RADIUS = 6
ACTIVE_WAYPOINT_HALO_PADDING = 6
START_MARKER_RADIUS = 6
FRONTIER_OR_ENDPOINT_MARKER_RADIUS = 7

DEFAULT_RENDER_THROTTLE_FPS = 30.0


def ipp_uncertainty_rgba(values, mask) -> np.ndarray:
    """Map a variance raster to a perceptually ordered, transparent RGBA image."""
    grid = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    if grid.ndim != 2 or valid.shape != grid.shape or not np.any(valid):
        raise ValueError("IPP variance and mask must be matching non-empty 2-D arrays.")
    low = float(np.min(grid[valid]))
    high = float(np.max(grid[valid]))
    if high - low <= 1e-12:
        normalized = np.full(grid.shape, 0.5, dtype=np.float64)
    else:
        normalized = np.clip((grid - low) / (high - low), 0.0, 1.0)

    # Viridis-like anchors: low uncertainty is dark violet/blue, high
    # uncertainty is yellow.  The ordered lightness remains readable on both
    # application themes and cannot be mistaken for the red/orange hazard map.
    anchors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float64,
    )
    scaled = normalized * (len(anchors) - 1)
    lower = np.minimum(scaled.astype(np.int32), len(anchors) - 2)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[lower + 1] * fraction
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 142, 0).astype(np.uint8)
    return rgba


def _fit_world_span_to_plot_aspect(
    span_x: float,
    span_y: float,
    plot_width: float,
    plot_height: float,
) -> tuple[float, float]:
    """Expand a logical world viewport span to match the plot area's
    aspect ratio, growing only the one axis that needs it.

    world_to_screen()/screen_to_world() use rect.width()/span_x and
    rect.height()/span_y as their X/Y scale factors. Those two are only
    equal (uniform scale -- circles stay circles, squares stay squares)
    when span_x/span_y already equals plot_width/plot_height. Rather than
    stretch X and Y independently (the previous behavior, which distorted
    all rendered geometry whenever the configured viewport's aspect ratio
    did not match the canvas's), this instead grows span_x or span_y just
    enough to match the plot's aspect ratio -- so the canvas fills
    completely (no letterboxing) without ever cropping or shrinking the
    logical viewport, and without stretching either axis independently.

    Symmetric by construction: the caller re-centers the result on the
    same world point as the logical viewport (see
    render_view_bounds_world()), so the added margin is split evenly
    left/right or top/bottom -- never a one-sided crop.

    Guarantees render_span_x >= span_x and render_span_y >= span_y.

    Falls back to the logical span unchanged when plot_width/plot_height
    are not yet valid (e.g. before the widget's first layout pass) --
    better to render at the logical, possibly momentarily non-uniform,
    scale for one frame than divide by zero.
    """
    span_x = max(1e-6, float(span_x))
    span_y = max(1e-6, float(span_y))
    if plot_width <= 0.0 or plot_height <= 0.0:
        return span_x, span_y

    logical_aspect = span_x / span_y
    plot_aspect = float(plot_width) / float(plot_height)

    if plot_aspect > logical_aspect:
        # Canvas is proportionally wider than the logical viewport --
        # keep height, expand width.
        return span_y * plot_aspect, span_y

    # Canvas is proportionally taller than (or as tall as) the logical
    # viewport -- keep width, expand height.
    return span_x, span_x / plot_aspect


class RenderThrottler:
    """Decides whether a high-frequency, simulation-driven repaint request
    should actually trigger self.update() right now, or be coalesced
    (skipped) because a repaint already happened recently enough to hit
    target_fps.

    Pure/Qt-free on purpose (no QWidget dependency) so it is unit-testable
    without a running Qt application. Coalescing loses nothing visually:
    Qt's paintEvent always paints the CURRENT widget/simulation state, not
    a queue of past ones, so skipping an update() call between two accepted
    calls only skips a redundant repaint of state that either looked
    identical or is about to be superseded by the next accepted call.

    Only wired into the two per-tick setters (set_runtime_state()/
    set_multi_runtime_state()) that the engine calls every simulation
    tick while running and unpaused -- every other self.update() call in
    this class (mouse/editor interactions, status/config changes, which
    already only ever fire on user action or while not actively
    simulating) is untouched and stays immediate, matching "render
    immediately after user interactions".

    target_fps defaults to the SIM_RENDER_FPS environment variable
    (read at construction time, mirroring RobotTrace/PerfMonitor's own
    env-reading convention) when not given explicitly, falling back to
    DEFAULT_RENDER_THROTTLE_FPS if that env var is unset. Pass `env=`
    explicitly in tests for a deterministic instance.
    """

    def __init__(
        self,
        target_fps: float | None = None,
        *,
        env: "dict[str, str] | None" = None,
    ):
        if target_fps is None:
            source = env if env is not None else os.environ
            target_fps = float(source.get("SIM_RENDER_FPS", DEFAULT_RENDER_THROTTLE_FPS))
        self.target_fps = float(target_fps)
        self._min_interval = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0
        self._last_render_time: float | None = None

    def should_render(self, now: float | None = None, *, force: bool = False) -> bool:
        now = time.perf_counter() if now is None else float(now)
        if force or self._last_render_time is None or (now - self._last_render_time) >= self._min_interval:
            self._last_render_time = now
            return True
        return False


def _visible_fire_sources(sources, observed, *, bounds, resolution):
    """Pure anti-omniscience filter for fire markers: keep only the
    `sources` whose CENTER cell -- the cell containing `source.position`,
    computed with the exact same floor-division convention GridGeometry.
    world_to_grid() uses -- is observed=True in `observed`.

    No Qt, no engine, no HazardField/RuntimeHazardService: `sources` (any
    objects exposing `.position`), `observed` (a 2D bool array), `bounds`,
    and `resolution` are the only inputs, so this is directly unit-testable
    with plain data. Never infers a marker from the source's radius and
    never marks a source visible because only part of its thermal halo was
    observed -- only the single center cell matters, matching the spec's
    own "no infieras el marker desde el radius" rule.
    """
    if observed is None or getattr(observed, "size", 0) == 0:
        return []

    x_min, x_max, y_min, y_max = (float(v) for v in bounds)
    resolution = float(resolution)
    height, width = observed.shape[0], observed.shape[1]

    visible = []
    for source in sources:
        x, y = float(source.position[0]), float(source.position[1])
        if not (x_min <= x < x_max and y_min <= y < y_max):
            continue
        col = int(math.floor((x - x_min) / resolution))
        row = int(math.floor((y - y_min) / resolution))
        if not (0 <= row < height and 0 <= col < width):
            continue
        if observed[row, col]:
            visible.append(source)
    return visible


def _draw_fire_beacon(painter: QPainter, cx: float, cy: float, colors, *, discovered: bool) -> None:
    """Draw one minimalist vectorial source beacon centered at screen
    position (cx, cy) -- constant ~16px screen-space size regardless of
    zoom (every shape below is a fixed pixel offset from (cx, cy), never
    scaled by pixels_per_meter()).

    Deliberately NOT a flame silhouette (see the removed _fire_marker_
    flame_path()/_draw_fire_marker() this replaces) -- three simple
    circular layers back to front: halo, thin outer ring, core. No
    QPainterPath, no images, emoji, blur, particles, animation, or timers.

    `discovered` selects the palette and weight, never the shape: a
    discovered source (its center cell is observed=True) gets a warm ring
    and a solid, fully opaque core; an undiscovered source -- only ever
    drawn at all while Fire Markers is ON, see draw_fire_markers() -- gets
    a tenue blue ring, a tiny near-transparent core, and lower opacity
    throughout, reading clearly as "ground-truth debug info", not
    something the team has actually detected.

    `colors` is a theme.ThemeColors -- callers read it once per paint via
    theme_colors(self._theme_mode) and pass it in, so this function itself
    never imports/calls theme_colors directly.
    """
    ring_hex = colors.fire_discovered_ring if discovered else colors.fire_undiscovered_ring
    core_hex = colors.fire_discovered_core if discovered else colors.fire_undiscovered_core
    halo_alpha = 55 if discovered else 26
    ring_alpha = 235 if discovered else 130
    core_alpha = 235 if discovered else 70
    core_radius = 3.2 if discovered else 1.6

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(cx, cy)

    # 1. Halo -- small, soft, low-alpha circle behind everything else.
    halo_color = QColor(ring_hex)
    halo_color.setAlpha(halo_alpha)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(halo_color))
    painter.drawEllipse(QRectF(-8.0, -8.0, 16.0, 16.0))

    # 2. Thin outer ring.
    ring_color = QColor(ring_hex)
    ring_color.setAlpha(ring_alpha)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(ring_color, 1.6))
    painter.drawEllipse(QRectF(-5.0, -5.0, 10.0, 10.0))

    # 3. Core circle -- solid/luminous when discovered, tiny/near-
    # transparent otherwise.
    core_color = QColor(core_hex)
    core_color.setAlpha(core_alpha)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(core_color))
    painter.drawEllipse(QRectF(-core_radius, -core_radius, 2 * core_radius, 2 * core_radius))

    painter.restore()


class SimulationCanvas(QWidget):
    goalClicked = Signal(float, float)
    robotDragged = Signal(int, float, float)
    robotSelected = Signal(int)
    editor_interaction_started = Signal(tuple)
    editor_interaction_progress = Signal(tuple)
    editor_interaction_finished = Signal(tuple, tuple)
    editor_camera_changed = Signal(tuple)
    editor_camera_interaction_started = Signal()
    editor_obstacle_move_started = Signal()
    editor_obstacle_moved = Signal(tuple)
    editor_view_changed = Signal()
    # Emitted instead of goalClicked when in exploration mode -- the canvas
    # only reports "user clicked here", never decides whether that means
    # add or remove a fire (main_window.py checks proximity to existing
    # fires and decides).
    fireToggleRequested = Signal(float, float)
    # Human Demonstration mode only (see set_human_demo_mode()): a live,
    # running robot was clicked (carries robot_id, not a preview index).
    humanDemoRobotClicked = Signal(int)
    # Human Demonstration mode only: one of the frozen candidate frontier
    # markers currently shown via set_human_demo_candidate_markers() was
    # clicked (carries candidate_id -- never a coordinate).
    humanDemoCandidateClicked = Signal(str)

    def __init__(self):
        super().__init__()

        self.setObjectName("canvasCard")
        self.setMinimumSize(610, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._theme_mode = ThemeMode.LIGHT

        self.robot = None
        self.robots: list = []
        self.config = SimulationConfig()
        self.path_points: list[tuple[float, float]] = []

        # Human Demonstration mode (see set_human_demo_mode()): while
        # active, clicks are routed to humanDemoRobotClicked/
        # humanDemoCandidateClicked instead of goalClicked/
        # fireToggleRequested. Markers are (candidate_id, world_x,
        # world_y, enabled) tuples for the currently-focused robot only --
        # never regenerated here, only ever set by the host.
        self._human_demo_mode_active = False
        self._human_demo_candidate_markers: tuple[tuple[str, float, float, bool], ...] = ()
        self.multi_path_points: list[list[tuple[float, float]]] = []
        self.multi_last_controls: list[np.ndarray] = []
        self.planned_path_points: list[tuple[float, float]] = []
        # Optional, validated paper-experiment layer.  It is separate from the
        # hazard and occupancy maps because it represents a scalar field and GP
        # posterior uncertainty, not collision geometry.
        self._ipp_experiment_bundle = None
        self._ipp_variance_pixmap_cache: QPixmap | None = None
        self._ipp_variance_pixmap_cache_key: tuple | None = None
        # Read-only snapshot of the independent continuous hazard field
        # (GROUND TRUTH -- kept for legacy/potential editor use only; see
        # draw_fires(), no longer called from the live paint loop).
        # The canvas never owns fire sources and never writes occupancy.
        self._hazard_snapshot: dict | None = None
        self._hazard_pixmap_cache: QPixmap | None = None
        self._hazard_pixmap_cache_key: tuple | None = None
        # Separate pixmap cache for the full ground-truth BLUE inspection
        # heatmap (see draw_ground_truth_hazard_map()/set_hazard_map_
        # enabled()) -- entirely independent of _hazard_pixmap_cache above
        # (draw_fires(), a different legacy yellow/red palette, unrelated
        # to this toggle) and of the warm discovered-hazard caches below.
        self._ground_truth_hazard_pixmap_cache: QPixmap | None = None
        self._ground_truth_hazard_pixmap_cache_key: tuple | None = None
        # Read-only frame of the team's DISCOVERED hazard belief -- what
        # live simulation actually renders (see draw_discovered_hazard()).
        # {"frame": HazardBeliefFrame, "bounds": ..., "resolution": ...}.
        self._discovered_hazard_frame: dict | None = None
        self._discovered_hazard_pixmap_cache: QPixmap | None = None
        self._discovered_hazard_pixmap_cache_key: tuple | None = None
        # Separate pixmap cache for the HISTORICAL discovered-hazard frame
        # (see _decoded_navigation_debug_hazard_belief()) -- kept distinct
        # from the live cache above so toggling between LIVE and HISTORY
        # repeatedly never thrashes either one.
        self._discovered_hazard_history_pixmap_cache: QPixmap | None = None
        self._discovered_hazard_history_pixmap_cache_key: tuple | None = None
        # Two independent rendering-only toggles, both OFF by default: the
        # discovered hazard belief and its discovered fire sources are
        # ALWAYS shown regardless of these -- they only ever ADD
        # ground-truth debug information on top (see draw_ground_truth_
        # hazard_map()/draw_fire_markers()'s own docstrings). Never
        # persisted, never touch SimulationConfig/HazardBelief/planning,
        # and safe to flip while a simulation is running, same contract as
        # grid_overlay_enabled below.
        self.show_hazard_map = False
        self.show_fire_markers = False
        self.exploration_target_xy: tuple[float, float] | None = None
        self.frontier_reasoning_overlay_enabled = False
        self.frontier_reasoning_simulation_paused = False
        self.frontier_reasoning_decision: dict | None = None
        self.frontier_reasoning_inspection: dict | None = None
        self.frontier_reasoning_cluster_view_enabled = False
        self.frontier_reasoning_clusters: tuple[dict, ...] = ()
        self.cursor_coordinates_enabled = True
        self.cursor_coordinate_position: tuple[float, float] | None = None
        self.cursor_coordinate_world: tuple[float, float] | None = None
        self.multi_exploration_targets: list[tuple[float, float] | None] = []
        self.multi_invalidated_exploration_targets: list[list[tuple[float, float]]] = []
        self.explored_area_polygons: list[list[tuple[float, float]]] = []
        self.mapped_obstacle_points: list[tuple[float, float]] = []
        self.known_obstacles: list[tuple[float, float, float, float]] = []
        self.status_message = "Configure parameters and press Start."
        self.status_history: list[str] = []
        self.status_history_limit = 2000
        self._append_status_history(self.status_message)
        self.last_control = np.array([[0.0], [0.0]], dtype=float)

        self.plot_margin_left = 30
        self.plot_margin_top = 60
        self.plot_margin_right = 30
        self.plot_margin_bottom = 70

        # Cached visual layers. Static map background is expensive because it
        # draws topographic curves and grid lines. Explored area is expensive
        # when rebuilt from hundreds of polygons. Both are cached as pixmaps.
        self._static_plot_cache: QPixmap | None = None
        self._static_plot_cache_size: QSize | None = None
        self._explored_area_cache: QPixmap | None = None
        self._explored_area_cache_size: QSize | None = None
        self._explored_area_cached_count = 0
        self._explored_area_caches_by_robot: dict[int, QPixmap] = {}
        self._explored_area_cache_sizes_by_robot: dict[int, QSize] = {}
        # The authoritative source rebuild_explored_area_cache() replays
        # whenever the live explored-area cache is invalidated (theme
        # toggle/resize/pan-zoom/set_explored_area_polygons([])) -- see
        # set_explored_area_seed(). Two lifecycles share this field:
        # engine.py publishes the LIVE belief_map.explored_by_robot mask
        # here once per fresh run (see engine._publish_explored_area_
        # source_to_canvas()), so it always reflects current coverage with
        # no copying; engine.restore_navigation_debug_snapshot() publishes
        # the just-restored mask the same way. Without this, the
        # authoritative belief_map.explored_by_robot stays correct while
        # the cosmetic explored-area trail (bounded to the last
        # EXPLORED_POLYGON_HISTORY_LIMIT sensor sweeps) visibly resets to
        # nothing on the next rebuild, even though nothing was actually
        # un-explored. Read-only -- the canvas never writes into this
        # array. None until a run publishes one; reset to the new run's
        # own (empty) mask on a fresh run (start_simulation()/reset_
        # simulation()/start_multi_robot_simulation()) -- never left
        # pointing at a previous run's replaced BeliefMap.
        self._explored_area_seed_mask: np.ndarray | None = None
        self._explored_area_seed_resolution: float | None = None
        self._explored_area_seed_bounds: tuple[float, float, float, float] | None = None

        # Continuous, world-coordinate FoV sweep geometry -- the
        # AUTHORITATIVE visual source once any sweep has been painted this
        # session (see append_explored_area_polygon()/rebuild_explored_
        # area_cache()). Keyed by robot_index (int) for multi-robot mode,
        # or None for single-robot mode -- the same robot_index=None
        # convention used throughout this file. Takes priority over
        # _explored_area_seed_mask during a rebuild, per robot: the mask
        # stays as a fallback only for a robot that has no continuous path
        # of its own yet (e.g. right after a snapshot restore -- see
        # clear_explored_area_geometry()). Never truncated (unlike
        # explored_area_polygons/EXPLORED_POLYGON_HISTORY_LIMIT) -- that is
        # the whole point: a theme toggle/resize/pan/zoom must never
        # degrade smooth diagonal FoV sweeps into grid-cell-quantized
        # squares just because the bounded python-side polygon list was
        # trimmed. Cleared only by clear_explored_area_geometry() (a fresh
        # run/reset/restore) -- never by invalidate_explored_area_cache()
        # (theme toggle/resize/pan/zoom), which only drops the QPixmap
        # render caches built FROM this geometry, not the geometry itself.
        self._explored_area_paths_by_robot: dict[int | None, QPainterPath] = {}

        # Mapped obstacle points can become thousands of tiny ellipses. Drawing
        # each point every paintEvent is expensive, so they are rasterized into
        # a transparent cache and updated only when new points arrive.
        self._mapped_points_cache: QPixmap | None = None
        self._mapped_points_cache_size: QSize | None = None
        self._mapped_points_cached_count = 0

        # Obstacle completion opacity depends on mapped points. Recomputing
        # coverage in every paintEvent is O(boundary_samples * mapped_points),
        # so the values are cached and invalidated only when mapping changes.
        self._obstacle_coverage_cache: dict[int, float] = {}
        self._obstacle_coverage_cache_count = -1

        # Ground-truth obstacle rendering is cached separately. Showing obstacles
        # used to drop FPS because coverage was recomputed and rectangles were
        # redrawn during paintEvent. Now the obstacle layer is rasterized only
        # when the canvas changes size, the obstacle list changes, or new mapped
        # points may change completion opacity.
        self._obstacles_cache: QPixmap | None = None
        self._obstacles_cache_size: QSize | None = None
        self._obstacles_cache_mapped_count = -1
        self._obstacles_cache_signature: tuple | None = None

        # Runtime metrics. FPS is measured in paintEvent because that is the
        # rate the user actually sees, not just the QTimer tick rate.
        self.fps = 0.0
        self._fps_frame_count = 0
        self._fps_last_time = time.perf_counter()
        self.simulation_time = 0.0
        self.simulation_speed = 1.0
        self.metrics_visible = True

        # Temporary red grid-resolution preview shown while the user adjusts
        # SimulationConfig.grid_resolution in the config panel. Purely visual
        # -- it never touches self.config or any simulation-facing state, and
        # auto-hides itself shortly after the last change so it never becomes
        # a permanent, easy-to-forget overlay.
        self._grid_resolution_preview_active = False
        self._grid_resolution_preview_resolution: float | None = None
        self._grid_resolution_preview_timer = QTimer(self)
        self._grid_resolution_preview_timer.setSingleShot(True)
        self._grid_resolution_preview_timer.timeout.connect(self.hide_grid_resolution_preview)

        # Persistent "Show Grid" overlay ("Grid Overlay" toggle). Unlike the
        # temporary preview above, this does not auto-hide -- it stays on
        # until the user turns it off, including while the simulation is
        # running. Purely visual/debug: it never touches self.config and
        # never rebuilds any occupancy/planning grid. _grid_overlay_snapshot
        # is an optional read-only copy of the current belief/occupancy
        # grid (resolution/bounds/grid array) pushed in from outside, used
        # to color occupied/free/unknown cells while running; when absent
        # (not running, or no belief map yet) only resolution grid lines
        # are drawn.
        self.grid_overlay_enabled = False
        self.grid_cell_values_enabled = False
        self.frontier_decisions_enabled = False
        self._grid_overlay_resolution = 0.50
        self._grid_overlay_snapshot: dict | None = None
        self._grid_overlay_snapshot_version = 0
        self._grid_overlay_snapshot_pushed_at: float | None = None

        # Rendered-overlay cache. Rebuilding requires looping over every
        # visible occupancy cell and issuing a QPainter.drawRect() call per
        # cell -- fine once, ruinous if repeated every frame at a fine
        # grid_resolution. The cache is reused as long as resolution, canvas
        # size, view bounds, and the occupancy snapshot are all unchanged;
        # otherwise it is rebuilt once and reused again.
        self._grid_overlay_cache: QPixmap | None = None
        self._grid_overlay_cache_key: tuple | None = None
        self._grid_overlay_last_cache_status = "off"
        self._grid_overlay_last_visible_cells = 0
        self._grid_overlay_degraded = False
        # Fine-grained grid-overlay timings, reported via the optional
        # [RENDER] detail line only -- see draw_grid_overlay()'s/
        # _rebuild_grid_overlay_cache()'s own comments for where each is
        # measured. Purely observational: reading/writing these never
        # changes which branch draw_grid_overlay() takes.
        self._grid_overlay_rebuild_ms = 0.0
        self._grid_overlay_blit_ms = 0.0
        self._grid_overlay_cells_ms = 0.0
        self._grid_overlay_lines_ms = 0.0

        # Navigation debug overlay. Off by default. The canvas never
        # imports/calls planning, navigation, or collision-checking code --
        # it only ever holds the single most recent NavigationDebugSnapshot
        # (an immutable, plain-data value pushed in from engine.py) and
        # renders whatever is already inside it. Pausing the simulation
        # simply stops new set_navigation_debug_snapshot() calls; nothing
        # here ever clears _nav_debug_snapshot on its own, so the last
        # relevant snapshot survives untouched across repeated repaints.
        self.navigation_debug_enabled = False
        self._nav_debug_snapshot = None
        # The last RELEVANT event (PLAN_ACCEPTED/ROUTE_REJECTED/SAFETY_REPLAN/
        # PREDICTED_COLLISION/HOLD/EXHAUSTED/...), separate from
        # _nav_debug_snapshot: the latter updates every tick while running
        # (for a live HUD), this one only updates when engine.py's bounded
        # ring buffer actually gains a new entry -- so "what was the last
        # relevant thing that happened" survives many quiet ticks after it.
        self._nav_debug_last_event = None
        # (position, total) while stepping through history (paused), or
        # (None, total) while showing the live snapshot.
        self._nav_debug_history_position: tuple[int | None, int] = (None, 0)
        self._nav_debug_overlay_cache: dict | None = None
        self._nav_debug_overlay_cache_key: tuple | None = None
        # Decoded historical belief/exploration state and its raster cache.
        # Both are rebuilt only when the selected snapshot revision or view
        # transform changes, never on every paintEvent.
        self._nav_debug_environment_decode_key: tuple | None = None
        self._nav_debug_environment_decoded: dict | None = None
        self._nav_debug_explored_cache: QPixmap | None = None
        self._nav_debug_explored_cache_key: tuple | None = None
        # Decoded historical HazardBelief state (values/observed/bounds/
        # resolution) -- same one-decode-per-(frame, revision) pattern as
        # _nav_debug_environment_decode_key above, see
        # _decoded_navigation_debug_hazard_belief().
        self._nav_debug_hazard_belief_decode_key: tuple | None = None
        self._nav_debug_hazard_belief_decoded: dict | None = None
        # Optional docked panel the full field breakdown is forwarded
        # to -- see set_navigation_reasoning_window(). None until
        # main_window.py registers one.
        self._navigation_reasoning_window = None
        # MainWindow registers a real QWidget action bar that occupies the
        # footer previously used by the painted telemetry strip.
        self._action_bar = None

        # History stepping has exactly one control now: the navigation_
        # snapshot_bar docked above the canvas (main_window.
        # _build_navigation_snapshot_bar()). There used to be a second,
        # redundant pair of `<`/`>` QPushButtons parented directly to the
        # canvas -- removed to avoid two independent controls driving the
        # same engine state.

        # Render-only FPS/frame-time telemetry. Independent of the engine --
        # this only ever measures how fast paintEvent itself is running.
        # Routine samples are NEVER printed to stdout/terminal and NEVER
        # appended to the GUI console (that would just trade one spam
        # problem for another) -- they are only kept in-memory as
        # latest_perf_status, inspectable by an optional in-app "Show FPS"
        # display without any terminal or GUI console output. Only a
        # genuinely severe, much less frequent FPS drop reaches the GUI
        # console, via _perf_gui_warning_gate.
        self._render_perf_monitor = RenderPerfMonitor()
        self._perf_gui_warning_gate = PerfGuiWarningGate()
        # Optional, throttled per-layer paint breakdown -- off by default,
        # SIM_RENDER_DETAIL_LOG=1 enables a [RENDER] line at most every 2s.
        # Independent of _render_perf_monitor's own routine (never printed)
        # paint_fps/paint_ms tracking above.
        self._render_detail_logger = RenderDetailLogger()
        # The optional detailed render line is retained in memory for
        # diagnostics/export, never printed to the terminal by the GUI.
        self._latest_render_detail_line: str | None = None
        self._last_background_cache_hit = False
        self._render_layer_ms: dict[str, float] = {
            "background": 0.0, "map_layer": 0.0, "robot_body": 0.0, "robot_fov": 0.0,
            "route_path": 0.0, "sensor_debug_overlay": 0.0, "overlays": 0.0,
            # map_layer_ms sub-buckets -- these four must sum back to
            # map_layer_ms (plus negligible measurement overhead).
            "grid_overlay": 0.0, "explored_area": 0.0,
            "ground_truth_obstacles": 0.0, "mapped_obstacle_points": 0.0,
            # overlays_ms sub-buckets -- these six must sum back to
            # overlays_ms (plus negligible measurement overhead).
            "editor_overlays": 0.0, "grid_preview": 0.0, "plot_border": 0.0,
            "card": 0.0, "title": 0.0, "telemetry": 0.0,
        }
        # Fine-grained robot-FOV timings, reported via the optional
        # [RENDER] detail line -- see draw_sensor_range()'s own comments
        # for where each is measured. Mirrors _route_detail's pattern.
        self._fov_detail: dict = {
            "robot_fov_cache_hit": True,
            "robot_fov_compute_ms": 0.0,
            "robot_fov_paint_ms": 0.0,
        }
        # Render caches for robot-related dynamic layers -- see
        # draw_executed_path()/draw_planned_route()'s own docstrings for
        # the invalidation rules.
        #
        # The executed trail is painted into a persistent QPixmap rather
        # than cached as a QPainterPath: a QPainterPath cache still costs
        # painter.drawPath() proportional to total point count every
        # single frame, so it grows unboundedly as the trail accumulates
        # over a long run (this is exactly what the real Office.sim
        # route_path_ms evidence showed -- 17ms growing to 431ms as the
        # trail got longer, even though the path object itself was never
        # rebuilt). A pixmap is blitted in ~constant time regardless of
        # how many points it was painted from.
        self._executed_trail_pixmap: QPixmap | None = None
        self._executed_trail_pixmap_count = 0
        self._executed_trail_view_signature: tuple | None = None
        self._executed_trail_source: list | None = None
        self._executed_trail_style: tuple | None = None
        self._executed_trail_last_screen_point: tuple | None = None
        self._executed_trail_segments_painted_last_frame = 0
        # Multiple mode uses the same incremental raster strategy, with one
        # count/last-point cursor per robot.  Keeping the inner path-list
        # identities lets runtime updates append cheaply while a restart (new
        # lists) still invalidates and clears the cache deterministically.
        self._multi_executed_trail_pixmap: QPixmap | None = None
        self._multi_executed_trail_counts: list[int] = []
        self._multi_executed_trail_view_signature: tuple | None = None
        self._multi_executed_trail_sources: tuple = ()
        self._multi_executed_trail_style: tuple | None = None
        self._multi_executed_trail_last_screen_points: list[tuple | None] = []
        self._multi_executed_trail_segments_painted_last_frame = 0
        self._planned_route_cache: QPainterPath | None = None
        self._planned_route_cache_signature: tuple | None = None
        # Fine-grained route/trail timings, reported via the optional
        # [RENDER] detail line -- see draw_planned_route()/
        # draw_executed_path() for where each is measured.
        self._route_detail: dict = {
            "planned_route_build_ms": 0.0,
            "planned_route_paint_ms": 0.0,
            "executed_trail_build_ms": 0.0,
            "executed_trail_paint_ms": 0.0,
            "executed_trail_points": 0,
            "executed_trail_segments_painted": 0,
            "executed_trail_cache_hit": False,
        }
        # Throttles only the high-frequency, simulation-driven repaint
        # requests (set_runtime_state()/set_multi_runtime_state()) to at
        # most DEFAULT_RENDER_THROTTLE_FPS repaints/second -- see
        # RenderThrottler's docstring. Does not affect any other
        # self.update() call in this class.
        self._render_throttler = RenderThrottler()
        self.latest_perf_status: dict | None = None
        # Gates GUI-console perf warnings only (see
        # _maybe_emit_perf_gui_warning/draw_grid_overlay's degraded notice)
        # -- a low paint_fps during setup/load/reset, or with the overlay
        # off, is not meaningful and must not be reported as if Show Grid
        # were the cause. Set via set_simulation_running_for_perf().
        self._simulation_running_for_perf = False
        # Tracks whether the one-time "grid overlay degraded" console line
        # has already been shown for the CURRENT run + degraded streak --
        # separate from _grid_overlay_degraded (which also gates cache-key/
        # snapshot-throttle logic and must stay accurate even while idle).
        self._grid_overlay_degraded_notice_shown = False

        # Dragging support for pre-simulation multi-robot placement.
        self.dragging_robot_index: int | None = None
        self.dragging_robot_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_mode = False
        self.editor_tool = "rectangles"
        self.editor_drag_start: tuple[float, float] | None = None
        self.editor_drag_current: tuple[float, float] | None = None
        self.editor_preview_points: list[tuple[float, float]] = []
        self.editor_pan_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_zoom = 1.0
        self.editor_brush_size = 0.2
        self.editor_interaction_mode = "paint"
        self.editor_pan_active = False
        self.editor_last_pan_pos: tuple[float, float] | None = None
        self.editor_camera_active_handle: str | None = None
        self.editor_camera_drag_start_world: tuple[float, float] | None = None
        self.editor_camera_start_bounds: tuple[float, float, float, float] | None = None
        self.editor_obstacle_drag_index: int | None = None
        self.editor_obstacle_drag_indices: list[int] = []
        self.editor_obstacle_drag_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_obstacle_drag_last_world: tuple[float, float] | None = None

        # Cached current blue sensor footprint. This avoids recomputing
        # ray-casting in every paintEvent when the robot moved only a tiny
        # amount since the previous frame.
        self._sensor_polygon_cache: list[tuple[float, float]] = []
        self._sensor_polygon_pose: tuple[float, float, float] | None = None
        self._sensor_polygon_signature: tuple | None = None
        self._sensor_polygon_caches_by_robot: dict[int, tuple[tuple[float, float, float], tuple, list[tuple[float, float]]]] = {}

    def resizeEvent(self, event):
        self.invalidate_static_plot_cache()
        self.invalidate_explored_area_cache()
        self.invalidate_mapped_points_cache()
        self.invalidate_obstacles_cache()
        self._position_action_bar()
        super().resizeEvent(event)

    def set_action_bar(self, action_bar: QWidget | None) -> None:
        """Register the real runtime-control footer owned by MainWindow."""
        previous = getattr(self, "_action_bar", None)
        if previous is not None and previous is not action_bar:
            previous.hide()
        self._action_bar = action_bar
        if action_bar is not None:
            action_bar.setParent(self)
            action_bar.show()
            action_bar.raise_()
            self._position_action_bar()

    def set_action_bar_visible(self, visible: bool) -> None:
        bar = getattr(self, "_action_bar", None)
        if bar is not None:
            bar.setVisible(bool(visible))
            if visible:
                bar.raise_()

    def _position_action_bar(self) -> None:
        bar = getattr(self, "_action_bar", None)
        if bar is None:
            return
        rect = self.telemetry_rect()
        if isinstance(rect, QRectF):
            rect = rect.toAlignedRect()
        bar.setGeometry(rect)
        bar.raise_()

    def invalidate_static_plot_cache(self):
        self._static_plot_cache = None
        self._static_plot_cache_size = None

    def is_monochrome_discovery_mode(self) -> bool:
        return (
            getattr(self.config, "map_visualization", DEFAULT_MAP_VISUALIZATION)
            == "Monochrome Discovery"
        )

    def is_custom_discovery_mode(self) -> bool:
        return (
            getattr(self.config, "map_visualization", DEFAULT_MAP_VISUALIZATION)
            == "Custom Discovery"
        )

    def is_shared_discovery_mode(self) -> bool:
        """Rendering-only discovery presentations with one shared team color."""
        return self.is_monochrome_discovery_mode() or self.is_custom_discovery_mode()

    @staticmethod
    def _valid_config_color(value: str, fallback: str) -> QColor:
        color = QColor(str(value))
        return color if color.isValid() else QColor(str(fallback))

    def plot_background_color(self) -> QColor:
        if self.is_monochrome_discovery_mode():
            return QColor(0, 0, 0)
        if self.is_custom_discovery_mode():
            return self._valid_config_color(
                getattr(self.config, "custom_unexplored_color", DEFAULT_CUSTOM_UNEXPLORED_COLOR),
                DEFAULT_CUSTOM_UNEXPLORED_COLOR,
            )
        return QColor(theme_colors(self._theme_mode).app_background)

    def explored_area_color(self, robot_index: int | None = None) -> QColor:
        """Return the free-space discovery color without affecting obstacles."""
        if self.is_monochrome_discovery_mode():
            if self._theme_mode == ThemeMode.DARK:
                return QColor(184, 190, 198)
            return QColor(248, 249, 251)
        if self.is_custom_discovery_mode():
            return self._valid_config_color(
                getattr(self.config, "custom_explored_color", DEFAULT_CUSTOM_EXPLORED_COLOR),
                DEFAULT_CUSTOM_EXPLORED_COLOR,
            )
        if robot_index is None or int(robot_index) < 0:
            return QColor(BLUE)
        return robot_color(int(robot_index))

    def sensor_display_color(self, robot_index: int | None = None) -> QColor:
        if self.is_monochrome_discovery_mode():
            return QColor(255, 255, 255)
        if self.is_custom_discovery_mode():
            return self.explored_area_color(robot_index)
        if robot_index is None or int(robot_index) < 0:
            return QColor(BLUE)
        return robot_color(int(robot_index))

    def discovery_contrast_color(self, alpha: int) -> QColor:
        """Black or white canvas chrome chosen against the unknown color."""
        background = self.plot_background_color()
        color = QColor(0, 0, 0) if background.lightness() >= 150 else QColor(255, 255, 255)
        color.setAlpha(max(0, min(255, int(alpha))))
        return color

    def set_theme_mode(self, mode: ThemeMode | str) -> None:
        """Re-theme the canvas chrome (card/header/plot backdrop/borders and
        the obstacle/explored-area tints that must stand out against a dark
        canvas). Only invalidates the *theme-dependent* pixmap caches --
        belief map, routes, hazards, and the explored-area seed/history state
        are untouched, so a theme switch never looks like a fresh run."""
        mode = ThemeMode(mode)
        if mode == self._theme_mode:
            return
        self._theme_mode = mode
        self.invalidate_static_plot_cache()
        self.invalidate_obstacles_cache()
        self.invalidate_explored_area_cache()
        self.invalidate_mapped_points_cache()
        self._grid_overlay_cache = None
        self._grid_overlay_cache_key = None
        self.update()

    def invalidate_explored_area_cache(self):
        self._explored_area_cache = None
        self._explored_area_cache_size = None
        self._explored_area_cached_count = 0
        self._explored_area_caches_by_robot = {}
        self._explored_area_cache_sizes_by_robot = {}

    def set_explored_area_seed(
        self,
        mask: np.ndarray,
        resolution: float,
        bounds: tuple[float, float, float, float],
    ) -> None:
        """Point the live explored-area cache at an authoritative
        belief_map.explored_by_robot mask -- the DISCRETE source
        rebuild_explored_area_cache() replays whenever the cache is
        invalidated (theme toggle/resize/pan-zoom -- see invalidate_
        explored_area_cache()) and the robot in question has no continuous
        path of its own yet (see _explored_area_paths_by_robot -- the
        smooth, continuous geometry painted per FoV sweep via append_
        explored_area_polygon() takes priority over this mask, per robot,
        once it exists).

        Not exclusively a restored-snapshot seed: engine.py calls this once
        per fresh run too, right after (re)creating BeliefMap (see engine.
        _publish_explored_area_source_to_canvas()), passing the run's LIVE,
        continuously-updated mask -- not a one-shot snapshot. The canvas
        keeps only this ndarray reference (never copies it -- see
        clear_explored_area_seed()), and every mutating belief_map update
        (mark_free_cell/mark_occupied_cell/mark_visible_polygon/etc.)
        writes into it in place, so a later rebuild always sees current
        coverage with no extra wiring per tick. engine.restore_navigation_
        debug_snapshot() calls this too, with a just-restored mask, for the
        same underlying reason: without it, the authoritative belief.
        explored_by_robot stays correct while the cosmetic sensor-sweep
        trail (explored_area_polygons, bounded to EXPLORED_POLYGON_
        HISTORY_LIMIT entries) visibly resets to nothing on the next
        rebuild, even though nothing was actually un-explored.

        Unlike painting a one-shot pixmap, this persists across cache
        rebuilds (resize/pan/zoom/set_explored_area_polygons([])):
        rebuild_explored_area_cache() replays this mask first, every time,
        exactly like the historical-replay view already replays a frozen
        mask (see _draw_historical_explored_area()). For a single-robot
        mask (shape[0] == 1) this rebuilds the combined cache directly from
        the mask -- explored_area_polygons is not replayed on top of it,
        since the mask already is the complete authoritative state. For a
        multi-robot mask (shape[0] > 1) each robot's slice rebuilds that
        robot's own attributed cache (_explored_area_caches_by_robot), so
        one robot's coverage never depends on another robot's cache already
        existing. Live sensor sweeps recorded after this call are then
        painted on top as usual (see append_explored_area_polygon()).

        The canvas never writes into `mask` -- it is a read-only reference.
        """
        self._explored_area_seed_mask = mask
        self._explored_area_seed_resolution = float(resolution)
        self._explored_area_seed_bounds = tuple(float(v) for v in bounds)
        self.invalidate_explored_area_cache()
        self.update()

    def clear_explored_area_seed(self) -> None:
        """Drop the restored-mask seed -- called at the start of a fresh
        run so a previous run's restore does not leak into it."""
        self._explored_area_seed_mask = None
        self._explored_area_seed_resolution = None
        self._explored_area_seed_bounds = None

    def clear_explored_area_geometry(self) -> None:
        """Drop the authoritative continuous-path coverage and the bounded
        polygon history for a fresh run/reset/restore.

        Unlike invalidate_explored_area_cache() (safe to call on every
        resize/pan/zoom/theme toggle, since it only drops rendering-cache
        QPixmaps that get rebuilt from the geometry below), this drops the
        geometry itself, so a previous run's coverage can never bleed into
        a new one -- see _explored_area_paths_by_robot's docstring.

        Does not touch _explored_area_seed_mask or the BeliefMap: callers
        are responsible for publishing the new run's belief_map.
        explored_by_robot as the discrete fallback afterward (see engine.
        _publish_explored_area_source_to_canvas() / restore_navigation_
        debug_snapshot()) -- invalidate_explored_area_cache() must never
        be used as a substitute for this method.
        """
        self._explored_area_paths_by_robot = {}
        self.explored_area_polygons = []
        self.invalidate_explored_area_cache()

    def invalidate_mapped_points_cache(self):
        self._mapped_points_cache = None
        self._mapped_points_cache_size = None
        self._mapped_points_cached_count = 0

    def invalidate_obstacle_coverage_cache(self):
        self._obstacle_coverage_cache = {}
        self._obstacle_coverage_cache_count = -1

    def invalidate_obstacles_cache(self):
        self._obstacles_cache = None
        self._obstacles_cache_size = None
        self._obstacles_cache_mapped_count = -1
        self._obstacles_cache_signature = None

    def invalidate_view_transform_caches(self):
        """
        Invalidate all pixmap caches whose pixels depend on world_to_screen().

        Pan/zoom changes do not change widget size, but they do change the
        world-to-screen transform. Any cached layer drawn in screen coordinates
        must be rebuilt after camera movement.
        """
        self.invalidate_static_plot_cache()
        self.invalidate_explored_area_cache()
        self.invalidate_mapped_points_cache()
        self.invalidate_obstacles_cache()

    def invalidate_sensor_cache(self):
        self._sensor_polygon_cache = []
        self._sensor_polygon_pose = None
        self._sensor_polygon_signature = None
        self._sensor_polygon_caches_by_robot = {}

    def set_preview_config(self, config: SimulationConfig):
        previous_spacing = getattr(self.config, "mapping_point_spacing", None)
        previous_obstacles = getattr(self.config, "obstacles", None)
        previous_vision = getattr(self.config, "vision", None)
        previous_vision_model = getattr(self.config, "vision_model", None)
        previous_visualization = getattr(
            self.config, "map_visualization", DEFAULT_MAP_VISUALIZATION
        )
        previous_discovery_colors = (
            getattr(self.config, "custom_unexplored_color", DEFAULT_CUSTOM_UNEXPLORED_COLOR),
            getattr(self.config, "custom_explored_color", DEFAULT_CUSTOM_EXPLORED_COLOR),
            getattr(
                self.config,
                "custom_explored_opacity",
                DEFAULT_CUSTOM_EXPLORED_OPACITY,
            ),
        )
        previous_custom_obstacle_color = getattr(
            self.config,
            "custom_obstacle_color",
            DEFAULT_CUSTOM_OBSTACLE_COLOR,
        )
        previous_mapped_obstacle_line_width = getattr(
            self.config,
            "mapped_obstacle_line_width",
            DEFAULT_MAPPED_OBSTACLE_LINE_WIDTH,
        )
        previous_camera = (
            getattr(self.config, "camera_center_x", None),
            getattr(self.config, "camera_center_y", None),
            getattr(self.config, "camera_width", None),
            getattr(self.config, "camera_height", None),
        )
        self.config = config
        current_discovery_colors = (
            config.custom_unexplored_color,
            config.custom_explored_color,
            config.custom_explored_opacity,
        )
        if (
            previous_visualization != config.map_visualization
            or previous_discovery_colors != current_discovery_colors
        ):
            self.invalidate_static_plot_cache()
            self.invalidate_explored_area_cache()
            self._grid_overlay_cache = None
            self._grid_overlay_cache_key = None
        if (
            previous_visualization != config.map_visualization
            or previous_custom_obstacle_color != config.custom_obstacle_color
        ):
            self.invalidate_obstacles_cache()
        if previous_mapped_obstacle_line_width != config.mapped_obstacle_line_width:
            self.invalidate_mapped_points_cache()
        if previous_spacing != config.mapping_point_spacing or previous_obstacles != config.obstacles:
            self.invalidate_obstacle_coverage_cache()
            self.invalidate_obstacles_cache()
        if (
            previous_obstacles != config.obstacles
            or previous_vision != config.vision
            or previous_vision_model != config.vision_model
        ):
            self.invalidate_sensor_cache()

        current_camera = (
            getattr(config, "camera_center_x", None),
            getattr(config, "camera_center_y", None),
            getattr(config, "camera_width", None),
            getattr(config, "camera_height", None),
        )
        if previous_camera != current_camera:
            self.invalidate_view_transform_caches()

        self.update()

    def set_ipp_experiment_bundle(self, bundle) -> None:
        """Install a validated RSS26 visualization bundle, or clear it."""
        self._ipp_experiment_bundle = bundle
        self._ipp_variance_pixmap_cache = None
        self._ipp_variance_pixmap_cache_key = None
        self.update()

    def set_robot(self, robot):
        self.robot = robot
        if robot is not None:
            self.robots = []
            self.multi_path_points = []
            self.multi_planned_path_points = []
            self.multi_last_controls = []
            self.multi_exploration_targets = []
        self.update()

    def set_multi_robots(
        self,
        robots,
        path_points=None,
        last_controls=None,
        planned_path_points=None,
        exploration_targets=None,
    ):
        self.robots = list(robots or [])
        self.robot = self.robots[0] if self.robots else None
        if path_points is not None:
            # Preserve each inner list identity so the persistent trail cache
            # can distinguish normal in-place growth from a reset/new run.
            self.multi_path_points = list(path_points)
        if planned_path_points is not None:
            self.multi_planned_path_points = [list(path) for path in planned_path_points]
        if last_controls is not None:
            self.multi_last_controls = list(last_controls)
        if exploration_targets is not None:
            self.multi_exploration_targets = [None if target is None else tuple(target) for target in exploration_targets]
        self.update()

    def set_path(self, path_points):
        self.path_points = path_points
        self.update()

    def set_planned_path(self, planned_path_points):
        self.planned_path_points = planned_path_points
        self.update()

    def set_hazard_snapshot(self, snapshot: dict | None) -> None:
        """Store a read-only GROUND-TRUTH hazard snapshot. Kept for legacy/
        potential editor use -- draw_fires() (the only consumer) is no
        longer called from the live paint loop; runtime rendering uses
        set_discovered_hazard_frame() instead. Never mix the two payload
        shapes into one ambiguous dict."""
        self._hazard_snapshot = snapshot
        self._hazard_pixmap_cache = None
        self._hazard_pixmap_cache_key = None
        self.update()

    def set_discovered_hazard_frame(self, payload: dict | None) -> None:
        """Store a read-only frame of the team's DISCOVERED hazard belief --
        {"frame": HazardBeliefFrame, "bounds": ..., "resolution": ...}. This
        is what live simulation actually renders (see draw_discovered_
        hazard()); the frame's arrays are already immutable copies (see
        HazardBelief.snapshot()), so this never stores a mutable reference
        the runtime could later change out from under a paint call."""
        self._discovered_hazard_frame = payload
        self.update()

    def set_fires(self, fires) -> None:
        """Deprecated compatibility shim. Fire centers are no longer rendered.

        Runtime callers should use set_hazard_snapshot(); keeping this no-op
        prevents older helper fakes from crashing while ensuring no stale icon
        layer can reappear.
        """
        self.update()

    def set_exploration_target(self, target_xy):
        self.exploration_target_xy = None if target_xy is None else tuple(target_xy)
        self.update()

    def set_multi_exploration_targets(self, targets):
        """Store one exploration target per robot for drawing independent F markers."""
        self.multi_exploration_targets = [None if target is None else tuple(target) for target in (targets or [])]
        self.update()

    def set_explored_area_polygons(self, polygons):
        new_polygons = [list(polygon) for polygon in polygons]

        # Incremental update: if polygons were appended, paint only the new
        # polygons onto the explored-area pixmap. If the history was reset or
        # truncated, rebuild the cache once.
        previous_count = len(self.explored_area_polygons)
        self.explored_area_polygons = new_polygons

        if len(new_polygons) == 0:
            self.invalidate_explored_area_cache()
        elif (
            self._explored_area_cache is not None
            and self._explored_area_cache_size == self.size()
            and len(new_polygons) > previous_count
            and previous_count == self._explored_area_cached_count
        ):
            for polygon in new_polygons[previous_count:]:
                self.paint_explored_polygon_to_cache(polygon)
            self._explored_area_cached_count = len(new_polygons)
        else:
            self.rebuild_explored_area_cache()

        self.update()

    def append_explored_area_polygon(self, polygon: list[tuple[float, float]], robot_index: int | None = None):
        """
        Append one explored sensor footprint without copying the whole history.

        For single-robot mode the footprint is painted into the standard blue
        homogeneous cache. Multi-robot Current mode keeps one attributed cache
        per robot; shared monochrome/custom styles use one team-union cache so
        translucent overlaps remain uniform.
        """
        if len(polygon) < 3:
            return

        polygon_copy = list(polygon)
        self.explored_area_polygons.append(polygon_copy)
        if len(self.explored_area_polygons) > EXPLORED_POLYGON_HISTORY_LIMIT:
            self.explored_area_polygons = self.explored_area_polygons[-EXPLORED_POLYGON_HISTORY_LIMIT:]

        # Continuous, world-coordinate geometry -- the authoritative visual
        # source (see _explored_area_paths_by_robot's docstring). O(1): just
        # adds this one polygon as a new closed subpath onto whatever this
        # robot's QPainterPath already has. Never unions/simplifies/rebuilds
        # the accumulated path here -- that cost must never scale with
        # sweep count on every single sweep, only rebuild_explored_area_
        # cache() (an invalidation-triggered, not per-sweep, event) pays
        # the cost of rasterizing the whole path to screen space.
        world_path = self._explored_area_paths_by_robot.get(robot_index)
        if world_path is None:
            world_path = QPainterPath()
            world_path.setFillRule(Qt.WindingFill)
            self._explored_area_paths_by_robot[robot_index] = world_path
        world_path.moveTo(polygon_copy[0][0], polygon_copy[0][1])
        for x, y in polygon_copy[1:]:
            world_path.lineTo(x, y)
        world_path.closeSubpath()

        # A stale cache (invalidate_explored_area_cache() -- theme toggle,
        # resize, pan/zoom) means whatever is cached, combined or
        # per-robot, no longer matches this canvas size and must be rebuilt
        # from the authoritative source (the seed mask, if set -- see
        # rebuild_explored_area_cache()) before this new footprint is
        # painted on top. Without this, the FIRST append after an
        # invalidation would create/paint only its own (this robot's)
        # cache from scratch, and draw_explored_area_trace() would then
        # show only that one cache -- silently hiding every other robot's,
        # or the seed mask's, still-uncached coverage.
        if self._explored_area_cache is None or self._explored_area_cache_size != self.size():
            self.rebuild_explored_area_cache()

        self.paint_explored_polygon_to_cache(polygon_copy, robot_index=robot_index)
        if robot_index is None:
            self._explored_area_cached_count = len(self.explored_area_polygons)

        self.update()

    def set_runtime_state(
        self,
        robot=None,
        path_points=None,
        last_control=None,
        simulation_time: float | None = None,
        simulation_speed: float | None = None,
    ):
        """
        Update high-frequency runtime data with a single repaint request.

        The old code called update() three times per physics tick via separate
        setters. At 60 FPS, redundant repaint requests can become visible as
        frame jitter.
        """
        if robot is not None:
            self.robot = robot
        if path_points is not None:
            self.path_points = path_points
        if last_control is not None:
            self.last_control = last_control
        if simulation_time is not None:
            self.simulation_time = float(simulation_time)
        if simulation_speed is not None:
            self.simulation_speed = float(simulation_speed)
        if self._render_throttler.should_render():
            self.update()

    def set_multi_runtime_state(
        self,
        robots=None,
        path_points=None,
        last_controls=None,
        planned_path_points=None,
        exploration_targets=None,
        simulation_time: float | None = None,
        simulation_speed: float | None = None,
    ):
        if robots is not None:
            self.robots = list(robots)
            self.robot = self.robots[0] if self.robots else None
        if path_points is not None:
            # The engine owns and appends to these inner lists.  A shallow
            # outer copy keeps canvas access isolated without defeating the
            # incremental trail cache on every physics tick.
            self.multi_path_points = list(path_points)
        if planned_path_points is not None:
            self.multi_planned_path_points = [list(path) for path in planned_path_points]
        if last_controls is not None:
            self.multi_last_controls = list(last_controls)
        if exploration_targets is not None:
            self.multi_exploration_targets = [None if target is None else tuple(target) for target in exploration_targets]
        if simulation_time is not None:
            self.simulation_time = float(simulation_time)
        if simulation_speed is not None:
            self.simulation_speed = float(simulation_speed)
        if self._render_throttler.should_render():
            self.update()

    def set_simulation_metrics(self, simulation_time: float, simulation_speed: float):
        self.simulation_time = float(simulation_time)
        self.simulation_speed = float(simulation_speed)
        self.update()

    def record_render_frame(self):
        """
        Estimate user-visible FPS from paintEvent calls.

        This deliberately measures rendering cadence, not physics updates. The
        value is refreshed about four times per second so the telemetry does not
        create extra repaint pressure by itself.
        """
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_time

        if elapsed >= 0.25:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_time = now

    def set_known_obstacles(self, obstacles):
        # Backward-compatible setter. Full obstacle rectangles are no longer
        # revealed during mapping, but this keeps older calls harmless.
        self.known_obstacles = [tuple(obstacle) for obstacle in obstacles]
        self.update()

    def set_mapped_obstacle_points(self, points):
        new_points = [tuple(point) for point in points]
        previous_count = len(self.mapped_obstacle_points)
        self.mapped_obstacle_points = new_points
        self.invalidate_obstacle_coverage_cache()
        self.invalidate_obstacles_cache()

        if len(new_points) == 0:
            self.invalidate_mapped_points_cache()
        elif (
            self._mapped_points_cache is not None
            and self._mapped_points_cache_size == self.size()
            and len(new_points) > previous_count
            and previous_count == self._mapped_points_cached_count
        ):
            self.paint_mapped_points_to_cache(new_points[previous_count:])
            self._mapped_points_cached_count = len(new_points)
        else:
            self.rebuild_mapped_points_cache()

        self.update()

    def append_mapped_obstacle_points(self, points: list[tuple[float, float]]):
        """
        Append newly sensed obstacle points without copying/rebuilding the full
        point cloud cache.

        This removes a growing cost that appeared late in long simulations. The
        gray obstacle opacity cache is refreshed only every
        OBSTACLE_VISUAL_REFRESH_POINT_STEP points because it is visual feedback,
        not collision logic.
        """
        if not points:
            return

        new_points = [tuple(point) for point in points]
        self.mapped_obstacle_points.extend(new_points)

        if (
            self._mapped_points_cache is None
            or self._mapped_points_cache_size != self.size()
        ):
            self.rebuild_mapped_points_cache()
        else:
            self.paint_mapped_points_to_cache(new_points)
            self._mapped_points_cached_count = len(self.mapped_obstacle_points)

        # Do not rebuild the obstacle opacity layer after every single sensor
        # point. That was the main cause of FPS falling as mapping progressed.
        if (
            self._obstacles_cache_mapped_count < 0
            or len(self.mapped_obstacle_points) - self._obstacles_cache_mapped_count
            >= OBSTACLE_VISUAL_REFRESH_POINT_STEP
        ):
            self.invalidate_obstacle_coverage_cache()
            self.invalidate_obstacles_cache()

        self.update()

    def _append_status_history(self, message: str) -> None:
        raw_message = str(message).strip()
        if not raw_message:
            return

        timestamp = time.strftime("%H:%M:%S")
        for line in raw_message.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = f"[{timestamp}] {line}"

            # Avoid flooding the console with repeated status messages emitted by
            # periodic replanning gates. The latest visible status is still updated
            # every time; only identical consecutive console lines are collapsed.
            if self.status_history and self.status_history[-1].endswith(line):
                continue

            self.status_history.append(entry)

        if len(self.status_history) > self.status_history_limit:
            self.status_history = self.status_history[-self.status_history_limit:]

    def append_console_message(self, message: str) -> None:
        """Append a message to the console history without changing the top status."""
        self._append_status_history(message)

    def set_status(self, message: str):
        self.status_message = str(message)
        self._append_status_history(self.status_message)
        self.update()

    def status_history_lines(self) -> list[str]:
        return list(self.status_history)

    def clear_status_history(self) -> None:
        self.status_history.clear()
        self._append_status_history("Console cleared.")
        self.update()

    def set_last_control(self, control):
        self.last_control = control
        self.update()

    def plot_rect(self):
        return self.rect().adjusted(
            self.plot_margin_left,
            self.plot_margin_top,
            -self.plot_margin_right,
            -self.plot_margin_bottom,
        )

    def editor_view_span_world(self) -> tuple[float, float]:
        """Return the world span currently visible in editor mode."""
        zoom = max(0.10, float(self.editor_zoom))
        return (
            max(0.25, (WORLD_X_MAX - WORLD_X_MIN) / zoom),
            max(0.25, (WORLD_Y_MAX - WORLD_Y_MIN) / zoom),
        )

    def simulation_camera_span_world(self) -> tuple[float, float]:
        """Return the simulation camera span stored in the config."""
        return (
            max(0.50, float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN))),
            max(0.50, float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN))),
        )

    def active_view_center_world(self) -> tuple[float, float]:
        if self.editor_mode:
            return (float(self.editor_pan_offset[0]), float(self.editor_pan_offset[1]))
        return (
            float(getattr(self.config, "camera_center_x", (WORLD_X_MIN + WORLD_X_MAX) / 2.0)),
            float(getattr(self.config, "camera_center_y", (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0)),
        )

    def active_view_span_world(self) -> tuple[float, float]:
        if self.editor_mode:
            return self.editor_view_span_world()
        return self.simulation_camera_span_world()

    def active_view_bounds_world(self) -> tuple[float, float, float, float]:
        """Return left, right, bottom, top of the visible world rectangle."""
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.active_view_span_world()
        return (
            center_x - span_x / 2.0,
            center_x + span_x / 2.0,
            center_y - span_y / 2.0,
            center_y + span_y / 2.0,
        )

    # ------------------------------------------------------------------
    # Logical viewport vs. render viewport.
    #
    # The LOGICAL viewport (logical_view_span_world()/logical_view_bounds_
    # world(), currently plain aliases of active_view_span_world()/
    # active_view_bounds_world() above) is the user-configured/pan-zoom-
    # navigated rectangle: camera_center_x/y +/- camera_width/height/2 in
    # simulation mode, or the editor pan/zoom rectangle in editor mode. It
    # is what the editable viewport frame draws, what persists in
    # SimulationConfig/.sim files, and what the exploration-coverage
    # metric's ROI is built from (see engine.estimated_explored_percent())
    # -- it must never change just because the canvas resized, the theme
    # toggled, or a panel opened/closed.
    #
    # The RENDER viewport (render_view_span_world()/render_view_bounds_
    # world()) is that same rectangle expanded (via
    # _fit_world_span_to_plot_aspect(), never cropped, never independently
    # stretched) to match plot_rect()'s aspect ratio, so world_to_screen()/
    # screen_to_world() apply a uniform X/Y scale and geometry never
    # distorts. It is recomputed on demand from the logical viewport plus
    # plot_rect() -- never written back to SimulationConfig -- and is what
    # every render/culling call actually uses.
    # ------------------------------------------------------------------

    def logical_view_span_world(self) -> tuple[float, float]:
        """The configured/navigated logical viewport span -- see
        active_view_span_world(). Distinct name for clarity against
        render_view_span_world()."""
        return self.active_view_span_world()

    def logical_view_bounds_world(self) -> tuple[float, float, float, float]:
        """The configured/navigated logical viewport bounds -- see
        active_view_bounds_world(). Distinct name for clarity against
        render_view_bounds_world()."""
        return self.active_view_bounds_world()

    def render_view_span_world(self) -> tuple[float, float]:
        """logical_view_span_world(), expanded to plot_rect()'s aspect
        ratio (see _fit_world_span_to_plot_aspect()) -- what world_to_
        screen()/screen_to_world() actually use, so rendered geometry
        (circles, squares, FoV footprints, obstacles, routes) never
        distorts just because the configured viewport's aspect ratio
        differs from the canvas's."""
        span_x, span_y = self.logical_view_span_world()
        rect = self.plot_rect()
        return _fit_world_span_to_plot_aspect(span_x, span_y, rect.width(), rect.height())

    def render_view_bounds_world(self) -> tuple[float, float, float, float]:
        """render_view_span_world(), centered on the same world point as
        logical_view_bounds_world() -- the expansion is always symmetric
        around the unchanged logical center, never a one-sided crop."""
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.render_view_span_world()
        return (
            center_x - span_x / 2.0,
            center_x + span_x / 2.0,
            center_y - span_y / 2.0,
            center_y + span_y / 2.0,
        )

    def _view_transform_signature(self) -> tuple:
        """Cheap signature capturing everything world_to_screen() depends
        on (widget size, view center/zoom/pan) -- any screen-space cache
        keyed on this must be rebuilt when it changes. Shared by
        draw_executed_path()/draw_planned_route()'s own caches."""
        return (
            self.width(),
            self.height(),
            tuple(round(float(bound), 3) for bound in self.render_view_bounds_world()),
        )

    def world_to_screen(self, x: float, y: float):
        rect = self.plot_rect()
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.render_view_span_world()
        sx = rect.left() + (rect.width() / 2.0) + (float(x) - center_x) * (rect.width() / span_x)
        sy = rect.bottom() - (rect.height() / 2.0) - (float(y) - center_y) * (rect.height() / span_y)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float):
        rect = self.plot_rect()
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.render_view_span_world()
        x = center_x + ((float(sx) - (rect.left() + rect.width() / 2.0)) / rect.width()) * span_x
        y = center_y - ((float(sy) - (rect.bottom() - rect.height() / 2.0)) / rect.height()) * span_y
        return x, y

    def telemetry_rect(self):
        """Footer rectangle now occupied by the runtime action bar."""
        r = self.rect()
        return r.adjusted(30, r.height() - 58, -30, -12)

    def metrics_rect(self) -> QRectF:
        """
        Center-top badge for FPS, simulation time and simulation speed.
        """
        width = min(272.0, max(224.0, self.width() * 0.30))
        height = 25.0
        eye_width = 28.0
        gap = 6.0
        group_width = width + gap + eye_width
        x = (self.width() - group_width) / 2.0
        y = 16.0
        return QRectF(x, y, width, height)

    def metrics_eye_rect(self) -> QRectF:
        """Return the clickable eye button rectangle."""
        height = 25.0
        eye_width = 28.0
        y = 16.0

        if self.metrics_visible:
            metrics = self.metrics_rect()
            return QRectF(metrics.right() + 6.0, y, eye_width, height)

        # When metrics are hidden, keep only the eye button centered so the user
        # can bring the counters back without searching elsewhere.
        return QRectF((self.width() - eye_width) / 2.0, y, eye_width, height)

    def metrics_reserved_rect(self) -> QRectF:
        """Area reserved by the metric controls in the header row.

        Navigation Debug's activator is the navigation_snapshot_bar docked
        above the canvas (main_window._build_navigation_snapshot_bar()), not
        a painted header control, so this only needs to reserve space for
        the FPS/metrics badge + its own eye button.
        """
        eye = self.metrics_eye_rect()
        if not self.metrics_visible:
            return eye
        metrics = self.metrics_rect()
        return QRectF(metrics.left(), metrics.top(), eye.right() - metrics.left(), metrics.height())


    def multi_robot_screen_positions(self) -> list[tuple[int, float, float, RobotStartConfig]]:
        if "Multiple" not in self.config.agent_mode:
            return []

        robots = normalized_robot_start_configs(self.config)
        positions: list[tuple[int, float, float, RobotStartConfig]] = []
        for index, robot_cfg in enumerate(robots):
            sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
            positions.append((index, sx, sy, robot_cfg))
        return positions

    def pixels_per_meter(self) -> float:
        span_x, _ = self.render_view_span_world()
        return max(1.0, self.plot_rect().width() / max(0.1, span_x))

    def robot_index_at_screen_position(self, sx: float, sy: float) -> tuple[int, RobotStartConfig] | None:
        """
        Return the preview robot under the cursor before the simulation starts.

        Index convention:
            -1  -> single-robot preview
             0+ -> multi-robot preview robot index

        Runtime robots are intentionally not draggable here. Dragging during
        simulation would teleport the state and invalidate dynamics/collision
        metrics.
        """
        if self.robot is not None or self.robots:
            return None

        px_per_meter = self.pixels_per_meter()
        body_px = max(7.0, float(self.config.body_radius) * px_per_meter)
        hit_radius = max(13.0, body_px + 5.0)

        if "Multiple" not in self.config.agent_mode:
            rx, ry = self.world_to_screen(float(self.config.x), float(self.config.y))
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return -1, RobotStartConfig(
                    x=float(self.config.x),
                    y=float(self.config.y),
                    theta=float(self.config.theta),
                    v=float(self.config.v),
                )
            return None

        # Reverse order so the visually topmost/highest-index robot is easier to pick.
        for index, rx, ry, robot_cfg in reversed(self.multi_robot_screen_positions()):
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return index, robot_cfg

        return None

    _HUMAN_DEMO_CANDIDATE_CLICK_TOLERANCE_PX = 14.0

    def set_human_demo_mode(self, active: bool) -> None:
        """Toggle click routing between the normal goal/fire flow and
        Human Demonstration's robot/candidate selection flow."""

        self._human_demo_mode_active = bool(active)
        if not self._human_demo_mode_active:
            self._human_demo_candidate_markers = ()
        self.update()

    def set_human_demo_candidate_markers(
        self, markers: list[tuple[str, float, float, bool]]
    ) -> None:
        """Set the frozen candidate markers to render/hit-test for the
        currently-focused robot. Never computed here -- always supplied
        by the host from the already-frozen candidate pool."""

        self._human_demo_candidate_markers = tuple(
            (str(candidate_id), float(x), float(y), bool(enabled))
            for candidate_id, x, y, enabled in markers
        )
        self.update()

    def runtime_robot_index_at_screen_position(self, sx: float, sy: float) -> int | None:
        """Hit-test the *live, running* robots (unlike
        robot_index_at_screen_position(), which only ever sees pre-run
        preview robots). Used by Human Demonstration's robot-selection
        click flow; the returned index is exactly the ``robot_id`` used in
        CoordinationRequest (see engine.multi_robot_coordination_states():
        robot_id is the 0-based position in self.robots)."""

        px_per_meter = self.pixels_per_meter()
        body_px = max(7.0, float(self.config.body_radius) * px_per_meter)
        hit_radius = max(13.0, body_px + 5.0)

        for index in range(len(self.robots) - 1, -1, -1):
            robot = self.robots[index]
            rx, ry = self.world_to_screen(float(robot.x), float(robot.y))
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return index
        return None

    def human_demo_candidate_marker_at_screen_position(self, sx: float, sy: float) -> str | None:
        """Hit-test the frozen candidate markers with an explicit pixel
        tolerance. Returns the candidate_id, never a coordinate, and never
        matches a disabled candidate."""

        best_id: str | None = None
        best_distance = self._HUMAN_DEMO_CANDIDATE_CLICK_TOLERANCE_PX
        for candidate_id, wx, wy, enabled in self._human_demo_candidate_markers:
            if not enabled:
                continue
            mx, my = self.world_to_screen(wx, wy)
            distance = math.hypot(float(sx) - mx, float(sy) - my)
            if distance <= best_distance:
                best_distance = distance
                best_id = candidate_id
        return best_id

    def draw_human_demo_candidate_markers(self, painter: QPainter) -> None:
        """Draw the frozen candidate frontier markers for Human
        Demonstration mode. Purely presentational -- the marker list comes
        entirely from set_human_demo_candidate_markers(); nothing here
        computes or regenerates a candidate."""

        if not self._human_demo_mode_active or not self._human_demo_candidate_markers:
            return

        painter.save()
        enabled_pen = QPen(QColor(255, 176, 32), 2.5)
        enabled_brush = QBrush(QColor(255, 176, 32, 90))
        disabled_pen = QPen(QColor(150, 150, 150), 1.5)
        disabled_brush = QBrush(QColor(150, 150, 150, 60))
        radius_px = 9.0
        for _candidate_id, wx, wy, enabled in self._human_demo_candidate_markers:
            sx, sy = self.world_to_screen(wx, wy)
            painter.setPen(enabled_pen if enabled else disabled_pen)
            painter.setBrush(enabled_brush if enabled else disabled_brush)
            painter.drawEllipse(QRectF(sx - radius_px, sy - radius_px, radius_px * 2, radius_px * 2))
        painter.restore()

    def set_editor_mode(self, enabled: bool) -> None:
        self.editor_mode = bool(enabled)
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_pan_active = False
        self.editor_last_pan_pos = None
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        if not self.editor_mode:
            self.editor_pan_offset = (0.0, 0.0)
            self.editor_zoom = 1.0
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        self.fit_to_obstacles(self.config.obstacles)

    def set_editor_tool(self, tool: str) -> None:
        self.editor_tool = str(tool)
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        self.editor_view_changed.emit()
        self.update()

    def set_editor_drag_start(self, start_xy: tuple[float, float]) -> None:
        self.editor_drag_start = tuple(start_xy)
        self.editor_drag_current = tuple(start_xy)
        self.editor_preview_points = [tuple(start_xy)]
        self.update()

    def set_editor_brush_size(self, brush_size: float) -> None:
        self.editor_brush_size = max(0.05, float(brush_size))
        self.invalidate_obstacles_cache()
        self.update()

    def set_editor_interaction_mode(self, mode: str) -> None:
        mode_name = str(mode).lower()
        self.editor_interaction_mode = "move" if mode_name == "move" else "paint"
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_pan_active = False
        self.editor_last_pan_pos = None
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        self.update()

    def fit_to_obstacles(self, obstacles: list[tuple[float, float, float, float]]) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return

        if not obstacles:
            self.editor_pan_offset = ((WORLD_X_MIN + WORLD_X_MAX) / 2.0, (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0)
            self.editor_zoom = 1.0
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        xs = [obstacle[0] for obstacle in obstacles] + [obstacle[0] + obstacle[2] for obstacle in obstacles]
        ys = [obstacle[1] for obstacle in obstacles] + [obstacle[1] + obstacle[3] for obstacle in obstacles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        padding_x = max(0.5, span_x * 0.08)
        padding_y = max(0.5, span_y * 0.08)

        world_span_x = max(span_x + padding_x * 2.0, 1.0)
        world_span_y = max(span_y + padding_y * 2.0, 1.0)
        zoom_x = (WORLD_X_MAX - WORLD_X_MIN) / world_span_x
        zoom_y = (WORLD_Y_MAX - WORLD_Y_MIN) / world_span_y
        self.editor_zoom = max(0.35, min(3.0, min(zoom_x, zoom_y)))
        self.editor_pan_offset = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        self.update()

    def editor_status_text(self) -> str:
        zoom_percent = max(0.0, self.editor_zoom * 100.0)
        cam_x = float(getattr(self.config, "camera_center_x", 0.0))
        cam_y = float(getattr(self.config, "camera_center_y", 0.0))
        cam_w = float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN))
        cam_h = float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN))
        return (
            f"Editor zoom {zoom_percent:.0f}%  ·  View center ({self.editor_pan_offset[0]:.1f}, {self.editor_pan_offset[1]:.1f})  ·  "
            f"Simulation camera center ({cam_x:.1f}, {cam_y:.1f}) size {cam_w:.1f} × {cam_h:.1f} m"
        )

    def camera_bounds_world(self) -> tuple[float, float, float, float]:
        """Return simulation camera left, right, bottom, top in world coordinates."""
        center_x = float(getattr(self.config, "camera_center_x", 0.0))
        center_y = float(getattr(self.config, "camera_center_y", 0.0))
        width = max(0.50, float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN)))
        height = max(0.50, float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN)))
        return (
            center_x - width / 2.0,
            center_x + width / 2.0,
            center_y - height / 2.0,
            center_y + height / 2.0,
        )

    def camera_rect_screen(self) -> QRectF:
        left, right, bottom, top = self.camera_bounds_world()
        x1, y1 = self.world_to_screen(left, bottom)
        x2, y2 = self.world_to_screen(right, top)
        return QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def set_camera_view(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        *,
        emit_signal: bool = False,
    ) -> None:
        """Update the red editor camera rectangle and simulation viewport."""
        width = max(0.50, float(width))
        height = max(0.50, float(height))
        self.config.camera_center_x = float(center_x)
        self.config.camera_center_y = float(center_y)
        self.config.camera_width = width
        self.config.camera_height = height
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        if emit_signal:
            self.editor_camera_changed.emit((
                self.config.camera_center_x,
                self.config.camera_center_y,
                self.config.camera_width,
                self.config.camera_height,
            ))
        self.update()

    def camera_handle_at_screen_position(self, sx: float, sy: float) -> str | None:
        """Return resize/move handle under the cursor for the camera frame."""
        rect = self.camera_rect_screen()
        if rect.isNull() or rect.width() <= 0.0 or rect.height() <= 0.0:
            return None

        point = QPointF(float(sx), float(sy))
        handle_radius = 10.0
        corners = {
            "nw": rect.topLeft(),
            "ne": rect.topRight(),
            "sw": rect.bottomLeft(),
            "se": rect.bottomRight(),
        }
        for name, corner in corners.items():
            if math.hypot(point.x() - corner.x(), point.y() - corner.y()) <= handle_radius:
                return name

        edge_tol = 7.0
        if rect.left() - edge_tol <= point.x() <= rect.right() + edge_tol:
            if abs(point.y() - rect.top()) <= edge_tol:
                return "n"
            if abs(point.y() - rect.bottom()) <= edge_tol:
                return "s"
        if rect.top() - edge_tol <= point.y() <= rect.bottom() + edge_tol:
            if abs(point.x() - rect.left()) <= edge_tol:
                return "w"
            if abs(point.x() - rect.right()) <= edge_tol:
                return "e"

        if rect.adjusted(0, 0, 0, 0).contains(point):
            return "move"
        return None

    def update_camera_from_drag(self, current_world: tuple[float, float]) -> None:
        if (
            self.editor_camera_active_handle is None
            or self.editor_camera_drag_start_world is None
            or self.editor_camera_start_bounds is None
        ):
            return

        start_x, start_y = self.editor_camera_drag_start_world
        dx = float(current_world[0]) - start_x
        dy = float(current_world[1]) - start_y
        left, right, bottom, top = self.editor_camera_start_bounds
        handle = self.editor_camera_active_handle
        min_size = 0.75

        if handle == "move":
            left += dx
            right += dx
            bottom += dy
            top += dy
        else:
            if "w" in handle:
                left += dx
            if "e" in handle:
                right += dx
            if "s" in handle:
                bottom += dy
            if "n" in handle:
                top += dy

            if right - left < min_size:
                if "w" in handle:
                    left = right - min_size
                else:
                    right = left + min_size
            if top - bottom < min_size:
                if "s" in handle:
                    bottom = top - min_size
                else:
                    top = bottom + min_size

        center_x = (left + right) / 2.0
        center_y = (bottom + top) / 2.0
        width = right - left
        height = top - bottom
        self.set_camera_view(center_x, center_y, width, height, emit_signal=True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()

            if self.metrics_eye_rect().contains(QPointF(pos.x(), pos.y())):
                self.metrics_visible = not self.metrics_visible
                self.update()
                return

            if self.plot_rect().contains(pos.toPoint()):
                if self.editor_mode and not self.robot and not self.robots:
                    # Pan/Zoom mode is exclusive. It must never select, move,
                    # erase, resize the viewport, or create obstacles.
                    if self.editor_interaction_mode == "move":
                        self.editor_pan_active = True
                        self.editor_last_pan_pos = (pos.x(), pos.y())
                        self.setCursor(Qt.ClosedHandCursor)
                        return

                    world_x, world_y = self.screen_to_world(pos.x(), pos.y())

                    if self.editor_tool == "camera":
                        handle = self.camera_handle_at_screen_position(pos.x(), pos.y())
                        if handle is not None:
                            self.editor_camera_interaction_started.emit()
                            self.editor_camera_active_handle = handle
                            self.editor_camera_drag_start_world = (float(world_x), float(world_y))
                            self.editor_camera_start_bounds = self.camera_bounds_world()
                            self.setCursor(Qt.ClosedHandCursor if handle == "move" else Qt.SizeAllCursor)
                            return
                        # Camera mode should never create obstacles by accident.
                        return

                    if self.editor_tool == "erase":
                        self.editor_drag_start = (world_x, world_y)
                        self.editor_drag_current = self.editor_drag_start
                        self.editor_interaction_started.emit(self.editor_drag_start)
                        self.update()
                        return

                    # Object movement is no longer a separate tool. In edit mode,
                    # clicking an existing connected object starts dragging it;
                    # clicking empty space keeps the currently selected draw tool.
                    group_indices = find_obstacle_group_at(self.config.obstacles, (world_x, world_y))
                    if group_indices:
                        self.editor_obstacle_move_started.emit()
                        self.editor_obstacle_drag_index = int(group_indices[-1])
                        self.editor_obstacle_drag_indices = list(group_indices)
                        self.editor_obstacle_drag_last_world = (float(world_x), float(world_y))
                        self.setCursor(Qt.ClosedHandCursor)
                        self.update()
                        return

                    self.editor_drag_start = (world_x, world_y)
                    self.editor_drag_current = self.editor_drag_start
                    self.editor_interaction_started.emit(self.editor_drag_start)
                    self.update()
                    return

                hit = self.robot_index_at_screen_position(pos.x(), pos.y())
                if hit is not None:
                    index, robot_cfg = hit
                    self.dragging_robot_index = index
                    world_x, world_y = self.screen_to_world(pos.x(), pos.y())
                    self.dragging_robot_offset = (robot_cfg.x - world_x, robot_cfg.y - world_y)
                    if index >= 0:
                        self.robotSelected.emit(index)
                    self.setCursor(Qt.ClosedHandCursor)
                    return

                if self._human_demo_mode_active:
                    # Robot click first: hit-test the live, running robots.
                    # Frontier-marker click only when no robot was hit --
                    # both use their own explicit pixel tolerance, never an
                    # arbitrary/closest-point fallback.
                    robot_index = self.runtime_robot_index_at_screen_position(pos.x(), pos.y())
                    if robot_index is not None:
                        self.humanDemoRobotClicked.emit(robot_index)
                        return
                    candidate_id = self.human_demo_candidate_marker_at_screen_position(
                        pos.x(), pos.y()
                    )
                    if candidate_id is not None:
                        self.humanDemoCandidateClicked.emit(candidate_id)
                    return

                x, y = self.screen_to_world(pos.x(), pos.y())
                # Goal-seeking: click relocates G (the only mode where it is
                # executable). Exploration: G is not executable (see
                # navigation_modes.py's docstring) -- a click there instead
                # adds/removes a fire hazard; main_window.py decides which
                # by checking proximity to existing fires.
                if is_goal_seeking_planner(self.config.exploration_planner):
                    self.goalClicked.emit(x, y)
                else:
                    self.fireToggleRequested.emit(x, y)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.cursor_coordinates_enabled and self.plot_rect().contains(int(pos.x()), int(pos.y())):
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            self.cursor_coordinate_position = (float(pos.x()), float(pos.y()))
            self.cursor_coordinate_world = (float(world_x), float(world_y))
        else:
            self.cursor_coordinate_position = None
            self.cursor_coordinate_world = None
        self.update()
        if self.editor_mode and self.editor_pan_active and self.editor_last_pan_pos is not None:
            pos = event.position()
            dx = pos.x() - self.editor_last_pan_pos[0]
            dy = pos.y() - self.editor_last_pan_pos[1]
            span_x, span_y = self.editor_view_span_world()
            self.editor_pan_offset = (
                self.editor_pan_offset[0] - dx * span_x / max(1.0, self.plot_rect().width()),
                self.editor_pan_offset[1] + dy * span_y / max(1.0, self.plot_rect().height()),
            )
            self.editor_last_pan_pos = (pos.x(), pos.y())
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        if self.editor_mode and self.editor_camera_active_handle is not None:
            pos = event.position()
            self.update_camera_from_drag(self.screen_to_world(pos.x(), pos.y()))
            return

        if self.editor_mode and self.editor_obstacle_drag_indices and self.editor_obstacle_drag_last_world is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            last_x, last_y = self.editor_obstacle_drag_last_world
            dx = float(world_x) - float(last_x)
            dy = float(world_y) - float(last_y)
            if abs(dx) > 1.0e-9 or abs(dy) > 1.0e-9:
                self.editor_obstacle_moved.emit((tuple(self.editor_obstacle_drag_indices), dx, dy))
                self.editor_obstacle_drag_last_world = (float(world_x), float(world_y))
            self.update()
            return

        if self.editor_mode and self.editor_drag_start is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            self.editor_drag_current = (world_x, world_y)
            if self.editor_tool == "free":
                if not self.editor_preview_points or math.hypot(world_x - self.editor_preview_points[-1][0], world_y - self.editor_preview_points[-1][1]) >= 0.05:
                    self.editor_preview_points.append((world_x, world_y))
                    self.editor_interaction_progress.emit((world_x, world_y))
            self.update()
            return

        if self.dragging_robot_index is None:
            return

        pos = event.position()
        x, y = self.screen_to_world(pos.x(), pos.y())
        dx, dy = self.dragging_robot_offset
        x = clamp(x + dx, WORLD_X_MIN, WORLD_X_MAX)
        y = clamp(y + dy, WORLD_Y_MIN, WORLD_Y_MAX)
        self.robotDragged.emit(int(self.dragging_robot_index), float(x), float(y))

    def leaveEvent(self, event):
        self.cursor_coordinate_position = None
        self.cursor_coordinate_world = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.editor_mode and self.editor_pan_active:
            self.editor_pan_active = False
            self.editor_last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_camera_active_handle is not None:
            self.editor_camera_active_handle = None
            self.editor_camera_drag_start_world = None
            self.editor_camera_start_bounds = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_obstacle_drag_indices:
            self.editor_obstacle_drag_index = None
            self.editor_obstacle_drag_indices = []
            self.editor_obstacle_drag_offset = (0.0, 0.0)
            self.editor_obstacle_drag_last_world = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_drag_start is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            self.editor_drag_current = (world_x, world_y)
            self.editor_interaction_finished.emit(self.editor_drag_start, (world_x, world_y))
            self.editor_drag_start = None
            self.editor_drag_current = None
            self.update()
            return

        if self.dragging_robot_index is not None:
            self.dragging_robot_index = None
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        if not self.editor_mode:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        pos = event.position()
        world_before = self.screen_to_world(pos.x(), pos.y())

        zoom_factor = 1.10 if delta > 0 else 0.90
        self.editor_zoom = max(0.35, min(8.0, self.editor_zoom * zoom_factor))

        world_after = self.screen_to_world(pos.x(), pos.y())
        self.editor_pan_offset = (
            self.editor_pan_offset[0] + (world_before[0] - world_after[0]),
            self.editor_pan_offset[1] + (world_before[1] - world_after[1]),
        )
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        self.update()

    def paintEvent(self, event):
        self.record_render_frame()
        frame_start = time.perf_counter()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        _card_start = time.perf_counter()
        self.draw_card(painter)
        card_ms = (time.perf_counter() - _card_start) * 1000.0
        self._render_layer_ms["card"] = card_ms

        _title_start = time.perf_counter()
        self.draw_title(painter)
        title_ms = (time.perf_counter() - _title_start) * 1000.0
        self._render_layer_ms["title"] = title_ms

        self.draw_plot(painter)

        # The footer is now a real QWidget action bar. Keep the render-perf
        # bucket for compatibility, but there is no painted telemetry work.
        telemetry_ms = 0.0
        self._render_layer_ms["telemetry"] = telemetry_ms

        # Card/title/telemetry chrome is small, one-off UI decoration, not
        # a simulation/map/robot layer -- folded into "overlays" for the
        # optional [RENDER] detail line rather than given its own top-level
        # bucket. Each is also kept as its own named sub-bucket above so a
        # telemetry/card/title spike is distinguishable from the editor-
        # overlay/grid-preview/plot-border sub-layers.
        self._render_layer_ms["overlays"] += card_ms + title_ms + telemetry_ms

        self._report_render_perf(frame_start)

    def _report_render_perf(self, frame_start: float) -> None:
        """Update in-app perf diagnostics from this frame's measured paint
        time. Purely observational -- never touches self.config or any
        simulation state.

        Routine samples are NEVER printed to stdout/terminal and NEVER
        appended to the GUI console: they are only stored in
        latest_perf_status, so an optional "Show FPS" display can read the
        current numbers without any terminal or GUI console output. Only a
        genuinely severe, heavily throttled FPS drop reaches the GUI
        console, via _maybe_emit_perf_gui_warning().
        """
        paint_ms = (time.perf_counter() - frame_start) * 1000.0

        snapshot_age_ms = None
        if self.grid_overlay_enabled and self._grid_overlay_snapshot_pushed_at is not None:
            snapshot_age_ms = (time.perf_counter() - self._grid_overlay_snapshot_pushed_at) * 1000.0

        # record_frame() still throttles its returned formatted line (kept
        # for callers/tests that want the exact [PERF] text), but nothing
        # here prints or GUI-console-appends it -- only the always-current
        # rolling paint_fps/paint_ms values are kept, in latest_perf_status.
        self._render_perf_monitor.record_frame(
            paint_ms=paint_ms,
            overlay_enabled=self.grid_overlay_enabled,
            grid_resolution=self._grid_overlay_resolution,
            visible_cells=self._grid_overlay_last_visible_cells if self.grid_overlay_enabled else None,
            cache_status=self._grid_overlay_last_cache_status,
            snapshot_age_ms=snapshot_age_ms,
        )

        self.latest_perf_status = {
            "paint_fps": self._render_perf_monitor.paint_fps,
            "paint_ms": self._render_perf_monitor.paint_ms,
            "overlay_enabled": self.grid_overlay_enabled,
            "grid_resolution": self._grid_overlay_resolution,
            "visible_cells": self._grid_overlay_last_visible_cells if self.grid_overlay_enabled else None,
            "cache_status": self._grid_overlay_last_cache_status,
            "snapshot_age_ms": snapshot_age_ms,
        }

        self._maybe_emit_perf_gui_warning()

        # Optional, throttled per-layer breakdown (SIM_RENDER_DETAIL_LOG=1);
        # a no-op unless explicitly enabled -- see RenderDetailLogger.
        self._render_detail_logger.maybe_log(
            total_ms=paint_ms,
            background_ms=self._render_layer_ms.get("background", 0.0),
            map_layer_ms=self._render_layer_ms.get("map_layer", 0.0),
            grid_overlay_ms=self._render_layer_ms.get("grid_overlay", 0.0),
            grid_overlay_cache_status=self._grid_overlay_last_cache_status,
            grid_overlay_visible_cells=self._grid_overlay_last_visible_cells,
            grid_overlay_rebuild_ms=self._grid_overlay_rebuild_ms,
            grid_overlay_blit_ms=self._grid_overlay_blit_ms,
            grid_overlay_cells_ms=self._grid_overlay_cells_ms,
            grid_overlay_lines_ms=self._grid_overlay_lines_ms,
            explored_area_ms=self._render_layer_ms.get("explored_area", 0.0),
            ground_truth_obstacles_ms=self._render_layer_ms.get("ground_truth_obstacles", 0.0),
            mapped_obstacle_points_ms=self._render_layer_ms.get("mapped_obstacle_points", 0.0),
            robot_body_ms=self._render_layer_ms.get("robot_body", 0.0),
            robot_fov_ms=self._render_layer_ms.get("robot_fov", 0.0),
            robot_fov_cache_hit=self._fov_detail.get("robot_fov_cache_hit", True),
            robot_fov_compute_ms=self._fov_detail.get("robot_fov_compute_ms", 0.0),
            robot_fov_paint_ms=self._fov_detail.get("robot_fov_paint_ms", 0.0),
            route_path_ms=self._render_layer_ms.get("route_path", 0.0),
            planned_route_build_ms=self._route_detail.get("planned_route_build_ms", 0.0),
            planned_route_paint_ms=self._route_detail.get("planned_route_paint_ms", 0.0),
            executed_trail_build_ms=self._route_detail.get("executed_trail_build_ms", 0.0),
            executed_trail_paint_ms=self._route_detail.get("executed_trail_paint_ms", 0.0),
            executed_trail_points=self._route_detail.get("executed_trail_points", 0),
            executed_trail_segments_painted=self._route_detail.get("executed_trail_segments_painted", 0),
            executed_trail_cache_hit=self._route_detail.get("executed_trail_cache_hit", False),
            sensor_debug_overlay_ms=self._render_layer_ms.get("sensor_debug_overlay", 0.0),
            overlays_ms=self._render_layer_ms.get("overlays", 0.0),
            editor_overlays_ms=self._render_layer_ms.get("editor_overlays", 0.0),
            grid_preview_ms=self._render_layer_ms.get("grid_preview", 0.0),
            plot_border_ms=self._render_layer_ms.get("plot_border", 0.0),
            card_ms=self._render_layer_ms.get("card", 0.0),
            title_ms=self._render_layer_ms.get("title", 0.0),
            telemetry_ms=self._render_layer_ms.get("telemetry", 0.0),
            cache_hit=self._last_background_cache_hit,
            log=self._capture_render_detail_line,
        )

    def _capture_render_detail_line(self, line: str) -> None:
        """Retain optional render diagnostics without writing to stdout."""
        self._latest_render_detail_line = str(line)

    def _maybe_emit_perf_gui_warning(self) -> None:
        """Append a rare, heavily throttled GUI-console line when paint_fps
        is severely low -- the only case where perf diagnostics reach the
        GUI console at all, since routine samples never do (see
        _report_render_perf's latest_perf_status).

        Gated on simulation_running AND grid_overlay_enabled: a low
        paint_fps during setup/load/reset is not meaningful (nothing is
        actually rendering the overlay yet), and with the overlay off,
        Show Grid cannot be the cause -- reporting it as an "overlay is
        low fps" warning in either case would be a false lead.
        """
        if not self._simulation_running_for_perf or not self.grid_overlay_enabled:
            return

        if self._perf_gui_warning_gate.should_warn(self._render_perf_monitor.paint_fps):
            self.append_console_message(
                format_gui_perf_warning(
                    paint_fps=self._render_perf_monitor.paint_fps,
                    overlay_enabled=self.grid_overlay_enabled,
                    grid_resolution=self._grid_overlay_resolution,
                )
            )

    def draw_card(self, painter: QPainter):
        c = theme_colors(self._theme_mode)
        rect = QRectF(self.rect().adjusted(0, 0, -1, -1))
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        painter.fillPath(path, QColor(c.card_background))
        painter.setPen(QPen(QColor(c.border), 1))
        painter.drawPath(path)

    def draw_title(self, painter: QPainter):
        """
        Draw the canvas header.

        Layout rule:
            left   -> title
            center -> FPS / simulation time / speed + eye button
            right  -> short status message
        """
        c = theme_colors(self._theme_mode)
        reserved_rect = self.metrics_reserved_rect()

        # Left title. Keep it in its own small area so it never collides with
        # the centered metrics controls.
        painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
        painter.setPen(QColor(c.text_primary))
        title_rect = QRectF(24, 13, max(120.0, reserved_rect.left() - 36.0), 28)
        title = painter.fontMetrics().elidedText(
            "Simulation Preview",
            Qt.ElideRight,
            int(max(90.0, title_rect.width())),
        )
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, title)

        # Center metrics badge. The eye button remains visible even when the
        # counters are hidden.
        if self.metrics_visible:
            self.draw_metrics_badge(painter, self.metrics_rect())
        self.draw_metrics_eye_button(painter, self.metrics_eye_rect())

        # Right status. Long status messages are elided because the center
        # metrics controls have priority in this header row.
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(c.text_secondary))
        status_left = reserved_rect.right() + 16.0
        status_width = max(0.0, self.width() - status_left - 24.0)
        if status_width >= 70.0:
            status_rect = QRectF(status_left, 16, status_width, 22)
            status_text = self.editor_status_text() if self.editor_mode else self.status_message
            status = painter.fontMetrics().elidedText(
                status_text,
                Qt.ElideRight,
                int(status_rect.width()),
            )
            painter.drawText(status_rect, Qt.AlignRight | Qt.AlignVCenter, status)

    def draw_metrics_badge(self, painter: QPainter, rect: QRectF):
        """
        Draw runtime counters in a compact top-center pill.
        """
        painter.save()
        c = theme_colors(self._theme_mode)

        path = QPainterPath()
        path.addRoundedRect(rect, 12.5, 12.5)

        painter.setPen(QPen(QColor(c.border_strong), 1.0))
        painter.setBrush(QBrush(QColor(c.elevated_background)))
        painter.drawPath(path)

        dot_color = QColor(GREEN) if self.fps >= 50.0 else QColor(ORANGE)
        if self.fps < 35.0 and self.fps > 0.0:
            dot_color = QColor(RED)

        dot_x = rect.left() + 11.0
        dot_y = rect.center().y() - 3.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(QRectF(dot_x, dot_y, 6.0, 6.0))

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor(c.text_primary))

        text = (
            f"FPS {self.fps:04.1f}"
            f"  ·  {self.simulation_time:05.2f}s"
            f"  ·  {self.simulation_speed:.2f}x"
        )
        painter.drawText(
            rect.adjusted(23, 0, -8, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            text,
        )

        painter.restore()

    def draw_metrics_eye_button(self, painter: QPainter, rect: QRectF):
        """Draw the open/closed eye button used to hide/show counters."""
        painter.save()
        c = theme_colors(self._theme_mode)

        path = QPainterPath()
        path.addRoundedRect(rect, 12.5, 12.5)
        painter.setPen(QPen(QColor(c.border_strong), 1.0))
        painter.setBrush(QBrush(QColor(c.elevated_background)))
        painter.drawPath(path)

        cx = rect.center().x()
        cy = rect.center().y()
        eye_color = QColor(c.text_primary if self.metrics_visible else c.text_secondary)
        painter.setPen(QPen(eye_color, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        eye_path = QPainterPath()
        eye_path.moveTo(cx - 8.5, cy)
        eye_path.cubicTo(cx - 5.5, cy - 5.0, cx + 5.5, cy - 5.0, cx + 8.5, cy)
        eye_path.cubicTo(cx + 5.5, cy + 5.0, cx - 5.5, cy + 5.0, cx - 8.5, cy)
        painter.drawPath(eye_path)

        if self.metrics_visible:
            painter.setBrush(QBrush(eye_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(cx - 2.3, cy - 2.3, 4.6, 4.6))
        else:
            painter.setPen(QPen(eye_color, 1.7, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx - 8.0, cy + 7.0), QPointF(cx + 8.0, cy - 7.0))

        painter.restore()

    def ensure_static_plot_cache(self):
        if (
            self._static_plot_cache is not None
            and self._static_plot_cache_size == self.size()
        ):
            return

        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)

        cache_painter = QPainter(cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)

        rect = self.plot_rect()
        cache_painter.save()
        cache_painter.setClipRect(rect)
        cache_painter.fillRect(rect, self.plot_background_color())
        self.draw_grid(cache_painter, rect)
        cache_painter.restore()
        cache_painter.end()

        self._static_plot_cache = cache
        self._static_plot_cache_size = QSize(self.size())

    def polygon_to_screen_path(self, polygon: list[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        if len(polygon) < 3:
            return path

        sx, sy = self.world_to_screen(*polygon[0])
        path.moveTo(sx, sy)

        for point in polygon[1:]:
            px, py = self.world_to_screen(*point)
            path.lineTo(px, py)

        path.closeSubpath()
        return path

    def ensure_explored_area_cache(self):
        if (
            self._explored_area_cache is not None
            and self._explored_area_cache_size == self.size()
        ):
            return

        self.rebuild_explored_area_cache()

    def _explored_area_alpha(self, light_alpha: int) -> int:
        """Scale an explored-area wash alpha (tuned against the light-mode
        near-white canvas) up for dark mode, where the same low alpha over a
        near-black backdrop barely reads. Same hue/robot color either way --
        only how strongly it stands out from the backdrop changes."""
        if self.is_monochrome_discovery_mode():
            return 255
        if self.is_custom_discovery_mode():
            opacity = min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            self.config,
                            "custom_explored_opacity",
                            DEFAULT_CUSTOM_EXPLORED_OPACITY,
                        )
                    ),
                ),
            )
            return int(round(255.0 * opacity))
        if self._theme_mode == ThemeMode.DARK:
            return min(255, int(light_alpha * 2.8))
        return light_alpha

    def _paint_explored_mask_layer_to_cache(
        self,
        cache_painter: QPainter,
        layer: np.ndarray,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        color: QColor,
        cell_bounds: tuple[int, int, int, int],
    ) -> None:
        """Rasterize one 2D boolean explored-mask layer (one robot's slice)
        into screen space with an already-alpha'd color.

        Shared by _paint_explored_mask_to_cache() (draws every robot layer
        of a mask into the SAME pixmap -- historical replay / single-robot
        seed) and _paint_explored_mask_robot_slice_to_cache() (draws one
        robot's layer into that robot's own attributed cache -- live
        multi-robot seed rebuild).
        """
        x_min, _x_max, y_min, _y_max = bounds
        col_start, col_end, row_start, row_end = cell_bounds
        cache_painter.setBrush(QBrush(color))
        rows, cols = np.where(layer[row_start:row_end + 1, col_start:col_end + 1])
        for local_row, local_col in zip(rows, cols):
            row = row_start + int(local_row)
            col = col_start + int(local_col)
            x0 = x_min + col * resolution
            y0 = y_min + row * resolution
            sx0, sy0 = self.world_to_screen(x0, y0)
            sx1, sy1 = self.world_to_screen(x0 + resolution, y0 + resolution)
            cache_painter.drawRect(QRectF(
                min(sx0, sx1), min(sy0, sy1),
                abs(sx1 - sx0), abs(sy1 - sy0),
            ))

    def _paint_explored_mask_to_cache(
        self,
        cache_painter: QPainter,
        mask: np.ndarray,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        alpha: int,
    ) -> None:
        """Per-cell rasterization of an explored_by_robot boolean mask into
        screen space, one robot layer at a time, all into the SAME pixmap
        -- shared by _draw_historical_explored_area() (frozen replay of a
        selected history snapshot) and rebuild_explored_area_cache()'s
        single-robot branch (replaying the live seed; see
        set_explored_area_seed()). For a live multi-robot mask that needs
        per-robot attributed caches instead, see
        _paint_explored_mask_robot_slice_to_cache().
        """
        cell_bounds = self._grid_overlay_cell_bounds(
            resolution, {"grid": mask[0], "bounds": bounds, "resolution": resolution}
        )
        if cell_bounds is None:
            return
        cache_painter.setPen(Qt.NoPen)
        if self.is_shared_discovery_mode():
            # Custom/monochrome discovery represents team coverage as one
            # shared set. Rasterize the OR-union once so opacity cannot stack
            # where two robots explored the same cell.
            color = self.explored_area_color(None)
            color.setAlpha(alpha)
            self._paint_explored_mask_layer_to_cache(
                cache_painter,
                np.any(mask, axis=0),
                resolution,
                bounds,
                color=color,
                cell_bounds=cell_bounds,
            )
            return
        for robot_index in range(mask.shape[0]):
            color = self.explored_area_color(
                None if mask.shape[0] == 1 else robot_index
            )
            color.setAlpha(alpha)
            self._paint_explored_mask_layer_to_cache(
                cache_painter, mask[robot_index], resolution, bounds,
                color=color, cell_bounds=cell_bounds,
            )

    def _paint_explored_mask_robot_slice_to_cache(
        self,
        cache_painter: QPainter,
        mask: np.ndarray,
        robot_index: int,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        alpha: int,
    ) -> None:
        """Rasterize a single robot's slice of a live multi-robot
        explored_by_robot mask into that robot's own attributed cache --
        see rebuild_explored_area_cache()'s multi-robot branch."""
        cell_bounds = self._grid_overlay_cell_bounds(
            resolution, {"grid": mask[0], "bounds": bounds, "resolution": resolution}
        )
        if cell_bounds is None:
            return
        color = self.explored_area_color(robot_index)
        color.setAlpha(alpha)
        cache_painter.setPen(Qt.NoPen)
        self._paint_explored_mask_layer_to_cache(
            cache_painter, mask[robot_index], resolution, bounds,
            color=color, cell_bounds=cell_bounds,
        )

    def _world_explored_path_to_screen_path(self, world_path: QPainterPath) -> QPainterPath:
        """Transform a world-coordinate explored-area QPainterPath into
        screen coordinates for the canvas's CURRENT view transform (pan/
        zoom/plot_rect/Y-axis inversion) -- reuses world_to_screen() per
        path element rather than duplicating its math. `world_path` itself
        is never modified; a new QPainterPath is returned."""
        screen_path = QPainterPath()
        screen_path.setFillRule(Qt.WindingFill)
        for i in range(world_path.elementCount()):
            element = world_path.elementAt(i)
            sx, sy = self.world_to_screen(element.x, element.y)
            if element.type == QPainterPath.ElementType.MoveToElement:
                screen_path.moveTo(sx, sy)
            else:
                screen_path.lineTo(sx, sy)
        return screen_path

    def _paint_explored_path_to_cache(
        self,
        cache_painter: QPainter,
        world_path: QPainterPath,
        *,
        color: QColor,
        alpha: int,
    ) -> None:
        """Rasterize one robot's continuous, world-coordinate explored-area
        geometry into screen space -- see
        _world_explored_path_to_screen_path().

        Antialiased, like paint_explored_polygon_to_cache()'s single-sweep
        painter: smooth edges are the entire point of this continuous
        geometry (vs. the discrete mask's blocky per-cell rects, which
        stay non-antialiased). This also keeps the very first rebuild
        triggered by append_explored_area_polygon() (which repaints the
        same just-added polygon a second time via paint_explored_polygon_
        to_cache(), see its docstring) idempotent -- both passes must use
        the same antialiasing setting or the two would disagree on
        boundary-pixel coverage.
        """
        screen_path = self._world_explored_path_to_screen_path(world_path)
        fill_color = QColor(color)
        fill_color.setAlpha(alpha)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setPen(Qt.NoPen)
        cache_painter.setBrush(QBrush(fill_color))
        cache_painter.drawPath(screen_path)

    def rebuild_explored_area_cache(self):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._explored_area_cache = cache
        self._explored_area_cache_size = QSize(self.size())
        self._explored_area_cached_count = 0
        self._explored_area_caches_by_robot = {}
        self._explored_area_cache_sizes_by_robot = {}

        mask = self._explored_area_seed_mask
        paths_by_robot = self._explored_area_paths_by_robot
        int_path_indices = {
            key for key, path in paths_by_robot.items()
            if isinstance(key, int) and not path.isEmpty()
        }
        multi_robot_mask_indices = set(range(mask.shape[0])) if (mask is not None and mask.shape[0] > 1) else set()
        multi_robot_indices = int_path_indices | multi_robot_mask_indices

        if multi_robot_indices and self.is_shared_discovery_mode():
            # A shared discovery style has no per-robot visual attribution.
            # Paint the geometric union into ONE cache and apply opacity once;
            # compositing independent translucent robot caches would make
            # overlaps progressively more opaque.
            cache_painter = QPainter(cache)
            cache_painter.save()
            cache_painter.setClipRect(self.plot_rect())
            cache_painter.setCompositionMode(QPainter.CompositionMode_Source)

            combined_path = QPainterPath()
            combined_path.setFillRule(Qt.WindingFill)
            mask_fallback_indices: list[int] = []
            for robot_index in sorted(multi_robot_indices):
                world_path = paths_by_robot.get(robot_index)
                if world_path is not None and not world_path.isEmpty():
                    combined_path.addPath(world_path)
                elif mask is not None and 0 <= robot_index < mask.shape[0]:
                    mask_fallback_indices.append(robot_index)

            if not combined_path.isEmpty():
                self._paint_explored_path_to_cache(
                    cache_painter,
                    combined_path,
                    color=self.explored_area_color(None),
                    alpha=self._explored_area_alpha(24),
                )

            if mask is not None and mask_fallback_indices:
                cell_bounds = self._grid_overlay_cell_bounds(
                    self._explored_area_seed_resolution,
                    {
                        "grid": mask[0],
                        "bounds": self._explored_area_seed_bounds,
                        "resolution": self._explored_area_seed_resolution,
                    },
                )
                if cell_bounds is not None:
                    combined_mask = np.any(mask[mask_fallback_indices], axis=0)
                    color = self.explored_area_color(None)
                    color.setAlpha(self._explored_area_alpha(24))
                    cache_painter.setPen(Qt.NoPen)
                    self._paint_explored_mask_layer_to_cache(
                        cache_painter,
                        combined_mask,
                        self._explored_area_seed_resolution,
                        self._explored_area_seed_bounds,
                        color=color,
                        cell_bounds=cell_bounds,
                    )

            cache_painter.restore()
            cache_painter.end()
            self._explored_area_cached_count = len(self.explored_area_polygons)
            return

        if multi_robot_indices:
            # Multi-robot: one attributed cache per robot. Continuous
            # geometry takes priority over the discrete mask, per robot --
            # never both painted for the same robot in the same rebuild
            # (see _explored_area_paths_by_robot's docstring). A robot with
            # no continuous path yet (e.g. right after a snapshot restore
            # -- see clear_explored_area_geometry()) falls back to its mask
            # slice, same as before this geometry existed.
            for robot_index in sorted(multi_robot_indices):
                robot_cache = QPixmap(self.size())
                robot_cache.fill(Qt.transparent)
                cache_painter = QPainter(robot_cache)
                cache_painter.save()
                cache_painter.setClipRect(self.plot_rect())
                world_path = paths_by_robot.get(robot_index)
                if world_path is not None and not world_path.isEmpty():
                    self._paint_explored_path_to_cache(
                        cache_painter, world_path,
                        color=self.explored_area_color(robot_index),
                        alpha=self._explored_area_alpha(24),
                    )
                elif mask is not None and 0 <= robot_index < mask.shape[0]:
                    self._paint_explored_mask_robot_slice_to_cache(
                        cache_painter,
                        mask,
                        robot_index,
                        self._explored_area_seed_resolution,
                        self._explored_area_seed_bounds,
                        alpha=self._explored_area_alpha(24),
                    )
                cache_painter.restore()
                cache_painter.end()
                self._explored_area_caches_by_robot[robot_index] = robot_cache
                self._explored_area_cache_sizes_by_robot[robot_index] = QSize(self.size())
            return

        single_path = paths_by_robot.get(None)
        if single_path is not None and not single_path.isEmpty():
            # Single-robot: the continuous path is authoritative once it
            # exists -- the mask/explored_area_polygons are not replayed on
            # top of it here (same "one source of truth per rebuild" rule
            # as the multi-robot branch above).
            cache_painter = QPainter(cache)
            cache_painter.save()
            cache_painter.setClipRect(self.plot_rect())
            self._paint_explored_path_to_cache(
                cache_painter, single_path,
                color=self.explored_area_color(None), alpha=self._explored_area_alpha(24),
            )
            cache_painter.restore()
            cache_painter.end()
            self._explored_area_cached_count = len(self.explored_area_polygons)
            return

        if mask is not None:
            # Single-robot, no continuous path yet (e.g. right after a
            # snapshot restore): the mask already represents the complete
            # authoritative coverage, so it alone rebuilds the cache --
            # explored_area_polygons (bounded to EXPLORED_POLYGON_HISTORY_
            # LIMIT sweeps) is not replayed on top of it here, only
            # afterwards for new sweeps (see append_explored_area_polygon()).
            cache_painter = QPainter(cache)
            cache_painter.save()
            cache_painter.setClipRect(self.plot_rect())
            self._paint_explored_mask_to_cache(
                cache_painter,
                mask,
                self._explored_area_seed_resolution,
                self._explored_area_seed_bounds,
                alpha=self._explored_area_alpha(24),
            )
            cache_painter.restore()
            cache_painter.end()
            self._explored_area_cached_count = len(self.explored_area_polygons)
            return

        # Legacy fallback: no continuous path and no authoritative mask has
        # ever been published (see set_explored_area_seed()'s docstring)
        # -- e.g. isolated tests/callers that only ever set polygons
        # directly. Replay the bounded polygon history exactly as before.
        for polygon in self.explored_area_polygons:
            self.paint_explored_polygon_to_cache(polygon)
        self._explored_area_cached_count = len(self.explored_area_polygons)

    def ensure_robot_explored_area_cache(self, robot_index: int) -> QPixmap:
        cache = self._explored_area_caches_by_robot.get(int(robot_index))
        if cache is not None and self._explored_area_cache_sizes_by_robot.get(int(robot_index)) == self.size():
            return cache

        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._explored_area_caches_by_robot[int(robot_index)] = cache
        self._explored_area_cache_sizes_by_robot[int(robot_index)] = QSize(self.size())
        return cache

    def paint_explored_polygon_to_cache(self, polygon: list[tuple[float, float]], robot_index: int | None = None):
        if len(polygon) < 3:
            return

        if robot_index is None or self.is_shared_discovery_mode():
            if (
                self._explored_area_cache is None
                or self._explored_area_cache_size != self.size()
            ):
                self.rebuild_explored_area_cache()
                return
            target_cache = self._explored_area_cache
            fill_color = self.explored_area_color(None)
            fill_color.setAlpha(self._explored_area_alpha(24))
            composition_mode = QPainter.CompositionMode_Source
        else:
            target_cache = self.ensure_robot_explored_area_cache(int(robot_index))
            fill_color = self.explored_area_color(int(robot_index))
            fill_color.setAlpha(self._explored_area_alpha(24))

            # Same principle as single-robot explored area: each robot owns a
            # homogeneous cache. Repainting the same zone by the same robot
            # should not get darker over time. Different robot caches are drawn
            # on top of each other later, so overlap between robots remains
            # visually distinguishable without accumulating within one robot.
            composition_mode = QPainter.CompositionMode_Source

        path = self.polygon_to_screen_path(polygon)
        if path.isEmpty():
            return

        cache_painter = QPainter(target_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())
        cache_painter.setCompositionMode(composition_mode)
        cache_painter.setPen(Qt.NoPen)
        cache_painter.setBrush(QBrush(fill_color))
        cache_painter.drawPath(path)
        cache_painter.end()

    def ensure_mapped_points_cache(self):
        if (
            self._mapped_points_cache is not None
            and self._mapped_points_cache_size == self.size()
            and self._mapped_points_cached_count == len(self.mapped_obstacle_points)
        ):
            return
        self.rebuild_mapped_points_cache()

    def rebuild_mapped_points_cache(self):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._mapped_points_cache = cache
        self._mapped_points_cache_size = QSize(self.size())
        self._mapped_points_cached_count = 0

        if self.mapped_obstacle_points:
            self.paint_mapped_points_to_cache(self.mapped_obstacle_points)
            self._mapped_points_cached_count = len(self.mapped_obstacle_points)

    def paint_mapped_points_to_cache(self, points: list[tuple[float, float]]):
        if self._mapped_points_cache is None:
            self.rebuild_mapped_points_cache()
            return

        if not points:
            return

        cache_painter = QPainter(self._mapped_points_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())
        line_width = min(
            6.0,
            max(
                0.25,
                float(
                    getattr(
                        self.config,
                        "mapped_obstacle_line_width",
                        DEFAULT_MAPPED_OBSTACLE_LINE_WIDTH,
                    )
                ),
            ),
        )
        # Mapped obstacle samples deliberately keep their established neon
        # palette. They are sensor-derived map evidence, not the physical
        # obstacle layer configured by custom_obstacle_color.
        stroke = QColor(179, 0, 54, 210)
        fill = QColor(255, 23, 92, 235)
        cache_painter.setPen(QPen(stroke, line_width))
        cache_painter.setBrush(QBrush(fill))

        # The map sample spacing controls geometric density; this setting is
        # screen-space only and therefore cannot change occupancy/planning.
        point_radius = max(0.24, line_width * 0.5)

        for px, py in points:
            sx, sy = self.world_to_screen(px, py)
            cache_painter.drawEllipse(
                QRectF(
                    sx - point_radius,
                    sy - point_radius,
                    2 * point_radius,
                    2 * point_radius,
                )
            )

        cache_painter.end()

    def draw_plot(self, painter: QPainter):
        rect = self.plot_rect()

        painter.save()
        painter.setClipRect(rect)

        # Layer timing buckets for the optional [RENDER] detail line (see
        # RenderDetailLogger) -- cheap (a handful of time.perf_counter()
        # calls) and only ever printed when SIM_RENDER_DETAIL_LOG=1.
        _background_start = time.perf_counter()
        cache_hit = self._static_plot_cache is not None
        self.ensure_static_plot_cache()
        if self._static_plot_cache is not None:
            painter.drawPixmap(0, 0, self._static_plot_cache)
        else:
            painter.fillRect(rect, self.plot_background_color())
            self.draw_grid(painter, rect)
        self._last_background_cache_hit = cache_hit
        self._render_layer_ms["background"] = (time.perf_counter() - _background_start) * 1000.0

        _map_layer_start = time.perf_counter()
        # Persistent "Show Grid" overlay, drawn just above the background so
        # every other layer below (obstacles, mapped points, routes, robot,
        # safety radius, FoV, labels) stays clearly visible on top of it.
        # Broken into its own sub-bucket (plus draw_grid_overlay()'s own
        # cache_status/visible_cells/rebuild_ms/blit_ms fields) so a
        # rebuild spike is distinguishable from the other map_layer_ms
        # sub-layers below.
        _grid_overlay_start = time.perf_counter()
        self.draw_grid_overlay(painter, rect)
        self._render_layer_ms["grid_overlay"] = (time.perf_counter() - _grid_overlay_start) * 1000.0

        # Scientific raster for the RSS26 reproduction.  It sits above the
        # generic map/grid and below every piece of physical geometry.
        self.draw_ipp_uncertainty_heatmap(painter)

        # Always-visible physical world layers.
        # These are not "robot orders"; they are what the simulation world
        # actually contains or what the robot has already sensed.
        _explored_area_start = time.perf_counter()
        self.draw_explored_area_trace(painter)
        self._render_layer_ms["explored_area"] = (time.perf_counter() - _explored_area_start) * 1000.0

        # Ground-truth obstacles are a human-facing visual layer. They can be
        # hidden without changing the robot's partial map or planner inputs.
        _ground_truth_obstacles_start = time.perf_counter()
        if self.config.show_obstacles:
            self.draw_ground_truth_obstacles(painter)
        self._render_layer_ms["ground_truth_obstacles"] = (
            (time.perf_counter() - _ground_truth_obstacles_start) * 1000.0
        )

        # Mapped points remain visible because they represent the discovered map.
        # They are drawn above the vision/r layer and below routes/waypoints/robot.
        _mapped_obstacle_points_start = time.perf_counter()
        if self._navigation_debug_history_snapshot() is None:
            self.draw_mapped_obstacle_points(painter)
        self._render_layer_ms["mapped_obstacle_points"] = (
            (time.perf_counter() - _mapped_obstacle_points_start) * 1000.0
        )

        # Ground-truth debug overlay (Hazard Map toggle, default OFF) --
        # BELOW the discovered layer so the warm discovered heatmap always
        # reads on top of the cold ground-truth one (draw_fires(), the
        # unrelated legacy yellow/red ground-truth renderer, is still not
        # called here).
        self.draw_ground_truth_hazard_map(painter)
        # Live simulation ALWAYS renders what the team has actually
        # discovered -- see draw_discovered_hazard()'s own docstring; no
        # toggle can hide this layer, only add to it.
        self.draw_discovered_hazard(painter)
        # Independent of both heatmaps above -- its own anti-omniscience
        # source filter, no shared cache (see draw_fire_markers()'s own
        # docstring).
        self.draw_fire_markers(painter)
        self.draw_human_demo_candidate_markers(painter)
        self.draw_ipp_reference_overlay(painter)
        self._render_layer_ms["map_layer"] = (time.perf_counter() - _map_layer_start) * 1000.0

        # Robot-related layers, broken down into named sub-buckets for the
        # optional [RENDER] detail line -- moved out of map_layer/a single
        # combined robot_layer bucket so it's clear at a glance which
        # specific robot-drawing concern dominates paint cost.
        _fov_start = time.perf_counter()
        self.draw_sensor_range(painter)
        self._render_layer_ms["robot_fov"] = (time.perf_counter() - _fov_start) * 1000.0

        _sensor_debug_start = time.perf_counter()
        # Body/safety-radius rings are now drawn as part of the Navigation
        # Debug overlay itself (draw_navigation_debug_overlay()) -- no
        # separate always-available "Robot Orders" copy of the same rings.
        self._render_layer_ms["sensor_debug_overlay"] = (time.perf_counter() - _sensor_debug_start) * 1000.0

        _overlays_start = time.perf_counter()
        self.draw_editor_preview(painter)
        self.draw_editor_move_selection(painter)
        self.draw_editor_camera_frame(painter)
        _editor_overlays_ms = (time.perf_counter() - _overlays_start) * 1000.0
        self._render_layer_ms["editor_overlays"] = _editor_overlays_ms
        self._render_layer_ms["overlays"] = _editor_overlays_ms

        _route_path_start = time.perf_counter()
        self._route_detail["planned_route_build_ms"] = 0.0
        self._route_detail["planned_route_paint_ms"] = 0.0
        self._route_detail["executed_trail_build_ms"] = 0.0
        self._route_detail["executed_trail_paint_ms"] = 0.0
        self._route_detail["executed_trail_segments_painted"] = 0
        history_position, _history_total = self._nav_debug_history_position
        history_active = self.navigation_debug_enabled and history_position is not None
        if self.config.show_traveled_path and not history_active:
            if self.robots and "Multiple" in self.config.agent_mode:
                self.draw_multi_executed_paths(painter)
            else:
                self.draw_executed_path(painter)
        if self.config.show_path:
            if history_active:
                self.draw_historical_planned_route(painter)
            elif self.robots and "Multiple" in self.config.agent_mode:
                self.draw_multi_planned_routes(painter)
            else:
                self.draw_planned_route(painter)
        self._render_layer_ms["route_path"] = (time.perf_counter() - _route_path_start) * 1000.0

        _robot_body_start = time.perf_counter()
        self.draw_goal_and_robot(painter)
        self._render_layer_ms["robot_body"] = (time.perf_counter() - _robot_body_start) * 1000.0

        _nav_debug_start = time.perf_counter()
        if self.navigation_debug_enabled:
            self.draw_navigation_debug_overlay(painter)
        self._render_layer_ms["navigation_debug"] = (time.perf_counter() - _nav_debug_start) * 1000.0

        self.draw_frontier_clusters(painter)
        # Candidate inspection is intentionally the final map-space overlay:
        # Navigation Debug paints dense grid/frontier annotations and would
        # otherwise cover the focus ring selected with the panel's < > buttons.
        self.draw_frontier_candidate_inspection(painter)

        _grid_preview_start = time.perf_counter()
        # Drawn last so the temporary red preview is clearly visible over
        # every other layer while the user is comparing grid resolutions.
        self.draw_grid_resolution_preview(painter, rect)
        _grid_preview_ms = (time.perf_counter() - _grid_preview_start) * 1000.0
        self._render_layer_ms["grid_preview"] = _grid_preview_ms

        self.draw_cursor_coordinates(painter)

        painter.restore()

        _plot_border_start = time.perf_counter()
        painter.setPen(QPen(QColor(theme_colors(self._theme_mode).border), 1))
        painter.drawRect(rect)
        _plot_border_ms = (time.perf_counter() - _plot_border_start) * 1000.0
        self._render_layer_ms["plot_border"] = _plot_border_ms
        self._render_layer_ms["overlays"] += _grid_preview_ms + _plot_border_ms

    def draw_topography(self, painter: QPainter, rect):
        painter.save()
        painter.setPen(QPen(QColor(96, 110, 130, 24), 1))

        centers = [
            (rect.left() + rect.width() * 0.20, rect.top() + rect.height() * 0.28, 90, 55, 0.3),
            (rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.26, 125, 70, 1.9),
            (rect.left() + rect.width() * 0.42, rect.top() + rect.height() * 0.72, 145, 82, 2.7),
            (rect.left() + rect.width() * 0.82, rect.top() + rect.height() * 0.72, 110, 64, 0.9),
        ]

        for cx, cy, rx0, ry0, phase in centers:
            for level in range(1, 6):
                path = QPainterPath()
                rx = rx0 + level * 16
                ry = ry0 + level * 11

                for k in range(90):
                    t = 2 * math.pi * k / 89
                    wobble = 1.0 + 0.04 * math.sin(3 * t + phase)
                    px = cx + rx * wobble * math.cos(t)
                    py = cy + ry * wobble * math.sin(t)
                    if k == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)

                path.closeSubpath()
                painter.drawPath(path)

        painter.restore()

    def nice_grid_step(self, visible_span: float, pixel_span: float, target_pixels: float = 58.0) -> float:
        """Choose a readable coordinate-grid spacing for the current zoom."""
        approx_lines = max(2.0, float(pixel_span) / max(20.0, float(target_pixels)))
        raw_step = max(1.0e-9, float(visible_span) / approx_lines)
        exponent = math.floor(math.log10(raw_step))
        scale = 10.0 ** exponent
        for multiplier in (1.0, 2.0, 5.0, 10.0):
            step = multiplier * scale
            if step >= raw_step:
                return step
        return 10.0 * scale

    def format_grid_label(self, value: float, step: float) -> str:
        if abs(value) < step * 1.0e-4:
            value = 0.0
        if step >= 1.0:
            return f"{value:.0f}"
        if step >= 0.1:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def draw_grid(self, painter: QPainter, rect):
        """Draw an infinite-style coordinate grid for the current view.

        The faint hatching/labels below are canvas chrome (a backdrop aid,
        not simulated-world data) so they read from the current theme. The
        world-axis line itself (GRID_AXIS) is semantic and deliberately
        untouched -- see theme.py's module docstring."""
        c = theme_colors(self._theme_mode)
        discovery = self.is_shared_discovery_mode()
        left, right, bottom, top = self.render_view_bounds_world()
        span_x = max(0.1, right - left)
        span_y = max(0.1, top - bottom)
        step = self.nice_grid_step(min(span_x, span_y), min(rect.width(), rect.height()))
        minor_step = step / 2.0

        painter.save()

        # Minor grid.
        minor_color = self.discovery_contrast_color(18) if discovery else QColor(c.border)
        painter.setPen(QPen(minor_color, 1))
        start_x = math.floor(left / minor_step) * minor_step
        x = start_x
        while x <= right + minor_step * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += minor_step

        start_y = math.floor(bottom / minor_step) * minor_step
        y = start_y
        while y <= top + minor_step * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += minor_step

        # Major grid and coordinate labels.
        painter.setFont(QFont("Consolas", 7))
        label_color = self.discovery_contrast_color(150) if discovery else QColor(c.text_secondary)
        major_pen = QPen(
            self.discovery_contrast_color(34) if discovery else QColor(c.border_strong),
            1.15,
        )
        axis_pen = QPen(self.discovery_contrast_color(72) if discovery else GRID_AXIS, 1.8)

        x = math.floor(left / step) * step
        while x <= right + step * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            is_axis = left <= 0.0 <= right and abs(x) <= step * 1.0e-4
            painter.setPen(axis_pen if is_axis else major_pen)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))

            if rect.left() + 4 <= sx <= rect.right() - 4:
                painter.setPen(label_color)
                painter.drawText(QRectF(sx - 22, rect.bottom() - 18, 44, 14), Qt.AlignCenter, self.format_grid_label(x, step))
            x += step

        y = math.floor(bottom / step) * step
        while y <= top + step * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            is_axis = bottom <= 0.0 <= top and abs(y) <= step * 1.0e-4
            painter.setPen(axis_pen if is_axis else major_pen)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))

            if rect.top() + 6 <= sy <= rect.bottom() - 6:
                painter.setPen(label_color)
                painter.drawText(QRectF(rect.left() + 5, sy - 7, 42, 14), Qt.AlignLeft | Qt.AlignVCenter, self.format_grid_label(y, step))
            y += step

        painter.restore()

    # ------------------------------------------------------------------
    # Grid resolution preview (temporary red overlay).
    #
    # Shown while the user adjusts SimulationConfig.grid_resolution in the
    # config panel, so 0.50 vs 0.25 m/cell can be compared visually before
    # running the simulation. Purely a rendering overlay: it never mutates
    # self.config or any simulation/runtime state, and it does not rebuild
    # any occupancy/planning grid. It auto-hides itself shortly after the
    # last change via a single-shot QTimer, so it is never left on
    # permanently.
    # ------------------------------------------------------------------

    def show_grid_resolution_preview(self, resolution: float, duration_ms: int = 800) -> None:
        """Show the red grid preview at *resolution*, auto-hiding after
        *duration_ms* (default within the requested 700-1000ms range).

        Safe to call repeatedly while the user is still adjusting the
        control -- each call restarts the auto-hide timer, so the preview
        only disappears after the user stops changing the value. If the
        persistent "Show Grid" overlay is enabled, the overlay already
        keeps a grid visible permanently, so the auto-hide timer is not
        armed -- there is nothing for it to hide.
        """
        self._grid_resolution_preview_active = True
        self._grid_resolution_preview_resolution = max(0.01, float(resolution))
        if self.grid_overlay_enabled:
            self._grid_resolution_preview_timer.stop()
        else:
            self._grid_resolution_preview_timer.start(int(duration_ms))
        self.update()

    def hide_grid_resolution_preview(self) -> None:
        """Hide the red grid preview immediately."""
        self._grid_resolution_preview_active = False
        self._grid_resolution_preview_timer.stop()
        self.update()

    def is_grid_resolution_preview_active(self) -> bool:
        return bool(self._grid_resolution_preview_active)

    def grid_resolution_preview_value(self) -> float | None:
        return self._grid_resolution_preview_resolution

    def draw_grid_resolution_preview(self, painter: QPainter, rect) -> None:
        """Draw a lightweight red grid at the previewed resolution.

        Only within the visible world bounds, only while the preview is
        active. Deliberately simpler than draw_grid(): no minor/major
        distinction, no coordinate labels -- this is a quick visual
        comparison aid, not a permanent map layer.
        """
        if not self._grid_resolution_preview_active or not self._grid_resolution_preview_resolution:
            return

        resolution = self._grid_resolution_preview_resolution
        left, right, bottom, top = self.render_view_bounds_world()

        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(220, 40, 40, 190), 1))

        x = math.floor(left / resolution) * resolution
        while x <= right + resolution * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += resolution

        y = math.floor(bottom / resolution) * resolution
        while y <= top + resolution * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += resolution

        painter.restore()

    # ------------------------------------------------------------------
    # Persistent grid overlay ("Show Grid" toggle).
    #
    # Unlike the temporary preview above, this stays visible until the user
    # turns it off -- including while the simulation is running -- so it
    # uses its own state instead of the preview's auto-hide timer. Purely a
    # rendering overlay: it never mutates self.config, never rebuilds any
    # occupancy/planning grid, and the occupancy snapshot it colors is a
    # read-only copy pushed in from outside (see engine.py's
    # occupancy_grid_snapshot()), never a live reference.
    # ------------------------------------------------------------------

    def set_grid_overlay_enabled(self, enabled: bool) -> None:
        self.grid_overlay_enabled = bool(enabled)
        self._grid_overlay_cache_key = None
        self.update()

    def set_frontier_reasoning_overlay_enabled(self, enabled: bool) -> None:
        self.frontier_reasoning_overlay_enabled = bool(enabled)
        self.update()

    def set_frontier_reasoning_simulation_paused(self, paused: bool) -> None:
        self.frontier_reasoning_simulation_paused = bool(paused)
        self.update()

    def set_frontier_reasoning_decision(self, decision: dict | None) -> None:
        self.frontier_reasoning_decision = None if decision is None else dict(decision)
        if decision is None:
            self.frontier_reasoning_inspection = None
        self.update()

    def set_frontier_reasoning_inspection(self, candidate: dict | None) -> None:
        """Highlight a ranked candidate without changing the planner's target."""
        self.frontier_reasoning_inspection = None if candidate is None else dict(candidate)
        self.update()

    def set_frontier_reasoning_cluster_view_enabled(self, enabled: bool) -> None:
        self.frontier_reasoning_cluster_view_enabled = bool(enabled)
        # Frontier labels/fills are part of the cached grid overlay.
        self._grid_overlay_cache_key = None
        self.update()

    def set_frontier_reasoning_clusters(self, clusters) -> None:
        self.frontier_reasoning_clusters = tuple(dict(cluster) for cluster in (clusters or ()))
        self.update()

    def set_cursor_coordinates_enabled(self, enabled: bool) -> None:
        self.cursor_coordinates_enabled = bool(enabled)
        if not self.cursor_coordinates_enabled:
            self.cursor_coordinate_position = None
            self.cursor_coordinate_world = None
        self.update()

    def is_grid_overlay_enabled(self) -> bool:
        return bool(self.grid_overlay_enabled)

    def set_grid_cell_values_enabled(self, enabled: bool) -> None:
        self.grid_cell_values_enabled = bool(enabled)
        self._grid_overlay_cache_key = None
        self.update()

    def set_frontier_decisions_enabled(self, enabled: bool) -> None:
        self.frontier_decisions_enabled = bool(enabled)
        self._grid_overlay_cache_key = None
        self.update()

    # ------------------------------------------------------------------
    # Hazard Map / Fire Markers toggles. Independent of each other and of
    # grid_overlay_enabled above -- same "rendering-only, never touches
    # SimulationConfig/HazardBelief/planning, safe to flip mid-run" contract.
    # ------------------------------------------------------------------

    def set_hazard_map_enabled(self, enabled: bool) -> None:
        """Toggle draw_ground_truth_hazard_map() (the full ground-truth
        blue debug heatmap) only -- draw_discovered_hazard() is unaffected
        and always renders. Disabling never drops the underlying frame/
        cache -- re-enabling shows the current frame immediately on the
        next paint, with no re-decode."""
        self.show_hazard_map = bool(enabled)
        self.update()

    def is_hazard_map_enabled(self) -> bool:
        return bool(self.show_hazard_map)

    def set_fire_markers_enabled(self, enabled: bool) -> None:
        """Toggle whether draw_fire_markers() also draws UNDISCOVERED
        sources -- independent of show_hazard_map. Discovered sources are
        always drawn by draw_fire_markers() regardless of this flag."""
        self.show_fire_markers = bool(enabled)
        self.update()

    def is_fire_markers_enabled(self) -> bool:
        return bool(self.show_fire_markers)

    # ------------------------------------------------------------------
    # Navigation debug overlay. _nav_debug_snapshot is an immutable
    # NavigationDebugSnapshot pushed in from engine.py -- the canvas never
    # constructs, mutates, or recomputes any part of it.
    # ------------------------------------------------------------------

    def set_navigation_debug_enabled(self, enabled: bool) -> None:
        self.navigation_debug_enabled = bool(enabled)
        self.update()

    def is_navigation_debug_enabled(self) -> bool:
        return bool(self.navigation_debug_enabled)

    def _navigation_debug_history_snapshot(self):
        """Return the selected frozen frame, never the live snapshot."""
        position, _total = self._nav_debug_history_position
        if not self.navigation_debug_enabled or position is None:
            return None
        return self._nav_debug_snapshot

    def _decoded_navigation_debug_environment(self) -> dict | None:
        """Decode the selected snapshot's compressed map exactly once.

        The returned arrays are read-only replay data. They are never fed back
        into planning or the live BeliefMap.
        """
        snapshot = self._navigation_debug_history_snapshot()
        if snapshot is None:
            return None
        maybe_frame = getattr(snapshot, "belief_map", None)
        if maybe_frame is None or maybe_frame.unavailable or maybe_frame.value is None:
            return None
        frame = maybe_frame.value
        key = (id(frame), int(frame.revision))
        if self._nav_debug_environment_decode_key == key:
            return self._nav_debug_environment_decoded

        grid_bytes = zlib.decompress(frame.grid_zlib)
        grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape(frame.grid_shape)

        packed = np.frombuffer(
            zlib.decompress(frame.explored_packbits_zlib), dtype=np.uint8
        )
        explored_count = int(np.prod(frame.explored_shape))
        explored = np.unpackbits(
            packed, bitorder="little", count=explored_count
        ).reshape(frame.explored_shape).astype(bool, copy=False)

        decoded = {
            "frame": frame,
            "resolution": float(frame.resolution),
            "bounds": tuple(frame.bounds),
            "grid": grid,
            "explored_by_robot": explored,
            "revision": int(frame.revision),
        }
        self._nav_debug_environment_decode_key = key
        self._nav_debug_environment_decoded = decoded
        return decoded

    def _decoded_navigation_debug_hazard_belief(self) -> dict | None:
        """Decode the selected historical snapshot's HazardBeliefDebug
        exactly once per (frame identity, revision) -- same pattern as
        _decoded_navigation_debug_environment() for BeliefMapDebug.

        Returns None when the selected snapshot has no hazard_belief field
        at all (an older snapshot captured before this existed) -- callers
        must hide the hazard layer entirely in that case, never fall back
        to the live frame or ground truth.

        bounds/resolution are read from the SAME snapshot's belief_map
        frame rather than duplicated into HazardBeliefDebug -- both are
        captured from the same tick's BeliefMap/HazardBelief, which always
        share one GridGeometry (see hazard_service.RuntimeHazardService).
        """
        snapshot = self._navigation_debug_history_snapshot()
        if snapshot is None:
            return None
        maybe_hazard_belief = getattr(snapshot, "hazard_belief", None)
        if maybe_hazard_belief is None or maybe_hazard_belief.unavailable or maybe_hazard_belief.value is None:
            return None
        maybe_belief_map = getattr(snapshot, "belief_map", None)
        if maybe_belief_map is None or maybe_belief_map.unavailable or maybe_belief_map.value is None:
            return None

        frame = maybe_hazard_belief.value
        belief_map_frame = maybe_belief_map.value
        key = (id(frame), int(frame.revision))
        if self._nav_debug_hazard_belief_decode_key == key:
            return self._nav_debug_hazard_belief_decoded

        values = (
            np.frombuffer(zlib.decompress(frame.values_zlib), dtype=np.float32)
            .reshape(frame.shape)
            .copy()
        )
        observed_packed = np.frombuffer(zlib.decompress(frame.observed_packbits_zlib), dtype=np.uint8)
        observed = np.unpackbits(
            observed_packed, bitorder="little", count=int(np.prod(frame.shape))
        ).reshape(frame.shape).astype(bool, copy=False)

        decoded = {
            "values": values,
            "observed": observed,
            "revision": int(frame.revision),
            "bounds": tuple(belief_map_frame.bounds),
            "resolution": float(belief_map_frame.resolution),
        }
        self._nav_debug_hazard_belief_decode_key = key
        self._nav_debug_hazard_belief_decoded = decoded
        return decoded

    def set_navigation_reasoning_window(self, window) -> None:
        """Register the docked NavigationReasoningWindow so the 3
        setters below can forward pushes to it too. Optional -- None (the
        default) just skips forwarding, so tests that never construct the
        window are unaffected."""
        self._navigation_reasoning_window = window

    def _refresh_navigation_reasoning_window(self) -> None:
        window = getattr(self, "_navigation_reasoning_window", None)
        if window is not None:
            window.update_snapshot(self._nav_debug_snapshot, self._nav_debug_last_event, self._nav_debug_history_position)

    def set_navigation_debug_snapshot(self, snapshot) -> None:
        """Store the latest NavigationDebugSnapshot for the overlay/HUD to
        read. Deliberately never called from paintEvent or any idle/hover
        path -- only from engine.py's per-tick/per-route-result assembly --
        so pausing the simulation (which just stops those calls) leaves the
        last relevant snapshot in place across any number of repaints."""
        self._nav_debug_snapshot = snapshot
        if snapshot is None:
            self._nav_debug_environment_decode_key = None
            self._nav_debug_environment_decoded = None
            self._nav_debug_explored_cache = None
            self._nav_debug_explored_cache_key = None
            self._nav_debug_hazard_belief_decode_key = None
            self._nav_debug_hazard_belief_decoded = None
        self._refresh_navigation_reasoning_window()
        self.update()

    def navigation_debug_snapshot(self):
        return self._nav_debug_snapshot

    def set_navigation_debug_last_event(self, event) -> None:
        """Store the last RELEVANT navigation-debug event (a
        NavigationDebugEvent), independent of the always-current live
        snapshot -- see the field comment in __init__. Never called from
        paintEvent or any idle path."""
        self._nav_debug_last_event = event
        self._refresh_navigation_reasoning_window()
        self.update()

    def navigation_debug_last_event(self):
        return self._nav_debug_last_event

    def set_navigation_debug_history_position(self, position: int | None, total: int) -> None:
        """position is 1-based while stepping through history, or None
        while showing the live snapshot. Never called from paintEvent."""
        self._nav_debug_history_position = (position, int(total))
        self._refresh_navigation_reasoning_window()
        self.update()

    def navigation_debug_history_position(self) -> tuple[int | None, int]:
        return self._nav_debug_history_position

    def set_simulation_running_for_perf(self, running: bool) -> None:
        """Tell the canvas whether the simulation is actively running, for
        perf-diagnostic gating only (see _maybe_emit_perf_gui_warning and
        the grid-overlay-degraded console notice) -- does not affect
        rendering or any simulation state. A low paint_fps while idle
        (before Start, or after Reset) is not a meaningful signal and must
        not produce a console warning."""
        running = bool(running)
        if running and not self._simulation_running_for_perf:
            # Fresh run starting: if the overlay was already degraded before
            # Start was pressed, give it a fresh chance to notify now that
            # the simulation is actually running, instead of staying
            # permanently suppressed because the degrade happened while idle.
            self._grid_overlay_degraded_notice_shown = False
        self._simulation_running_for_perf = running

    def set_grid_overlay_resolution(self, resolution: float) -> None:
        self._grid_overlay_resolution = max(0.01, float(resolution))
        self.update()

    def set_grid_overlay_snapshot(self, snapshot: dict | None) -> None:
        """Store a read-only occupancy snapshot (resolution/bounds/grid) for
        cell coloring. Pass None to fall back to resolution-only grid lines
        (e.g. before the simulation has started, or no belief map yet).

        Each call bumps a version counter (rather than diffing the grid
        array's contents, which would be as expensive as the render work
        it's meant to avoid) -- draw_grid_overlay()'s cache key includes
        this version, so a genuinely new snapshot always invalidates the
        cache, and repeated pushes of "no new data" never do.
        """
        self._grid_overlay_snapshot = snapshot
        self._grid_overlay_snapshot_version += 1
        self._grid_overlay_snapshot_pushed_at = time.perf_counter()
        self.update()

    def is_grid_overlay_degraded(self) -> bool:
        return bool(self._grid_overlay_degraded)

    def grid_overlay_cache_status(self) -> str:
        return self._grid_overlay_last_cache_status

    def grid_overlay_visible_cell_count(self) -> int:
        return int(self._grid_overlay_last_visible_cells)

    def _grid_overlay_cell_bounds(
        self, resolution: float, snapshot: dict | None
    ) -> tuple[int, int, int, int] | None:
        """(col_start, col_end, row_start, row_end) of snapshot cells inside
        the current view, or None if there is no snapshot/nothing visible."""
        if snapshot is None:
            return None

        grid = snapshot.get("grid")
        bounds = snapshot.get("bounds")
        snapshot_resolution = float(snapshot.get("resolution") or resolution)
        if grid is None or bounds is None or snapshot_resolution <= 0.0:
            return None

        x_min, x_max, y_min, y_max = bounds
        left, right, bottom, top = self.render_view_bounds_world()

        col_start = max(0, int(math.floor((left - x_min) / snapshot_resolution)))
        col_end = min(grid.shape[1] - 1, int(math.ceil((right - x_min) / snapshot_resolution)))
        row_start = max(0, int(math.floor((bottom - y_min) / snapshot_resolution)))
        row_end = min(grid.shape[0] - 1, int(math.ceil((top - y_min) / snapshot_resolution)))

        if col_start > col_end or row_start > row_end:
            return None

        return col_start, col_end, row_start, row_end

    def draw_grid_overlay(self, painter: QPainter, rect) -> None:
        """Draw the persistent grid overlay: resolution grid lines, plus
        translucent occupied/free/unknown cell colors when a snapshot is
        available. Deliberately drawn just above the background/base map
        and below obstacles, mapped points, routes, and the robot, so it
        never hides them.

        Rebuilding the overlay means looping over every visible occupancy
        cell (one QPainter.drawRect() call each), which is only affordable
        once, not every frame -- so the result is cached into a QPixmap and
        reused as long as resolution/canvas size/view bounds/snapshot are
        unchanged (see _grid_overlay_cache_key below). If the number of
        visible cells exceeds MAX_GRID_OVERLAY_CELLS (e.g. a fine
        grid_resolution zoomed far out), cell coloring is skipped for that
        rebuild -- grid lines are still drawn -- so this can never freeze
        the UI trying to draw every cell.
        """
        historical_environment = self._decoded_navigation_debug_environment()
        history_active = historical_environment is not None
        overlay_requested = bool(
            self.grid_overlay_enabled
            or self.grid_cell_values_enabled
            or self.frontier_decisions_enabled
        )
        if not overlay_requested and not history_active:
            self._grid_overlay_last_cache_status = "off"
            self._grid_overlay_last_visible_cells = 0
            self._grid_overlay_degraded = False
            self._grid_overlay_rebuild_ms = 0.0
            self._grid_overlay_blit_ms = 0.0
            self._grid_overlay_cells_ms = 0.0
            self._grid_overlay_lines_ms = 0.0
            return

        if history_active:
            resolution = float(historical_environment["resolution"])
            snapshot = historical_environment
            snapshot_version = ("history", int(historical_environment["revision"]))
        else:
            resolution = self._grid_overlay_resolution
            snapshot = self._grid_overlay_snapshot
            snapshot_version = ("live", self._grid_overlay_snapshot_version)

        cell_bounds = self._grid_overlay_cell_bounds(resolution, snapshot)
        if cell_bounds is not None:
            col_start, col_end, row_start, row_end = cell_bounds
            visible_cells = (col_end - col_start + 1) * (row_end - row_start + 1)
        else:
            visible_cells = 0

        degraded = visible_cells > MAX_GRID_OVERLAY_CELLS
        if degraded and not self._grid_overlay_degraded_notice_shown and self._simulation_running_for_perf:
            # Only surfaced to the console while the simulation is actually
            # running -- during setup/load/reset this would just be console
            # noise about a state the user isn't looking at yet. Still
            # tracked in latest_perf_status's cache_status field either way.
            self.append_console_message(
                f"[PERF] grid overlay degraded due visible_cells={visible_cells}"
            )
            self._grid_overlay_degraded_notice_shown = True
        if not degraded:
            self._grid_overlay_degraded_notice_shown = False
        if degraded:
            cell_bounds = None  # skip per-cell coloring; grid lines only.

        self._grid_overlay_degraded = degraded
        self._grid_overlay_last_visible_cells = visible_cells

        cache_key = (
            round(float(resolution), 3),
            self.width(),
            self.height(),
            tuple(round(float(bound), 2) for bound in self.render_view_bounds_world()),
            snapshot_version if cell_bounds is not None else -1,
            getattr(self.config, "map_visualization", DEFAULT_MAP_VISUALIZATION),
            getattr(self.config, "custom_unexplored_color", DEFAULT_CUSTOM_UNEXPLORED_COLOR),
            getattr(self.config, "custom_explored_color", DEFAULT_CUSTOM_EXPLORED_COLOR),
            bool(self.grid_overlay_enabled),
            bool(self.grid_cell_values_enabled),
            bool(self.frontier_decisions_enabled),
        )

        if self._grid_overlay_cache is not None and self._grid_overlay_cache_key == cache_key:
            self._grid_overlay_rebuild_ms = 0.0
            _blit_start = time.perf_counter()
            painter.drawPixmap(0, 0, self._grid_overlay_cache)
            self._grid_overlay_blit_ms = (time.perf_counter() - _blit_start) * 1000.0
            self._grid_overlay_last_cache_status = "hit"
            return

        self._grid_overlay_cache_key = cache_key
        _rebuild_start = time.perf_counter()
        self._grid_overlay_cache = self._rebuild_grid_overlay_cache(
            rect, resolution, snapshot, cell_bounds
        )
        self._grid_overlay_rebuild_ms = (time.perf_counter() - _rebuild_start) * 1000.0
        self._grid_overlay_last_cache_status = "degraded" if degraded else "rebuild"
        _blit_start = time.perf_counter()
        painter.drawPixmap(0, 0, self._grid_overlay_cache)
        self._grid_overlay_blit_ms = (time.perf_counter() - _blit_start) * 1000.0

    def _rebuild_grid_overlay_cache(
        self,
        rect,
        resolution: float,
        snapshot: dict | None,
        cell_bounds: tuple[int, int, int, int] | None,
    ) -> QPixmap:
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)

        cache_painter = QPainter(cache)
        cache_painter.save()
        cache_painter.setClipRect(rect)

        _cells_start = time.perf_counter()
        if self.grid_overlay_enabled and snapshot is not None and cell_bounds is not None:
            self._draw_grid_overlay_cells(cache_painter, snapshot, cell_bounds)
        self._grid_overlay_cells_ms = (time.perf_counter() - _cells_start) * 1000.0

        _lines_start = time.perf_counter()
        if self.grid_overlay_enabled:
            self._draw_grid_overlay_lines(cache_painter, rect, resolution)
        if snapshot is not None and cell_bounds is not None:
            self._draw_grid_debug_labels(cache_painter, snapshot, cell_bounds)
        self._grid_overlay_lines_ms = (time.perf_counter() - _lines_start) * 1000.0

        cache_painter.restore()
        cache_painter.end()
        return cache

    def _draw_grid_overlay_lines(self, painter: QPainter, rect, resolution: float) -> None:
        left, right, bottom, top = self.render_view_bounds_world()

        line_color = (
            self.discovery_contrast_color(45)
            if self.is_shared_discovery_mode()
            else QColor(90, 90, 90, 70)
        )
        painter.setPen(QPen(line_color, 1))

        x = math.floor(left / resolution) * resolution
        while x <= right + resolution * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += resolution

        y = math.floor(bottom / resolution) * resolution
        while y <= top + resolution * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += resolution

    def _draw_grid_overlay_cells(
        self,
        painter: QPainter,
        snapshot: dict,
        cell_bounds: tuple[int, int, int, int],
    ) -> None:
        """Fill each visible cell with a translucent color based on its
        occupancy state (unknown/free/occupied). All colors are low-alpha
        so obstacles, routes, and the robot underneath/above remain
        readable -- this is a debug aid, not an opaque map layer. Only
        called during a cache rebuild, never every frame.
        """
        grid = snapshot.get("grid")
        resolution = float(snapshot.get("resolution") or 0.0)
        bounds = snapshot.get("bounds")
        if grid is None or resolution <= 0.0 or bounds is None:
            return

        x_min, _x_max, y_min, _y_max = bounds
        col_start, col_end, row_start, row_end = cell_bounds

        if self.is_shared_discovery_mode():
            unknown_brush = QBrush(QColor(0, 0, 0, 0))
            free = self.explored_area_color()
            free.setAlpha(36)
            free_brush = QBrush(free)
        else:
            unknown_brush = QBrush(QColor(120, 120, 120, 35))
            free_brush = QBrush(QColor(60, 140, 220, 45))

        # Occupied cells keep the existing obstacle-map color in every mode.
        occupied_brush = QBrush(QColor(220, 40, 40, 80))

        painter.setPen(Qt.NoPen)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                state = int(grid[row, col])
                if state == 1:
                    painter.setBrush(occupied_brush)
                elif state == 0:
                    painter.setBrush(free_brush)
                else:
                    painter.setBrush(unknown_brush)

                cx0 = x_min + col * resolution
                cy0 = y_min + row * resolution
                sxA, syA = self.world_to_screen(cx0, cy0)
                sxB, syB = self.world_to_screen(cx0 + resolution, cy0 + resolution)
                painter.drawRect(
                    QRectF(
                        min(sxA, sxB),
                        min(syA, syB),
                        abs(sxB - sxA),
                        abs(syB - syA),
                    )
                )

    def _draw_grid_debug_labels(
        self,
        painter: QPainter,
        snapshot: dict,
        cell_bounds: tuple[int, int, int, int],
    ) -> None:
        """Draw occupancy values and frontier semantics as separate layers.

        Occupancy owns only -1/0/1. A frontier is a FREE cell adjacent to
        UNKNOWN, so it is highlighted with an ``F`` without replacing the
        underlying occupancy value. Labels are hidden when cells are too
        small to read; frontier tint remains visible while zoomed out.
        """
        grid = snapshot.get("grid")
        bounds = snapshot.get("bounds")
        resolution = float(snapshot.get("resolution") or 0.0)
        if grid is None or bounds is None or resolution <= 0.0:
            return

        col_start, col_end, row_start, row_end = cell_bounds
        x_min, _x_max, y_min, _y_max = bounds
        frontier_cells = {
            (int(row), int(col))
            for row, col in snapshot.get("frontier_cells", ())
            if row_start <= int(row) <= row_end and col_start <= int(col) <= col_end
        }
        bfs_steps = snapshot.get("bfs_steps")

        sx0, sy0 = self.world_to_screen(0.0, 0.0)
        sx1, sy1 = self.world_to_screen(resolution, resolution)
        cell_px = min(abs(sx1 - sx0), abs(sy1 - sy0))
        painter.save()
        font = QFont(painter.font())
        # Keep semantic labels visible at every zoom level where per-cell
        # drawing itself is active.  Scaling down is preferable to abruptly
        # hiding -1/0/1 or F; the visible-cell degradation cap above still
        # protects performance for extremely dense, zoomed-out grids.
        font.setPixelSize(max(4, min(14, int(round(cell_px * 0.48)))))
        font.setBold(True)
        painter.setFont(font)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                cx0 = x_min + col * resolution
                cy0 = y_min + row * resolution
                sx_a, sy_a = self.world_to_screen(cx0, cy0)
                sx_b, sy_b = self.world_to_screen(cx0 + resolution, cy0 + resolution)
                cell_rect = QRectF(
                    min(sx_a, sx_b), min(sy_a, sy_b),
                    abs(sx_b - sx_a), abs(sy_b - sy_a),
                )

                is_frontier = (row, col) in frontier_cells
                show_frontier_cells = self.frontier_decisions_enabled and not self.frontier_reasoning_cluster_view_enabled
                if show_frontier_cells and is_frontier:
                    painter.fillRect(cell_rect.adjusted(1, 1, -1, -1), QColor(255, 190, 40, 92))
                    painter.setPen(QPen(QColor(180, 105, 0, 220), 2))
                    painter.drawRect(cell_rect.adjusted(1, 1, -1, -1))

                if show_frontier_cells and is_frontier:
                    painter.setPen(QColor(110, 55, 0, 245))
                    step = int(bfs_steps[row, col]) if bfs_steps is not None else -1
                    painter.drawText(cell_rect, Qt.AlignCenter, f"F:{step}" if step >= 0 else "F")
                elif self.frontier_decisions_enabled and bfs_steps is not None and int(bfs_steps[row, col]) >= 0:
                    painter.setPen(QColor(80, 65, 25, 220))
                    painter.drawText(cell_rect, Qt.AlignCenter, str(int(bfs_steps[row, col])))
                elif self.grid_cell_values_enabled:
                    state = int(grid[row, col])
                    color = QColor(180, 30, 30, 235) if state == 1 else QColor(25, 65, 95, 225)
                    painter.setPen(color)
                    painter.drawText(cell_rect, Qt.AlignCenter, str(state))

        painter.restore()

    def current_robot_pose(self) -> tuple[float, float, float, float]:
        if self.robot is not None:
            return (
                float(self.robot.x),
                float(self.robot.y),
                float(self.robot.theta),
                float(self.robot.vision),
            )

        if "Multiple" in self.config.agent_mode:
            robots = normalized_robot_start_configs(self.config)
            if robots:
                index = max(0, min(int(self.config.selected_robot_index), len(robots) - 1))
                robot_cfg = robots[index]
                return (
                    float(robot_cfg.x),
                    float(robot_cfg.y),
                    float(robot_cfg.theta),
                    float(self.config.vision),
                )

        return (
            float(self.config.x),
            float(self.config.y),
            float(self.config.theta),
            float(self.config.vision),
        )

    def current_goal_xy(self) -> tuple[float, float]:
        # The final mission goal should always remain visible, even when the
        # robot is internally tracking an intermediate waypoint.
        return float(self.config.goal_x), float(self.config.goal_y)

    def draw_explored_area_trace(self, painter: QPainter):
        """Draw live coverage, or the frozen explored mask while replaying."""
        historical_environment = self._decoded_navigation_debug_environment()
        if historical_environment is not None:
            self._draw_historical_explored_area(painter, historical_environment)
            return

        if not self.config.show_explored_area:
            return

        if (
            not self.explored_area_polygons
            and self._explored_area_seed_mask is None
            and not self._explored_area_caches_by_robot
            and self._explored_area_cache is None
            and not self._explored_area_paths_by_robot
        ):
            return

        # ensure_explored_area_cache() (re)builds from the authoritative
        # seed mask when stale (see rebuild_explored_area_cache()) -- for a
        # multi-robot mask this populates _explored_area_caches_by_robot as
        # a side effect, so that dict must be read AFTER this call, not
        # before it: checking it first would see a stale/just-invalidated
        # empty dict and fall through to the combined cache, missing the
        # per-robot coverage the rebuild below is about to produce.
        self.ensure_explored_area_cache()

        painter.save()
        if self._explored_area_caches_by_robot:
            for robot_index in sorted(self._explored_area_caches_by_robot):
                cache = self._explored_area_caches_by_robot.get(robot_index)
                if cache is not None:
                    painter.drawPixmap(0, 0, cache)
        elif self._explored_area_cache is not None:
            painter.drawPixmap(0, 0, self._explored_area_cache)
        painter.restore()

    def _draw_historical_explored_area(self, painter: QPainter, environment: dict) -> None:
        """Rasterize the frozen per-robot explored masks for one replay frame."""
        frame = environment["frame"]
        cache_key = (
            int(frame.revision),
            self.width(),
            self.height(),
            self._view_transform_signature(),
            self._theme_mode,
            getattr(self.config, "map_visualization", DEFAULT_MAP_VISUALIZATION),
            getattr(self.config, "custom_unexplored_color", DEFAULT_CUSTOM_UNEXPLORED_COLOR),
            getattr(self.config, "custom_explored_color", DEFAULT_CUSTOM_EXPLORED_COLOR),
            getattr(
                self.config,
                "custom_explored_opacity",
                DEFAULT_CUSTOM_EXPLORED_OPACITY,
            ),
        )
        if self._nav_debug_explored_cache is None or self._nav_debug_explored_cache_key != cache_key:
            cache = QPixmap(self.size())
            cache.fill(Qt.transparent)
            cache_painter = QPainter(cache)
            cache_painter.save()
            cache_painter.setClipRect(self.plot_rect())
            self._paint_explored_mask_to_cache(
                cache_painter,
                environment["explored_by_robot"],
                float(environment["resolution"]),
                environment["bounds"],
                alpha=self._explored_area_alpha(30),
            )
            cache_painter.restore()
            cache_painter.end()
            self._nav_debug_explored_cache = cache
            self._nav_debug_explored_cache_key = cache_key

        painter.save()
        painter.drawPixmap(0, 0, self._nav_debug_explored_cache)
        painter.restore()

    def sensor_polygon_for_pose(
        self,
        cache_key: int,
        x: float,
        y: float,
        theta: float,
        vision: float,
    ) -> list[tuple[float, float]]:
        """Return a cached occlusion-aware sensor polygon for one robot."""
        signature = (
            round(float(vision), 3),
            str(self.config.vision_model),
            round(float(getattr(self.config, "camera_fov_degrees", 70.0)), 3),
            self.obstacles_cache_signature(),
        )
        pose = (float(x), float(y), float(theta))
        cached = self._sensor_polygon_caches_by_robot.get(int(cache_key))
        if cached is not None:
            cached_pose, cached_signature, cached_polygon = cached
            moved = math.hypot(pose[0] - cached_pose[0], pose[1] - cached_pose[1])
            rotated = abs(wrapped_angle_error(pose[2], cached_pose[2]))
            if (
                cached_signature == signature
                and moved < SENSOR_DRAW_RECOMPUTE_DISTANCE
                and rotated < SENSOR_DRAW_RECOMPUTE_ROTATION
            ):
                return cached_polygon

        polygon = sensor_visible_polygon_world(
            origin=(pose[0], pose[1]),
            theta=pose[2],
            vision=float(vision),
            vision_model=self.config.vision_model,
            obstacles=self.config.obstacles,
            ray_count=SENSOR_DRAW_RAYS_CAMERA if "Camera" in self.config.vision_model else SENSOR_DRAW_RAYS_OMNI,
            camera_fov_degrees=float(
                getattr(self.config, "camera_fov_degrees", 70.0)
            ),
        )
        self._sensor_polygon_caches_by_robot[int(cache_key)] = (pose, signature, polygon)
        return polygon

    def draw_sensor_polygon(
        self,
        painter: QPainter,
        polygon: list[tuple[float, float]],
        color: QColor,
        alpha_fill: int = 16,
        alpha_stroke: int = 58,
    ) -> None:
        if len(polygon) < 3:
            return

        fill = QColor(color)
        stroke = QColor(color)
        if self.is_custom_discovery_mode():
            # Custom Discovery's opacity describes accumulated team coverage.
            # A second translucent FoV fill per robot would stack on top and
            # create the non-uniform wedges the shared style is meant to avoid.
            # Keep the live FoV boundary, but leave its interior transparent.
            fill.setAlpha(0)
            stroke.setAlpha(max(alpha_stroke, 205))
        elif self.is_monochrome_discovery_mode():
            fill.setAlpha(max(alpha_fill, 48))
            stroke.setAlpha(max(alpha_stroke, 205))
        else:
            fill.setAlpha(alpha_fill)
            stroke.setAlpha(alpha_stroke)

        visible_path = QPainterPath()
        sx, sy = self.world_to_screen(*polygon[0])
        visible_path.moveTo(sx, sy)
        for point in polygon[1:]:
            px, py = self.world_to_screen(*point)
            visible_path.lineTo(px, py)
        visible_path.closeSubpath()

        if self.is_shared_discovery_mode():
            # Contrast outline keeps the shared FoV boundary readable over
            # either the unexplored or explored custom color.
            under = QColor(0, 0, 0, 125) if color.lightness() >= 128 else QColor(255, 255, 255, 135)
            painter.setPen(QPen(under, 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(visible_path)

        painter.setPen(QPen(stroke, 1.4))
        painter.setBrush(QBrush(fill))
        painter.drawPath(visible_path)

    def sensor_display_poses(self) -> list[tuple[int, float, float, float, float]]:
        """Return all sensor poses that should be visible on the canvas."""
        if "Multiple" in self.config.agent_mode:
            if self.robots:
                return [
                    (index, float(robot.x), float(robot.y), float(robot.theta), float(robot.vision))
                    for index, robot in enumerate(self.robots)
                ]
            if self.robot is None:
                return [
                    (index, float(cfg.x), float(cfg.y), float(cfg.theta), float(self.config.vision))
                    for index, cfg in enumerate(normalized_robot_start_configs(self.config))
                ]

        x, y, theta, vision = self.current_robot_pose()
        return [(-1, x, y, theta, vision)]

    def body_radius_for_display_key(self, cache_key: int) -> float:
        if int(cache_key) >= 0:
            if self.robots and int(cache_key) < len(self.robots):
                return float(getattr(self.robots[int(cache_key)], "_sim_body_radius", self.config.body_radius))
            configs = normalized_robot_start_configs(self.config)
            if int(cache_key) < len(configs):
                return float(configs[int(cache_key)].body_radius)
        return float(self.config.body_radius)

    def safety_radius_for_display_key(self, cache_key: int) -> float:
        if int(cache_key) >= 0:
            if self.robots and int(cache_key) < len(self.robots):
                body = float(getattr(self.robots[int(cache_key)], "_sim_body_radius", self.config.body_radius))
                return max(float(getattr(self.robots[int(cache_key)], "_sim_safety_radius", self.config.safety_radius)), body)
            configs = normalized_robot_start_configs(self.config)
            if int(cache_key) < len(configs):
                return max(float(configs[int(cache_key)].safety_radius), float(configs[int(cache_key)].body_radius))
        return max(float(self.config.safety_radius), float(self.config.body_radius))

    def draw_sensor_range(self, painter: QPainter):
        """
        Draw the actually visible sensor regions.

        In multi-robot mode the LiDAR/FoV of every robot is always drawn with
        the robot's own color. This layer is world/sensing information, not a
        robot-order/debug layer, so it does not depend on Robot Orders.

        Instrumentation only, no behavior change: robot_fov_compute_ms/
        robot_fov_paint_ms separately time sensor_polygon_for_pose() (the
        cached raycast lookup/rebuild) vs. draw_sensor_polygon() (the
        screen-space transform + paint), and robot_fov_cache_hit compares
        the polygon OBJECT returned by sensor_polygon_for_pose() against
        whatever was already cached for that cache_key before the call --
        purely a read of existing cache state, never a new cache or an
        extra recompute. Does not touch SENSOR_DRAW_RECOMPUTE_DISTANCE/
        ROTATION or any other existing cache threshold.
        """
        self._fov_detail["robot_fov_cache_hit"] = True
        self._fov_detail["robot_fov_compute_ms"] = 0.0
        self._fov_detail["robot_fov_paint_ms"] = 0.0
        if not self.config.show_vision:
            return

        historical_snapshot = self._navigation_debug_history_snapshot()
        if historical_snapshot is not None:
            sensor_debug = historical_snapshot.sensor
            polygon = []
            point_count = int(getattr(sensor_debug, "visible_polygon_count", 0))
            polygon_bytes = getattr(sensor_debug, "visible_polygon_f32_zlib", b"")
            if point_count > 0 and polygon_bytes:
                points = np.frombuffer(
                    zlib.decompress(polygon_bytes), dtype=np.float32
                ).reshape((point_count, 2))
                polygon = [(float(point[0]), float(point[1])) for point in points]
            if not polygon:
                polygon = self.sensor_polygon_for_pose(
                    -999,
                    float(historical_snapshot.robot_pose.x),
                    float(historical_snapshot.robot_pose.y),
                    float(historical_snapshot.robot_pose.theta),
                    float(getattr(historical_snapshot.sensor, "vision_range", 0.0) or self.config.vision),
                )
            painter.save()
            self.draw_sensor_polygon(painter, polygon, self.sensor_display_color(None))
            painter.restore()
            self._fov_detail["robot_fov_cache_hit"] = True
            return

        painter.save()
        cache_hit_this_frame = True
        for cache_key, x, y, theta, vision in self.sensor_display_poses():
            color = self.sensor_display_color(cache_key)

            _compute_start = time.perf_counter()
            previously_cached = self._sensor_polygon_caches_by_robot.get(int(cache_key))
            prev_polygon = previously_cached[2] if previously_cached is not None else None
            polygon = self.sensor_polygon_for_pose(cache_key, x, y, theta, vision)
            self._fov_detail["robot_fov_compute_ms"] += (time.perf_counter() - _compute_start) * 1000.0
            if polygon is not prev_polygon:
                cache_hit_this_frame = False

            _paint_start = time.perf_counter()
            self.draw_sensor_polygon(painter, polygon, color)
            self._fov_detail["robot_fov_paint_ms"] += (time.perf_counter() - _paint_start) * 1000.0
        painter.restore()
        self._fov_detail["robot_fov_cache_hit"] = cache_hit_this_frame

    def obstacle_boundary_samples_for_display(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> list[tuple[float, float]]:
        """
        Sample an obstacle boundary for display-only discovery coverage.

        This mirrors the mapping abstraction without changing planner behavior:
        a rectangle is treated as fully discovered only when the robot has
        observed most of its boundary samples from visible viewpoints.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(self.config.mapping_point_spacing), 0.015)
        points: list[tuple[float, float]] = []

        nx = max(1, int(math.ceil(ow / spacing)))
        ny = max(1, int(math.ceil(oh / spacing)))

        for i in range(nx + 1):
            x = ox + ow * i / nx
            points.append((x, oy))
            points.append((x, oy + oh))

        for j in range(1, ny):
            y = oy + oh * j / ny
            points.append((ox, y))
            points.append((ox + ow, y))

        return points

    def obstacle_boundary_sample_count(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> int:
        """
        Return how many boundary samples would represent this obstacle.

        This is used as the denominator for completion opacity. It avoids
        building and comparing every sample against every mapped point during
        paintEvent.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(self.config.mapping_point_spacing), 0.015)
        nx = max(1, int(math.ceil(ow / spacing)))
        ny = max(1, int(math.ceil(oh / spacing)))
        return max(1, 2 * (nx + 1) + 2 * max(0, ny - 1))

    def mapped_point_lies_on_obstacle_boundary(
        self,
        point: tuple[float, float],
        obstacle: tuple[float, float, float, float],
    ) -> bool:
        """
        Fast boundary-membership test for visual completion opacity.

        The mapped points are generated from obstacle boundaries, so we do not
        need the previous O(boundary_samples * mapped_points) nearest-neighbor
        coverage check. Testing each mapped point against each rectangle edge is
        much cheaper and removes the FPS drop caused by Show Obstacles.
        """
        px, py = point
        ox, oy, ow, oh = obstacle
        tol = max(0.025, float(self.config.mapping_point_spacing) * 0.75)

        inside_x_span = (ox - tol) <= px <= (ox + ow + tol)
        inside_y_span = (oy - tol) <= py <= (oy + oh + tol)

        on_bottom_or_top = inside_x_span and (
            abs(py - oy) <= tol or abs(py - (oy + oh)) <= tol
        )
        on_left_or_right = inside_y_span and (
            abs(px - ox) <= tol or abs(px - (ox + ow)) <= tol
        )

        return bool(on_bottom_or_top or on_left_or_right)

    def obstacle_mapping_coverage(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> float:
        """
        Estimate obstacle-boundary coverage in O(mapped_points), not O(samples
        * mapped_points).

        This is intentionally visual only. Planning still uses the mapped point
        cloud; this value only controls opacity of the gray ground-truth layer.
        """
        if not self.mapped_obstacle_points:
            return 0.0

        sample_count = self.obstacle_boundary_sample_count(obstacle)
        covered = 0

        for point in self.mapped_obstacle_points:
            if self.mapped_point_lies_on_obstacle_boundary(point, obstacle):
                covered += 1

        return min(1.0, covered / sample_count)

    def ensure_obstacle_coverage_cache(self):
        if self._obstacle_coverage_cache_count == len(self.mapped_obstacle_points):
            return

        self._obstacle_coverage_cache = {}
        for index, obstacle in enumerate(self.config.obstacles):
            self._obstacle_coverage_cache[index] = self.obstacle_mapping_coverage(tuple(obstacle))
        self._obstacle_coverage_cache_count = len(self.mapped_obstacle_points)

    def obstacles_cache_signature(self) -> tuple:
        geometry = tuple(
            (
                round(float(ox), 4),
                round(float(oy), 4),
                round(float(ow), 4),
                round(float(oh), 4),
            )
            for ox, oy, ow, oh in self.config.obstacles
        )
        return (
            geometry,
            bool(self.is_custom_discovery_mode()),
            str(
                getattr(
                    self.config,
                    "custom_obstacle_color",
                    DEFAULT_CUSTOM_OBSTACLE_COLOR,
                )
            ).upper(),
        )

    def ensure_obstacles_cache(self):
        signature = self.obstacles_cache_signature()
        mapped_count = len(self.mapped_obstacle_points)

        base_cache_is_valid = (
            self._obstacles_cache is not None
            and self._obstacles_cache_size == self.size()
            and self._obstacles_cache_signature == signature
        )

        if base_cache_is_valid:
            mapped_delta = mapped_count - self._obstacles_cache_mapped_count
            if 0 <= mapped_delta < OBSTACLE_VISUAL_REFRESH_POINT_STEP:
                return

        self.rebuild_obstacles_cache(signature)

    def obstacle_is_squareish_stamp(self, obstacle: tuple[float, float, float, float]) -> bool:
        """Return whether an obstacle looks like one free-draw brush stamp.

        Free-draw strokes are stored as small square bounding boxes because the
        runtime planner/collision code still consumes rectangles. Rendering is
        allowed to interpret connected dense stamps as circles so the user sees
        one smooth object instead of a chain of tiny squares.
        """
        _, _, width, height = obstacle
        width = abs(float(width))
        height = abs(float(height))
        if width <= 0.0 or height <= 0.0:
            return False

        squareish = abs(width - height) <= max(0.025, 0.12 * max(width, height))
        # Do not depend on the current brush slider value. The user may draw a
        # stroke, change brush size, then run the simulation. A visual stamp
        # should still render as a stamp. Keep the cap high enough for normal
        # editor brush sizes, but low enough that large square obstacles remain
        # rectangles.
        plausible_stamp_size = max(width, height) <= 2.25
        return bool(squareish and plausible_stamp_size)

    def obstacle_group_looks_like_free_draw(self, indices: list[int]) -> bool:
        """Heuristic for deciding when a connected object is a free-draw stroke."""
        if len(indices) < 3:
            return False

        obstacles = [tuple(self.config.obstacles[index]) for index in indices if 0 <= index < len(self.config.obstacles)]
        if len(obstacles) < 3:
            return False

        squareish_count = sum(1 for obstacle in obstacles if self.obstacle_is_squareish_stamp(obstacle))
        if squareish_count / len(obstacles) < 0.70:
            return False

        sizes = [max(abs(float(obstacle[2])), abs(float(obstacle[3]))) for obstacle in obstacles]
        min_size = max(min(sizes), 1.0e-9)
        max_size = max(sizes)
        similar_sizes = (max_size / min_size) <= 2.25
        return bool(similar_sizes)

    def obstacle_screen_path(
        self,
        obstacle: tuple[float, float, float, float],
        *,
        as_brush_stamp: bool = False,
    ) -> QPainterPath:
        """Return the visual path for one obstacle in screen coordinates."""
        ox, oy, ow, oh = obstacle
        x1, y1 = self.world_to_screen(ox, oy)
        x2, y2 = self.world_to_screen(ox + ow, oy + oh)
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        path = QPainterPath()
        if as_brush_stamp:
            path.addEllipse(rect)
        else:
            path.addRect(rect)
        return path

    def obstacle_visual_groups(self) -> list[list[int]]:
        """Return connected obstacle groups for seam-free rendering.

        Data remains as individual rectangles. This method affects only the
        display layer, so joined objects and free-draw strokes look like one
        object in both editor mode and simulation mode.
        """
        groups: list[list[int]] = []
        visited: set[int] = set()

        for index in range(len(self.config.obstacles)):
            if index in visited:
                continue
            group = connected_obstacle_indices(list(self.config.obstacles), index)
            if not group:
                group = [index]
            for group_index in group:
                visited.add(group_index)
            groups.append(group)

        return groups

    def obstacle_group_screen_path(self, indices: list[int]) -> QPainterPath:
        """Build a unified visual path for one connected obstacle object."""
        union_path = QPainterPath()
        draw_as_free_stroke = self.obstacle_group_looks_like_free_draw(indices)

        for index in indices:
            if index < 0 or index >= len(self.config.obstacles):
                continue

            obstacle_path = self.obstacle_screen_path(
                tuple(self.config.obstacles[index]),
                as_brush_stamp=draw_as_free_stroke,
            )
            if union_path.isEmpty():
                union_path = obstacle_path
            else:
                union_path = union_path.united(obstacle_path)

        return union_path.simplified()

    def obstacle_group_mapping_coverage(self, indices: list[int]) -> float:
        """Return display-only completion coverage for a connected object."""
        valid_indices = [index for index in indices if 0 <= index < len(self.config.obstacles)]
        if not valid_indices:
            return 0.0
        return float(
            sum(self._obstacle_coverage_cache.get(index, 0.0) for index in valid_indices)
            / len(valid_indices)
        )

    def draw_obstacle_group(
        self,
        painter: QPainter,
        indices: list[int],
    ) -> None:
        """Draw one connected obstacle object without internal seams.

        Default/monochrome modes retain the theme-aware neutral obstacle
        palette. Custom Discovery deliberately uses its configured obstacle
        color; this is rendering-only and never changes collision geometry.
        """
        path = self.obstacle_group_screen_path(indices)
        if path.isEmpty():
            return

        dark = self._theme_mode == ThemeMode.DARK

        if self.editor_mode:
            fill = QColor(210, 213, 219, 130) if dark else QColor(178, 181, 188, 105)
            stroke = QColor(160, 165, 175, 200) if dark else QColor(82, 84, 92, 165)
            pen_width = 1.35
        else:
            coverage = self.obstacle_group_mapping_coverage(indices)
            fully_discovered = coverage >= OBSTACLE_COMPLETE_COVERAGE

            if self.is_custom_discovery_mode():
                base = self._valid_config_color(
                    getattr(
                        self.config,
                        "custom_obstacle_color",
                        DEFAULT_CUSTOM_OBSTACLE_COLOR,
                    ),
                    DEFAULT_CUSTOM_OBSTACLE_COLOR,
                )
                fill = QColor(base)
                stroke = QColor(base)
                if fully_discovered:
                    fill.setAlpha(180)
                    stroke.setAlpha(240)
                    pen_width = 1.7
                else:
                    fill.setAlpha(85)
                    stroke.setAlpha(135)
                    pen_width = 1.2
            elif fully_discovered:
                fill = QColor(224, 227, 233, 195) if dark else QColor(190, 194, 202, 170)
                stroke = QColor(176, 181, 191, 235) if dark else QColor(82, 84, 92, 210)
                pen_width = 1.7
            else:
                fill = QColor(210, 213, 219, 105) if dark else QColor(178, 181, 188, 85)
                stroke = QColor(160, 165, 175, 130) if dark else QColor(82, 84, 92, 105)
                pen_width = 1.2

        painter.setPen(QPen(stroke, pen_width))
        painter.setBrush(QBrush(fill))
        painter.drawPath(path)

    def rebuild_obstacles_cache(self, signature: tuple | None = None):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._obstacles_cache = cache
        self._obstacles_cache_size = QSize(self.size())
        self._obstacles_cache_mapped_count = len(self.mapped_obstacle_points)
        self._obstacles_cache_signature = signature if signature is not None else self.obstacles_cache_signature()

        if not self.config.obstacles:
            return

        self.ensure_obstacle_coverage_cache()

        cache_painter = QPainter(self._obstacles_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())

        for group in self.obstacle_visual_groups():
            self.draw_obstacle_group(cache_painter, group)

        cache_painter.end()

    def draw_ground_truth_obstacles(self, painter: QPainter):
        """
        Draw scenario obstacles from a cached pixmap.

        This keeps the human-facing gray obstacles visible without recomputing
        completion opacity or redrawing rectangles every frame.
        """
        if not self.config.obstacles:
            return

        self.ensure_obstacles_cache()
        if self._obstacles_cache is None:
            return

        painter.save()
        painter.drawPixmap(0, 0, self._obstacles_cache)
        painter.restore()

    def draw_editor_preview(self, painter: QPainter):
        if not self.editor_mode or self.editor_drag_start is None or self.editor_drag_current is None:
            return

        if self.editor_tool == "free":
            if len(self.editor_preview_points) >= 2:
                painter.save()
                stroke_width = max(1.6, self.editor_brush_size * self.pixels_per_meter() * 0.8)
                painter.setPen(QPen(QColor(BLUE), stroke_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setOpacity(0.75)
                path = QPainterPath()
                first = self.editor_preview_points[0]
                x0, y0 = self.world_to_screen(first[0], first[1])
                path.moveTo(x0, y0)
                for point in self.editor_preview_points[1:]:
                    x, y = self.world_to_screen(point[0], point[1])
                    path.lineTo(x, y)
                painter.drawPath(path)

                # Live circular brush cursor at the last stamp position.
                last = self.editor_preview_points[-1]
                cx, cy = self.world_to_screen(last[0], last[1])
                radius = max(2.0, self.editor_brush_size * self.pixels_per_meter() / 2.0)
                painter.setPen(QPen(QColor(BLUE_DARK), 1.4))
                painter.setBrush(QBrush(QColor(255, 255, 255, 95)))
                painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
                painter.restore()
            return

        if self.editor_tool not in {"rectangles", "squares"}:
            return

        start_x, start_y = self.editor_drag_start
        current_x, current_y = self.editor_drag_current
        left = min(start_x, current_x)
        bottom = min(start_y, current_y)
        width = abs(current_x - start_x)
        height = abs(current_y - start_y)

        if width < MIN_EDITOR_OBSTACLE_SIZE and height < MIN_EDITOR_OBSTACLE_SIZE:
            return

        if self.editor_tool == "squares":
            size = max(width, height)
            width = size
            height = size

        x1, y1 = self.world_to_screen(left, bottom)
        x2, y2 = self.world_to_screen(left + width, bottom + height)
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        painter.save()
        painter.setPen(QPen(QColor(BLUE), 2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(BLUE_LIGHT)))
        painter.setOpacity(0.35)
        painter.drawRect(rect)
        painter.restore()


    def draw_editor_move_selection(self, painter: QPainter):
        """Highlight the connected object currently being moved in editor mode."""
        if not self.editor_mode or not self.editor_obstacle_drag_indices:
            return

        selection_path = QPainterPath()
        for index in self.editor_obstacle_drag_indices:
            if index < 0 or index >= len(self.config.obstacles):
                continue
            path = self.obstacle_screen_path(tuple(self.config.obstacles[index]))
            selection_path = path if selection_path.isEmpty() else selection_path.united(path)

        if selection_path.isEmpty():
            return

        painter.save()
        painter.setPen(QPen(QColor(220, 52, 52), 2.0, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(selection_path.simplified())
        painter.restore()

    def draw_editor_camera_frame(self, painter: QPainter):
        """Draw the adjustable red simulation camera frame in editor mode."""
        if not self.editor_mode:
            return

        rect = self.camera_rect_screen()
        plot = QRectF(self.plot_rect())
        if rect.isNull() or rect.width() <= 0.0 or rect.height() <= 0.0:
            return

        painter.save()
        painter.setClipRect(plot)

        # Soft outside overlay so the user understands this frame is the future
        # simulation viewport, not an obstacle.
        outside = QPainterPath()
        outside.addRect(plot)
        inside = QPainterPath()
        inside.addRect(rect)
        outside = outside.subtracted(inside)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(220, 52, 52, 20)))
        painter.drawPath(outside)

        painter.setPen(QPen(QColor(220, 52, 52), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(255, 255, 255), 1.4))
        painter.setBrush(QBrush(QColor(220, 52, 52)))
        handle_size = 7.0
        for point in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
            painter.drawRect(QRectF(point.x() - handle_size / 2.0, point.y() - handle_size / 2.0, handle_size, handle_size))

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        label = "Simulation camera viewport"
        label_rect = QRectF(rect.left() + 8, rect.top() + 8, 172, 20)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(label_rect, 6, 6)
        painter.setPen(QColor(160, 20, 20))
        painter.drawText(label_rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, label)

        painter.restore()

    def draw_safety_radius(self, painter: QPainter):
        """
        Draw safety radius r below mapped obstacles and waypoints.

        In multi-robot mode every robot gets its own colored safety radius when
        Robot Orders is enabled.
        """
        px_per_meter = self.pixels_per_meter()

        painter.save()
        for cache_key, x, y, _, _ in self.sensor_display_poses():
            rx, ry = self.world_to_screen(x, y)
            radius = self.safety_radius_for_display_key(cache_key) * px_per_meter
            color = QColor(122, 30, 36) if cache_key < 0 else robot_color(cache_key)
            stroke = QColor(color)
            stroke.setAlpha(105)
            fill = QColor(color)
            fill.setAlpha(18)
            painter.setPen(QPen(stroke, 1.8, Qt.DashLine))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QRectF(rx - radius, ry - radius, radius * 2, radius * 2))

        painter.restore()

    def _navigation_debug_path_from_maybe_points(self, maybe_points) -> QPainterPath | None:
        """Convert an already-computed Maybe[tuple[Point2D, ...]] into a
        QPainterPath. The only place raw/simplified/predicted world-space
        point lists become Qt geometry -- nothing upstream in
        simulation/planning/navigation ever builds a QPainterPath."""
        if maybe_points.unavailable or not maybe_points.value or len(maybe_points.value) < 2:
            return None
        return self._navigation_debug_path_from_points(maybe_points.value)

    def _navigation_debug_path_from_points(self, points) -> QPainterPath | None:
        if not points or len(points) < 2:
            return None
        path = QPainterPath()
        sx, sy = self.world_to_screen(*points[0])
        path.moveTo(sx, sy)
        for point in points[1:]:
            sx, sy = self.world_to_screen(*point)
            path.lineTo(sx, sy)
        return path

    def _rebuild_navigation_debug_overlay_cache(self, snapshot) -> dict:
        # The accepted route and waypoints are rendered by draw_planned_route().
        # Navigation Debug only adds decision-specific information; it must not
        # draw raw/simplified/pending copies of the same route.
        return {
            "predicted_trajectory": self._navigation_debug_path_from_maybe_points(
                snapshot.predicted_motion.trajectory
            ),
        }

    def draw_navigation_debug_overlay(self, painter: QPainter):
        """Draw live reasoning without duplicating the authoritative route."""
        snapshot = self._nav_debug_snapshot
        if snapshot is None:
            return

        cache_key = (snapshot.snapshot_id, self._view_transform_signature())
        if self._nav_debug_overlay_cache is None or self._nav_debug_overlay_cache_key != cache_key:
            self._nav_debug_overlay_cache = self._rebuild_navigation_debug_overlay_cache(snapshot)
            self._nav_debug_overlay_cache_key = cache_key
        cache = self._nav_debug_overlay_cache

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # A predicted trajectory is only useful on-screen when it explains an
        # intervention. Safe predictions are omitted to avoid looking like a
        # second planned route.
        predicted_blocked = (
            not snapshot.predicted_motion.collision.unavailable
            and snapshot.predicted_motion.collision.value is not None
            and snapshot.predicted_motion.collision.value.blocked
        )
        if predicted_blocked and cache["predicted_trajectory"] is not None:
            painter.setPen(QPen(QColor(RED), 2.0, Qt.DotLine, Qt.RoundCap))
            painter.drawPath(cache["predicted_trajectory"])

        # Footprint/safety circles remain local to the current robot pose.
        px_per_meter = self.pixels_per_meter()
        rx, ry = self.world_to_screen(snapshot.robot_pose.x, snapshot.robot_pose.y)
        body_r = snapshot.safety.robot_radius * px_per_meter
        safety_r = snapshot.safety.safety_radius * px_per_meter
        painter.setPen(QPen(QColor(40, 40, 40, 150), 1.2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(rx - body_r, ry - body_r, body_r * 2, body_r * 2))
        painter.setPen(QPen(QColor(230, 140, 20, 155), 1.2, Qt.DotLine))
        painter.drawEllipse(QRectF(rx - safety_r, ry - safety_r, safety_r * 2, safety_r * 2))

        blocking_terms = None
        for maybe_terms in (
            snapshot.route.first_segment,
            snapshot.safety.active_segment,
            snapshot.predicted_motion.collision,
        ):
            if not maybe_terms.unavailable and maybe_terms.value is not None and maybe_terms.value.blocked:
                blocking_terms = maybe_terms.value
                break

        if blocking_terms is not None and blocking_terms.blocking_point is not None:
            bx, by = self.world_to_screen(*blocking_terms.blocking_point)
            marker_r = 6.0
            painter.setPen(QPen(QColor(RED), 2.2))
            painter.setBrush(QBrush(QColor(220, 30, 30, 80)))
            painter.drawEllipse(QRectF(bx - marker_r, by - marker_r, marker_r * 2, marker_r * 2))
            painter.drawLine(QPointF(bx - marker_r, by - marker_r), QPointF(bx + marker_r, by + marker_r))
            painter.drawLine(QPointF(bx - marker_r, by + marker_r), QPointF(bx + marker_r, by - marker_r))

        self.draw_navigation_debug_heading_rays(painter)
        self.draw_navigation_debug_robot_label(painter)
        painter.restore()

    def draw_mapped_obstacle_points(self, painter: QPainter):
        """
        Draw discovered obstacle samples from a cached pixmap.

        This avoids redrawing thousands of tiny ellipses every frame. The cache
        is updated only when new mapped points are added or when the canvas is
        resized.
        """
        if not self.mapped_obstacle_points:
            return

        self.ensure_mapped_points_cache()
        if self._mapped_points_cache is None:
            return

        painter.save()
        painter.drawPixmap(0, 0, self._mapped_points_cache)
        painter.restore()

    def draw_ipp_uncertainty_heatmap(self, painter: QPainter) -> None:
        """Draw posterior GP variance from a validated paper bundle."""
        bundle = self._ipp_experiment_bundle
        if bundle is None:
            return
        variance = np.asarray(bundle.posterior_variance, dtype=np.float64)
        mask = np.asarray(bundle.mask, dtype=np.bool_)
        key = (
            str(getattr(bundle, "manifest_path", "")),
            variance.shape,
            float(np.min(variance[mask])),
            float(np.max(variance[mask])),
            str(getattr(bundle, "raster_origin", "lower")),
        )
        if self._ipp_variance_pixmap_cache is None or self._ipp_variance_pixmap_cache_key != key:
            rgba = ipp_uncertainty_rgba(variance, mask)
            # Dataset row zero follows raster_origin; QImage row zero is top.
            if str(getattr(bundle, "raster_origin", "lower")) == "lower":
                rgba = np.flipud(rgba)
            rgba = np.ascontiguousarray(rgba)
            height, width = rgba.shape[:2]
            image = QImage(
                rgba.data,
                width,
                height,
                int(rgba.strides[0]),
                QImage.Format_RGBA8888,
            ).copy()
            self._ipp_variance_pixmap_cache = QPixmap.fromImage(image)
            self._ipp_variance_pixmap_cache_key = key

        if self._ipp_variance_pixmap_cache is None:
            return
        x_min, x_max, y_min, y_max = map(float, bundle.data_world_bounds)
        left, top = self.world_to_screen(x_min, y_max)
        right, bottom = self.world_to_screen(x_max, y_min)
        target = QRectF(
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        )
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._ipp_variance_pixmap_cache, QRectF(self._ipp_variance_pixmap_cache.rect()))
        painter.restore()

    def draw_ipp_reference_overlay(self, painter: QPainter) -> None:
        """Draw the paper's pilot path, planned tour, sensing sites and FoVs."""
        bundle = self._ipp_experiment_bundle
        if bundle is None:
            return

        def draw_polyline(points, pen: QPen) -> None:
            points = np.asarray(points, dtype=float)
            if len(points) < 2:
                return
            path = QPainterPath()
            sx, sy = self.world_to_screen(float(points[0, 0]), float(points[0, 1]))
            path.moveTo(sx, sy)
            for point in points[1:]:
                sx, sy = self.world_to_screen(float(point[0]), float(point[1]))
                path.lineTo(sx, sy)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        painter.save()

        # FoV polygons are intentionally translucent violet.  Unlike the old
        # rendering bug, each patch is a bounded sensing footprint at a chosen
        # site, not a filled polygon connecting a robot to its frontier.
        painter.setPen(QPen(QColor(126, 87, 194, 145), 1.0))
        painter.setBrush(QBrush(QColor(126, 87, 194, 28)))
        for polygon in np.asarray(bundle.fovs, dtype=float):
            if len(polygon) < 3:
                continue
            path = QPainterPath()
            sx, sy = self.world_to_screen(float(polygon[0, 0]), float(polygon[0, 1]))
            path.moveTo(sx, sy)
            for point in polygon[1:]:
                sx, sy = self.world_to_screen(float(point[0]), float(point[1]))
                path.lineTo(sx, sy)
            path.closeSubpath()
            painter.drawPath(path)

        pilot_pen = QPen(QColor(0, 137, 123, 220), 2.0)
        pilot_pen.setStyle(Qt.DashLine)
        draw_polyline(bundle.pilot_path, pilot_pen)
        draw_polyline(bundle.solution_path, QPen(QColor(211, 47, 47, 235), 2.6))

        painter.setPen(QPen(QColor(117, 0, 30, 230), 1.0))
        painter.setBrush(QBrush(QColor(255, 235, 59, 245)))
        for point in np.asarray(bundle.sensing_points, dtype=float):
            sx, sy = self.world_to_screen(float(point[0]), float(point[1]))
            painter.drawEllipse(QPointF(sx, sy), 3.2, 3.2)

        metrics = getattr(bundle, "metrics", {})
        method = str(metrics.get("method", "uncertainty-guaranteed IPP"))
        maximum = float(np.max(np.asarray(bundle.posterior_variance)[np.asarray(bundle.mask)]))
        label = f"RSS 2026 · {method} · max posterior variance {maximum:.3g}"
        rect = self.plot_rect()
        label_rect = QRectF(rect.left() + 10, rect.top() + 9, max(240.0, rect.width() - 20), 25)
        painter.setPen(QPen(QColor(255, 255, 255, 225), 5.0))
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.setPen(QPen(QColor(25, 35, 45, 245), 1.0))
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.restore()

    def _build_hazard_pixmap(self, snapshot: dict) -> QPixmap | None:
        grid = np.asarray(snapshot.get("grid"), dtype=np.float32)
        if grid.ndim != 2 or grid.size == 0:
            return None

        heat = np.clip(grid, 0.0, 1.0)
        height, width = heat.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        low = heat <= 0.5
        high = ~low
        low_t = np.clip(heat * 2.0, 0.0, 1.0)
        high_t = np.clip((heat - 0.5) * 2.0, 0.0, 1.0)

        # Low hazard: pale yellow -> orange. High hazard: orange -> red.
        rgba[..., 0] = 255
        rgba[..., 1] = np.where(
            low,
            225.0 - 95.0 * low_t,
            130.0 - 95.0 * high_t,
        ).astype(np.uint8)
        rgba[..., 2] = np.where(
            low,
            70.0 - 45.0 * low_t,
            25.0 - 10.0 * high_t,
        ).astype(np.uint8)
        rgba[..., 3] = np.where(
            heat > 0.0,
            35.0 + 175.0 * np.power(heat, 0.72),
            0.0,
        ).astype(np.uint8)

        # Grid row 0 is the world's lower edge; QImage row 0 is the top.
        rgba = np.ascontiguousarray(np.flipud(rgba))
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    def draw_fires(self, painter: QPainter):
        """Draw the continuous thermal hazard field as a cached heatmap."""
        snapshot = self._hazard_snapshot
        if not snapshot:
            return
        grid = snapshot.get("grid")
        if grid is None or not np.any(grid):
            return

        cache_key = (
            int(snapshot.get("version", 0)),
            tuple(snapshot.get("bounds", ())),
            float(snapshot.get("resolution", 0.0)),
        )
        if self._hazard_pixmap_cache is None or self._hazard_pixmap_cache_key != cache_key:
            self._hazard_pixmap_cache = self._build_hazard_pixmap(snapshot)
            self._hazard_pixmap_cache_key = cache_key
        if self._hazard_pixmap_cache is None:
            return

        x_min, x_max, y_min, y_max = map(float, snapshot["bounds"])
        left, top = self.world_to_screen(x_min, y_max)
        right, bottom = self.world_to_screen(x_max, y_min)
        target = QRectF(
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        )

        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._hazard_pixmap_cache, QRectF(self._hazard_pixmap_cache.rect()))
        painter.restore()

    def _build_ground_truth_hazard_pixmap(self, grid) -> QPixmap | None:
        """Full ground-truth HazardField as a semi-transparent BLUE
        inspection heatmap -- deliberately a cold palette (theme.
        hazard_map_low/mid/high), never the warm discovered-hazard palette,
        so the two layers never look like the same information source even
        when both are visible at once (see draw_ground_truth_hazard_map()
        drawing this BELOW draw_discovered_hazard()).

        Reads theme_colors(self._theme_mode) -- unlike draw_fires()'s own
        _build_hazard_pixmap() above (kept theme-independent, see test_
        theme_palette.py), this palette is intentionally theme-dependent;
        callers must key their cache on ThemeMode too (see draw_ground_
        truth_hazard_map()'s own cache_key).
        """
        grid = np.asarray(grid, dtype=np.float32)
        if grid.ndim != 2 or grid.size == 0:
            return None

        heat = np.clip(grid, 0.0, 1.0)
        height, width = heat.shape

        colors = theme_colors(self._theme_mode)
        low_rgb = QColor(colors.hazard_map_low).getRgb()[:3]
        mid_rgb = QColor(colors.hazard_map_mid).getRgb()[:3]
        high_rgb = QColor(colors.hazard_map_high).getRgb()[:3]

        is_low = heat <= 0.5
        t_low = np.clip(heat * 2.0, 0.0, 1.0)
        t_high = np.clip((heat - 0.5) * 2.0, 0.0, 1.0)

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for channel in range(3):
            low_v, mid_v, high_v = low_rgb[channel], mid_rgb[channel], high_rgb[channel]
            rgba[..., channel] = np.where(
                is_low,
                low_v + (mid_v - low_v) * t_low,
                mid_v + (high_v - mid_v) * t_high,
            ).astype(np.uint8)
        # Semi-transparent throughout (capped well below the warm
        # discovered layer's own alpha) -- that layer draws on top of this
        # one and must always read clearly above it.
        rgba[..., 3] = np.where(
            heat > 0.0,
            20.0 + 100.0 * np.power(heat, 0.6),
            0.0,
        ).astype(np.uint8)

        rgba = np.ascontiguousarray(np.flipud(rgba))
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    def draw_ground_truth_hazard_map(self, painter: QPainter):
        """Full ground-truth HazardField as a semi-transparent BLUE
        inspection heatmap -- gated on show_hazard_map (default False; a
        debug/inspection overlay, never required to see what the team has
        actually discovered). Drawn BELOW draw_discovered_hazard() in the
        paint pipeline (see paintEvent()) so the warm discovered layer
        always reads on top.

        LIVE only: uses _hazard_snapshot["grid"] (RuntimeHazardService.
        snapshot()'s own continuous field). HISTORY has nothing safe to
        render here -- HazardDebug (the historical ground-truth contract,
        see navigation_snapshot.py) stores only discrete FireSource points,
        never a full continuous grid -- so this hides entirely while
        browsing history rather than inventing a grid or reusing the live
        one; draw_discovered_hazard()'s own historical replay is
        unaffected and keeps working normally.
        """
        if not self.show_hazard_map:
            return
        if self._navigation_debug_history_snapshot() is not None:
            return

        snapshot = self._hazard_snapshot
        if not snapshot:
            return
        grid = snapshot.get("grid")
        if grid is None or not np.any(grid):
            return

        cache_key = (
            int(snapshot.get("version", 0)),
            tuple(snapshot.get("bounds", ())),
            float(snapshot.get("resolution", 0.0)),
            str(self._theme_mode),
        )
        if (
            self._ground_truth_hazard_pixmap_cache is None
            or self._ground_truth_hazard_pixmap_cache_key != cache_key
        ):
            self._ground_truth_hazard_pixmap_cache = self._build_ground_truth_hazard_pixmap(grid)
            self._ground_truth_hazard_pixmap_cache_key = cache_key
        pixmap = self._ground_truth_hazard_pixmap_cache
        if pixmap is None:
            return

        x_min, x_max, y_min, y_max = map(float, snapshot["bounds"])
        left, top = self.world_to_screen(x_min, y_max)
        right, bottom = self.world_to_screen(x_max, y_min)
        target = QRectF(
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        )
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        painter.restore()

    def _build_discovered_hazard_pixmap(self, frame) -> QPixmap | None:
        """Warm amber -> orange -> coral-red -> light-core gradient (theme.
        discovered_hazard_low/mid/high/core) -- deliberately theme-
        dependent now (see test_theme_palette.py's semantic_methods list,
        which no longer includes draw_discovered_hazard()) -- but gated on
        observed=True per cell. An unobserved cell gets alpha=0 regardless
        of its (always 0.0, by HazardBelief's own contract) value: this is
        the actual omniscience guard, not an incidental byproduct of
        unobserved cells already being zero.

        Three color segments -- [0, 0.5] low->mid, (0.5, 0.85] mid->high,
        (0.85, 1] high->core -- so only the hottest handful of cells ever
        reach the bright core color, and a steeper (>1) alpha exponent
        keeps low/peripheral values much more transparent than the old
        formula did: a small, well-defined hot spot instead of one uniform
        blurry blob.
        """
        values = np.asarray(frame.values, dtype=np.float32)
        observed = np.asarray(frame.observed, dtype=bool)
        if values.ndim != 2 or values.size == 0:
            return None

        heat = np.clip(values, 0.0, 1.0)
        height, width = heat.shape

        colors = theme_colors(self._theme_mode)
        low_rgb = QColor(colors.discovered_hazard_low).getRgb()[:3]
        mid_rgb = QColor(colors.discovered_hazard_mid).getRgb()[:3]
        high_rgb = QColor(colors.discovered_hazard_high).getRgb()[:3]
        core_rgb = QColor(colors.discovered_hazard_core).getRgb()[:3]

        seg1 = heat <= 0.5
        seg2 = (heat > 0.5) & (heat <= 0.85)
        seg3 = heat > 0.85
        t1 = np.clip(heat / 0.5, 0.0, 1.0)
        t2 = np.clip((heat - 0.5) / 0.35, 0.0, 1.0)
        t3 = np.clip((heat - 0.85) / 0.15, 0.0, 1.0)

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for channel in range(3):
            low_v, mid_v, high_v, core_v = (
                low_rgb[channel], mid_rgb[channel], high_rgb[channel], core_rgb[channel]
            )
            stage1 = low_v + (mid_v - low_v) * t1
            stage2 = mid_v + (high_v - mid_v) * t2
            stage3 = high_v + (core_v - high_v) * t3
            rgba[..., channel] = np.select(
                [seg1, seg2, seg3], [stage1, stage2, stage3], default=stage1
            ).astype(np.uint8)

        rgba[..., 3] = np.where(
            observed & (heat > 0.0),
            18.0 + 190.0 * np.power(heat, 1.15),
            0.0,
        ).astype(np.uint8)

        # Grid row 0 is the world's lower edge; QImage row 0 is the top.
        rgba = np.ascontiguousarray(np.flipud(rgba))
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    def draw_discovered_hazard(self, painter: QPainter):
        """Draw the TEAM's DISCOVERED hazard belief -- observed=True cells
        only. Never the omniscient ground-truth HazardField (see draw_
        ground_truth_hazard_map() for that, drawn as a separate BELOW
        layer, gated on show_hazard_map): no FireSource, no real fire
        radius/center, ever read for this.

        ALWAYS drawn (no toggle gate here at all -- see paintEvent()):
        what the team has actually discovered must never be hideable, only
        ADDED to via show_hazard_map/show_fire_markers.

        Browsing HISTORY renders the SELECTED snapshot's own historical
        HazardBelief frame (via _decoded_navigation_debug_hazard_belief(),
        decoded/cached separately from the live pixmap so toggling between
        LIVE and HISTORY never thrashes either cache) -- never the live
        frame, never ground truth. A historical snapshot with no
        HazardBeliefDebug at all (captured before this existed) hides the
        layer entirely rather than falling back to anything else.

        Both pixmap caches are keyed on (revision, bounds, resolution,
        ThemeMode) -- never viewport/zoom (the pixmap is built in grid-
        aligned pixel space; world_to_screen() below repositions it per
        paint call cheaply, exactly like draw_fires()). ThemeMode IS part
        of the key now (see _build_discovered_hazard_pixmap()'s own
        docstring for why): a pixmap built with the LIGHT palette must
        never be reused after switching to DARK. Both reuse the same
        _build_discovered_hazard_pixmap() color formula -- no second copy
        of it here.
        """
        if self._navigation_debug_history_snapshot() is not None:
            historical = self._decoded_navigation_debug_hazard_belief()
            if historical is None:
                return
            if not np.any(historical["observed"]):
                return

            cache_key = (
                int(historical["revision"]),
                tuple(historical["bounds"]),
                float(historical["resolution"]),
                str(self._theme_mode),
            )
            if (
                self._discovered_hazard_history_pixmap_cache is None
                or self._discovered_hazard_history_pixmap_cache_key != cache_key
            ):
                frame_like = SimpleNamespace(values=historical["values"], observed=historical["observed"])
                self._discovered_hazard_history_pixmap_cache = self._build_discovered_hazard_pixmap(frame_like)
                self._discovered_hazard_history_pixmap_cache_key = cache_key
            pixmap = self._discovered_hazard_history_pixmap_cache
            if pixmap is None:
                return

            x_min, x_max, y_min, y_max = map(float, historical["bounds"])
            left, top = self.world_to_screen(x_min, y_max)
            right, bottom = self.world_to_screen(x_max, y_min)
            target = QRectF(
                min(left, right),
                min(top, bottom),
                abs(right - left),
                abs(bottom - top),
            )
            painter.save()
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
            painter.restore()
            return

        payload = self._discovered_hazard_frame
        if not payload:
            return
        frame = payload.get("frame")
        if frame is None or not np.any(frame.observed):
            return

        cache_key = (
            int(getattr(frame, "revision", 0)),
            tuple(payload.get("bounds", ())),
            float(payload.get("resolution", 0.0)),
            str(self._theme_mode),
        )
        if (
            self._discovered_hazard_pixmap_cache is None
            or self._discovered_hazard_pixmap_cache_key != cache_key
        ):
            self._discovered_hazard_pixmap_cache = self._build_discovered_hazard_pixmap(frame)
            self._discovered_hazard_pixmap_cache_key = cache_key
        if self._discovered_hazard_pixmap_cache is None:
            return

        x_min, x_max, y_min, y_max = map(float, payload["bounds"])
        left, top = self.world_to_screen(x_min, y_max)
        right, bottom = self.world_to_screen(x_max, y_min)
        target = QRectF(
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        )

        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(
            target, self._discovered_hazard_pixmap_cache, QRectF(self._discovered_hazard_pixmap_cache.rect())
        )
        painter.restore()

    def _current_fire_marker_context(self):
        """Resolve (sources, observed, bounds, resolution) for fire
        markers from EXACTLY one source of truth, matching draw_
        discovered_hazard()'s own LIVE/HISTORY split -- never mixing the
        two, never falling back to ground truth alone when no belief
        exists for the current view.

        Returns None when there is nothing safe to draw from: HISTORY with
        no HazardBeliefDebug, HISTORY with no HazardDebug (ground-truth
        sources) on that same snapshot, or LIVE with either the discovered
        frame or the ground-truth snapshot not yet pushed.
        """
        if self._navigation_debug_history_snapshot() is not None:
            historical = self._decoded_navigation_debug_hazard_belief()
            if historical is None:
                return None
            snapshot = self._navigation_debug_history_snapshot()
            maybe_hazard = getattr(snapshot, "hazard", None)
            if maybe_hazard is None or maybe_hazard.unavailable or maybe_hazard.value is None:
                return None
            return (
                maybe_hazard.value.sources,
                historical["observed"],
                historical["bounds"],
                historical["resolution"],
            )

        payload = self._discovered_hazard_frame
        if not payload:
            return None
        frame = payload.get("frame")
        if frame is None:
            return None
        hazard_snapshot = self._hazard_snapshot
        if not hazard_snapshot:
            return None
        return (
            hazard_snapshot.get("sources", ()),
            frame.observed,
            payload.get("bounds", ()),
            payload.get("resolution", 0.0),
        )

    def draw_fire_markers(self, painter: QPainter):
        """Draw one vectorial beacon per fire source -- DISCOVERED sources
        (center cell observed=True) are ALWAYS drawn, with the warm
        "discovered" style; show_fire_markers additionally draws every
        remaining UNDISCOVERED source with the tenue-blue "undiscovered"
        style (default OFF -- see set_fire_markers_enabled()). Never the
        reverse: this never hides a discovered source, only adds ground-
        truth ones on top of it.

            show_fire_markers=False -> render_sources = discovered_sources
            show_fire_markers=True  -> render_sources = all sources
                (each one still individually styled "discovered" or
                "undiscovered" -- a discovered source is drawn exactly
                once, never twice/duplicated with both styles.)

        Completely independent of draw_discovered_hazard()/draw_ground_
        truth_hazard_map() (the heatmaps): no shared cache or gate. Builds
        no pixmap/cache of its own -- the number of sources is small, so
        every rendered beacon is drawn directly each paint (see _draw_
        fire_beacon()).

        LIVE and HISTORY each read their own single source of truth via
        _current_fire_marker_context() (never mixed); _visible_fire_
        sources() determines which sources are "discovered" -- only by
        their center cell's observed=True, never by radius (see its own
        docstring for the exact anti-omniscience rule) -- regardless of
        show_fire_markers.
        """
        context = self._current_fire_marker_context()
        if context is None:
            return
        sources, observed, bounds, resolution = context
        if observed is None or getattr(observed, "size", 0) == 0:
            return

        discovered_sources = _visible_fire_sources(sources, observed, bounds=bounds, resolution=resolution)
        discovered_ids = {id(source) for source in discovered_sources}
        render_sources = sources if self.show_fire_markers else discovered_sources
        if not render_sources:
            return

        colors = theme_colors(self._theme_mode)
        for source in render_sources:
            sx, sy = self.world_to_screen(float(source.position[0]), float(source.position[1]))
            _draw_fire_beacon(painter, sx, sy, colors, discovered=(id(source) in discovered_ids))

    @staticmethod
    def _active_route_index(robot, route: list[tuple[float, float]]) -> int:
        """Index in a canvas route (which includes a start point) of the
        waypoint the physics robot is actually tracking now."""
        if robot is None or len(route) < 2:
            return -1

        manager = getattr(robot, "waypoints", None)
        active = robot.active_waypoint() if hasattr(robot, "active_waypoint") else None
        if active is not None:
            ax, ay = float(active[0]), float(active[1])
            candidates = [
                (math.hypot(float(point[0]) - ax, float(point[1]) - ay), index)
                for index, point in enumerate(route[1:], start=1)
            ]
            if candidates:
                distance, index = min(candidates)
                if distance <= 1e-4:
                    return index

        current_index = getattr(manager, "current_index", None)
        if isinstance(current_index, int):
            canvas_index = current_index + 1
            if 1 <= canvas_index < len(route):
                return canvas_index
        return -1

    def active_planned_waypoint_index(self) -> int:
        return self._active_route_index(self.robot, self.planned_path_points)

    def _remaining_single_planned_route(self) -> tuple[list[tuple[float, float]], int]:
        active_index = self.active_planned_waypoint_index()
        if self.robot is None or active_index < 1:
            return [], -1
        remaining = [
            (float(self.robot.x), float(self.robot.y)),
            *[tuple(map(float, point)) for point in self.planned_path_points[active_index:]],
        ]
        return remaining, active_index

    def _rebuild_route_path(self, points: list[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        sx, sy = self.world_to_screen(*points[0])
        path.moveTo(sx, sy)
        for point in points[1:]:
            sx, sy = self.world_to_screen(*point)
            path.lineTo(sx, sy)
        return path

    def _multi_robot_body_radius_px(self, robot_index: int) -> float:
        """Return the same screen-space body radius used by robot drawing."""
        body_radius = float(self.config.body_radius)
        if 0 <= int(robot_index) < len(self.robots):
            body_radius = float(
                getattr(self.robots[int(robot_index)], "_sim_body_radius", body_radius)
            )
        return max(5.0, body_radius * self.pixels_per_meter())

    def _multi_frontier_marker_radius_px(self, robot_index: int) -> float:
        """A compact assignment marker that never outweighs its robot.

        Frontiers are points of interest, not physical world footprints.  Keep
        them legible in screen space, but derive their size from the same
        body-radius scale as the owning robot and cap them at the established
        endpoint-marker size.  This is rendering-only; planner coordinates and
        tolerances are intentionally untouched.
        """
        body_px = self._multi_robot_body_radius_px(robot_index)
        return min(float(FRONTIER_OR_ENDPOINT_MARKER_RADIUS), max(5.0, body_px * 0.8))

    def _draw_waypoint_marker(
        self,
        painter: QPainter,
        point: tuple[float, float],
        *,
        active: bool,
        endpoint: bool,
        endpoint_is_goal: bool,
        color: QColor,
        endpoint_radius: float | None = None,
        endpoint_fill: QColor | None = None,
        endpoint_label: str | None = None,
    ) -> None:
        sx, sy = self.world_to_screen(*point)
        if endpoint:
            radius = (
                float(FRONTIER_OR_ENDPOINT_MARKER_RADIUS)
                if endpoint_radius is None
                else max(1.0, float(endpoint_radius))
            )
            fill = (
                QColor(GREEN)
                if endpoint_is_goal
                else QColor(endpoint_fill) if endpoint_fill is not None else QColor(146, 62, 160)
            )
            label = "G" if endpoint_is_goal else (endpoint_label or "F")
            painter.setPen(QPen(QColor("white"), 2.0))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius))
            painter.setFont(QFont("Segoe UI", 7 if len(label) > 1 else 8, QFont.Bold))
            painter.setPen(QPen(QColor("white")))
            painter.drawText(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius), Qt.AlignCenter, label)
            return

        radius = ACTIVE_WAYPOINT_MARKER_RADIUS if active else WAYPOINT_MARKER_RADIUS
        if active:
            halo = radius + ACTIVE_WAYPOINT_HALO_PADDING
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 45)))
            painter.drawEllipse(QRectF(sx - halo, sy - halo, 2 * halo, 2 * halo))
            fill = QColor(color)
            stroke = QColor("white")
        else:
            fill = QColor(255, 255, 255, 235)
            stroke = QColor(color)
        painter.setPen(QPen(stroke, 2.0))
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius))

    def draw_historical_planned_route(self, painter: QPainter):
        """Draw the route frozen inside the selected history snapshot.

        Never combine a historical robot pose with the current live route. The
        occupancy/heatmap background remains the current map, but motion,
        active waypoint and route all come from the same immutable snapshot.
        """
        if not self.config.show_path:
            return
        snapshot = self._nav_debug_snapshot
        if snapshot is None:
            return

        path_points = list(snapshot.path.active_path)
        active_index = snapshot.path.active_waypoint_index
        if active_index is None:
            active_index = 0
        active_index = max(0, min(int(active_index), len(path_points)))
        future_points = [
            (float(point[0]), float(point[1]))
            for point in path_points[active_index:]
        ]
        if not future_points and snapshot.path.active_segment is not None:
            future_points = [tuple(map(float, snapshot.path.active_segment[1]))]
        if not future_points:
            return

        start = (float(snapshot.robot_pose.x), float(snapshot.robot_pose.y))
        route_points = [start, *future_points]

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        route_color = QColor(ORANGE)
        route_color.setAlpha(215)
        painter.setPen(QPen(route_color, 2.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(self._rebuild_route_path(route_points))
        for offset, point in enumerate(future_points):
            self._draw_waypoint_marker(
                painter,
                point,
                active=(offset == 0),
                endpoint=(offset == len(future_points) - 1),
                endpoint_is_goal=False,
                color=QColor(ORANGE),
            )
        painter.restore()

    def draw_planned_route(self, painter: QPainter):
        """Draw exactly one authoritative route: the remaining accepted path.

        Past segments and the original start marker are omitted. The line begins
        at the robot's current pose and ends at the active/future waypoints, so
        a reached waypoint cannot remain visually connected after the robot has
        advanced to the next one.
        """
        if not self.config.show_path:
            return
        remaining, active_index = self._remaining_single_planned_route()
        if len(remaining) < 2:
            return

        future_points = remaining[1:]
        _build_start = time.perf_counter()
        view_signature = self._view_transform_signature()
        path_signature = (tuple(future_points), view_signature)
        if len(future_points) >= 2 and (
            self._planned_route_cache is None
            or self._planned_route_cache_signature != path_signature
        ):
            self._planned_route_cache = self._rebuild_route_path(future_points)
            self._planned_route_cache_signature = path_signature
        self._route_detail["planned_route_build_ms"] = (time.perf_counter() - _build_start) * 1000.0

        _paint_start = time.perf_counter()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        route_color = QColor(ORANGE)
        route_color.setAlpha(215)
        painter.setPen(QPen(route_color, 2.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
        robot_screen = self.world_to_screen(float(self.robot.x), float(self.robot.y))
        active_screen = self.world_to_screen(*future_points[0])
        painter.drawLine(QPointF(*robot_screen), QPointF(*active_screen))
        if len(future_points) >= 2 and self._planned_route_cache is not None:
            painter.drawPath(self._planned_route_cache)

        goal_xy = self.current_goal_xy()
        for offset, point in enumerate(future_points):
            endpoint = offset == len(future_points) - 1
            endpoint_is_goal = endpoint and math.hypot(
                point[0] - goal_xy[0], point[1] - goal_xy[1]
            ) <= max(0.20, self.config.goal_tolerance)
            self._draw_waypoint_marker(
                painter,
                point,
                active=(offset == 0),
                endpoint=endpoint,
                endpoint_is_goal=endpoint_is_goal,
                color=QColor(ORANGE),
            )

        painter.restore()
        self._route_detail["planned_route_paint_ms"] = (time.perf_counter() - _paint_start) * 1000.0

    def draw_multi_planned_routes(self, painter: QPainter):
        """Draw one remaining accepted route per robot, never historical paths."""
        if not self.config.show_path or not self.multi_planned_path_points or not self.robots:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        for robot_index, route in enumerate(self.multi_planned_path_points):
            if robot_index >= len(self.robots):
                break
            robot = self.robots[robot_index]
            active_index = self._active_route_index(robot, route)
            if active_index < 1:
                continue
            remaining = [
                (float(robot.x), float(robot.y)),
                *[tuple(map(float, point)) for point in route[active_index:]],
            ]
            if len(remaining) < 2:
                continue

            color = robot_color(robot_index)
            route_color = QColor(color)
            route_color.setAlpha(215)
            painter.setPen(QPen(route_color, 2.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
            # QPainterPath treats an open polyline as implicitly closed when
            # a brush is active.  _draw_waypoint_marker() intentionally uses
            # a purple brush for frontier endpoints, so without resetting the
            # brush here that state leaked into the *next* robot's route and
            # filled the area enclosed by its bends (often a large purple
            # wedge between the robot and F marker).  Routes are strokes only.
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._rebuild_route_path(remaining))

            goal_xy = self.current_goal_xy()
            future_points = remaining[1:]
            for offset, point in enumerate(future_points):
                endpoint = offset == len(future_points) - 1
                endpoint_is_goal = endpoint and math.hypot(
                    point[0] - goal_xy[0], point[1] - goal_xy[1]
                ) <= max(0.20, self.config.goal_tolerance)
                self._draw_waypoint_marker(
                    painter,
                    point,
                    active=(offset == 0),
                    endpoint=endpoint,
                    endpoint_is_goal=endpoint_is_goal,
                    color=color,
                    endpoint_radius=(
                        None
                        if endpoint_is_goal
                        else self._multi_frontier_marker_radius_px(robot_index)
                    ),
                    endpoint_fill=None if endpoint_is_goal else color,
                    endpoint_label=None if endpoint_is_goal else f"F{robot_index + 1}",
                )
        painter.restore()

    def draw_multi_executed_paths(self, painter: QPainter) -> None:
        """Draw each robot's complete trajectory using incremental rastering."""
        if not self.config.show_traveled_path or not self.multi_path_points:
            return

        _build_start = time.perf_counter()
        view_signature = self._view_transform_signature()
        style_signature = self._multi_executed_trail_style_signature()
        sources = tuple(self.multi_path_points)
        same_sources = (
            len(sources) == len(self._multi_executed_trail_sources)
            and all(
                current is cached
                for current, cached in zip(sources, self._multi_executed_trail_sources)
            )
        )
        truncated = (
            len(self._multi_executed_trail_counts) != len(sources)
            or any(
                len(points) < cached_count
                for points, cached_count in zip(
                    self.multi_path_points,
                    self._multi_executed_trail_counts,
                )
            )
        )

        if (
            self._multi_executed_trail_pixmap is None
            or not same_sources
            or self._multi_executed_trail_view_signature != view_signature
            or self._multi_executed_trail_style != style_signature
            or truncated
        ):
            self._rebuild_multi_executed_trail_cache(view_signature, style_signature)
            self._route_detail["executed_trail_cache_hit"] = False
        elif any(
            len(points) > cached_count
            for points, cached_count in zip(
                self.multi_path_points,
                self._multi_executed_trail_counts,
            )
        ):
            self._append_multi_executed_trail_segments()
            self._route_detail["executed_trail_cache_hit"] = True
        else:
            self._multi_executed_trail_segments_painted_last_frame = 0
            self._route_detail["executed_trail_cache_hit"] = True

        self._route_detail["executed_trail_build_ms"] = (
            time.perf_counter() - _build_start
        ) * 1000.0
        self._route_detail["executed_trail_points"] = sum(
            len(points) for points in self.multi_path_points
        )
        self._route_detail["executed_trail_segments_painted"] = (
            self._multi_executed_trail_segments_painted_last_frame
        )

        _paint_start = time.perf_counter()
        painter.save()
        painter.drawPixmap(0, 0, self._multi_executed_trail_pixmap)
        painter.restore()
        self._route_detail["executed_trail_paint_ms"] = (
            time.perf_counter() - _paint_start
        ) * 1000.0

    def _multi_executed_trail_style_signature(self) -> tuple:
        return (
            1.7,
            205,
            tuple(robot_color(index).name() for index in range(len(self.multi_path_points))),
        )

    def _multi_executed_trail_pen(self, robot_index: int) -> QPen:
        color = QColor(robot_color(robot_index))
        color.setAlpha(205)
        return QPen(color, 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _rebuild_multi_executed_trail_cache(
        self,
        view_signature: tuple,
        style_signature: tuple,
    ) -> None:
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        cache_painter = QPainter(pixmap)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        total_segments = 0
        last_points: list[tuple | None] = []
        for robot_index, points in enumerate(self.multi_path_points):
            cache_painter.setPen(self._multi_executed_trail_pen(robot_index))
            segments, last_point = self._paint_executed_trail_segments(
                cache_painter,
                points,
                None,
            )
            total_segments += segments
            last_points.append(last_point)
        cache_painter.end()

        self._multi_executed_trail_pixmap = pixmap
        self._multi_executed_trail_counts = [
            len(points) for points in self.multi_path_points
        ]
        self._multi_executed_trail_view_signature = view_signature
        self._multi_executed_trail_sources = tuple(self.multi_path_points)
        self._multi_executed_trail_style = style_signature
        self._multi_executed_trail_last_screen_points = last_points
        self._multi_executed_trail_segments_painted_last_frame = total_segments

    def _append_multi_executed_trail_segments(self) -> None:
        cache_painter = QPainter(self._multi_executed_trail_pixmap)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        total_segments = 0
        for robot_index, points in enumerate(self.multi_path_points):
            cache_painter.setPen(self._multi_executed_trail_pen(robot_index))
            new_points = points[self._multi_executed_trail_counts[robot_index]:]
            segments, last_point = self._paint_executed_trail_segments(
                cache_painter,
                new_points,
                self._multi_executed_trail_last_screen_points[robot_index],
            )
            total_segments += segments
            self._multi_executed_trail_counts[robot_index] = len(points)
            self._multi_executed_trail_last_screen_points[robot_index] = last_point
        cache_painter.end()
        self._multi_executed_trail_segments_painted_last_frame = total_segments

    def _executed_trail_style_signature(self) -> tuple:
        """Color/width the trail is stroked with. Currently fixed
        constants (no user-facing toggle exists yet), but kept as an
        explicit part of the cache key so that if a visibility/color/
        width control is ever added, changing it correctly invalidates
        the pixmap layer instead of leaving stale pixels behind."""
        return (BLUE, 1.7)

    def _paint_executed_trail_segments(
        self,
        cache_painter: QPainter,
        points: list,
        last_screen_point: tuple | None,
    ) -> tuple[int, tuple | None]:
        """Paint `points` as connected line segments continuing from
        last_screen_point (None if `points` starts the whole trail).
        Consecutive points landing within 1 screen pixel of the last
        actually-painted point are skipped -- visualization-only
        decimation; path_points/simulation data is never touched. Returns
        (segments_painted, new_last_screen_point)."""
        segments_painted = 0
        prev = last_screen_point
        for point in points:
            sx, sy = self.world_to_screen(*point)
            if prev is not None and abs(sx - prev[0]) <= 1.0 and abs(sy - prev[1]) <= 1.0:
                continue
            if prev is not None:
                cache_painter.drawLine(QPointF(prev[0], prev[1]), QPointF(sx, sy))
                segments_painted += 1
            prev = (sx, sy)
        return segments_painted, prev

    def _rebuild_executed_trail_cache(self, view_signature: tuple, style_signature: tuple) -> None:
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        cache_painter = QPainter(pixmap)
        cache_painter.setPen(QPen(QColor(style_signature[0]), style_signature[1]))
        segments_painted, last_point = self._paint_executed_trail_segments(
            cache_painter, self.path_points, None,
        )
        cache_painter.end()

        self._executed_trail_pixmap = pixmap
        self._executed_trail_pixmap_count = len(self.path_points)
        self._executed_trail_view_signature = view_signature
        self._executed_trail_style = style_signature
        self._executed_trail_source = self.path_points
        self._executed_trail_last_screen_point = last_point
        self._executed_trail_segments_painted_last_frame = segments_painted

    def _append_executed_trail_segments(self) -> None:
        cache_painter = QPainter(self._executed_trail_pixmap)
        style = self._executed_trail_style
        cache_painter.setPen(QPen(QColor(style[0]), style[1]))
        new_points = self.path_points[self._executed_trail_pixmap_count:]
        segments_painted, last_point = self._paint_executed_trail_segments(
            cache_painter, new_points, self._executed_trail_last_screen_point,
        )
        cache_painter.end()

        self._executed_trail_pixmap_count = len(self.path_points)
        self._executed_trail_last_screen_point = last_point
        self._executed_trail_segments_painted_last_frame = segments_painted

    def draw_executed_path(self, painter: QPainter):
        """The executed trail can grow to hundreds/thousands of points
        over a long run. It was previously cached as a single
        QPainterPath, appended to incrementally rather than rebuilt --
        but painter.drawPath() still rasterizes the ENTIRE accumulated
        path on every single paintEvent, so per-frame paint cost grew
        unboundedly with total trail length even though the path object
        itself was never rebuilt (this is exactly what the real
        Office.sim route_path_ms evidence showed: 17ms growing to
        431ms as the run went on).

        The trail is now painted into a persistent QPixmap instead: new
        points are painted into it ONCE, the moment they arrive, and every
        frame just blits the pixmap -- drawPixmap() cost depends on screen
        area, not on how many points were ever painted into it. Rebuilt
        (not incrementally appended to) when: the view transform changes,
        a reset replaces path_points (a new list object -- see the `is`
        check below), the list is explicitly truncated by an external
        caller, or the trail's stroke style changes."""
        if not self.config.show_traveled_path:
            return
        if len(self.path_points) < 2:
            self._route_detail["executed_trail_points"] = len(self.path_points)
            self._route_detail["executed_trail_segments_painted"] = 0
            self._route_detail["executed_trail_build_ms"] = 0.0
            self._route_detail["executed_trail_paint_ms"] = 0.0
            return

        _build_start = time.perf_counter()
        view_signature = self._view_transform_signature()
        style_signature = self._executed_trail_style_signature()
        same_source = self._executed_trail_source is self.path_points
        same_view = self._executed_trail_view_signature == view_signature
        same_style = self._executed_trail_style == style_signature
        truncated = (
            self._executed_trail_pixmap is not None
            and len(self.path_points) < self._executed_trail_pixmap_count
        )

        if self._executed_trail_pixmap is None or not same_source or not same_view or not same_style or truncated:
            self._rebuild_executed_trail_cache(view_signature, style_signature)
            self._route_detail["executed_trail_cache_hit"] = False
        elif len(self.path_points) > self._executed_trail_pixmap_count:
            self._append_executed_trail_segments()
            self._route_detail["executed_trail_cache_hit"] = True
        else:
            self._executed_trail_segments_painted_last_frame = 0
            self._route_detail["executed_trail_cache_hit"] = True
        self._route_detail["executed_trail_build_ms"] = (time.perf_counter() - _build_start) * 1000.0
        self._route_detail["executed_trail_points"] = len(self.path_points)
        self._route_detail["executed_trail_segments_painted"] = self._executed_trail_segments_painted_last_frame

        _paint_start = time.perf_counter()
        painter.save()
        painter.drawPixmap(0, 0, self._executed_trail_pixmap)
        painter.restore()
        self._route_detail["executed_trail_paint_ms"] = (time.perf_counter() - _paint_start) * 1000.0

    def draw_robot_icon(
        self,
        painter: QPainter,
        sx: float,
        sy: float,
        body_px: float,
        color: QColor,
        *,
        theta: float = 0.0,
        label: str | None = None,
        outline_width: float = 2.0,
        label_font_size: int = 8,
    ) -> None:
        """Draw one robot marker using the selected presentation icon.

        All three choices are vector shapes, so no external image assets or
        GUI-scale-dependent pixmaps are needed. The icon choice is rendering
        only and does not alter body/safety radii or collision geometry.
        """
        icon = getattr(self.config, "robot_icon", DEFAULT_ROBOT_ICON)
        requested_radius = float(body_px)
        radius = (
            requested_radius
            if icon == "Circle"
            else max(6.0, requested_radius)
        )

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(float(sx), float(sy))
        painter.rotate(-math.degrees(float(theta)))

        outline = QColor(255, 255, 255, 245)
        dark_detail = QColor(28, 31, 36, 235)

        if icon == "Drone":
            arm_extent = radius * 0.72
            painter.setPen(QPen(outline, max(1.4, radius * 0.18), Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(-arm_extent, -arm_extent), QPointF(arm_extent, arm_extent))
            painter.drawLine(QPointF(-arm_extent, arm_extent), QPointF(arm_extent, -arm_extent))

            rotor_radius = max(2.1, radius * 0.27)
            painter.setPen(QPen(outline, max(1.0, radius * 0.12)))
            painter.setBrush(QBrush(color.darker(118)))
            for cx, cy in (
                (-arm_extent, -arm_extent),
                (arm_extent, -arm_extent),
                (-arm_extent, arm_extent),
                (arm_extent, arm_extent),
            ):
                painter.drawEllipse(
                    QRectF(
                        cx - rotor_radius,
                        cy - rotor_radius,
                        2.0 * rotor_radius,
                        2.0 * rotor_radius,
                    )
                )

            body = QRectF(-radius * 0.48, -radius * 0.36, radius * 0.96, radius * 0.72)
            painter.setPen(QPen(outline, max(1.2, radius * 0.14)))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(body, radius * 0.18, radius * 0.18)

        elif icon == "Wheeled Robot":
            body = QRectF(-radius * 0.76, -radius * 0.58, radius * 1.52, radius * 1.16)
            wheel_w = max(2.0, radius * 0.24)
            wheel_h = radius * 0.72
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(dark_detail))
            painter.drawRoundedRect(
                QRectF(-radius * 0.88, -wheel_h / 2.0, wheel_w, wheel_h),
                wheel_w * 0.35,
                wheel_w * 0.35,
            )
            painter.drawRoundedRect(
                QRectF(radius * 0.88 - wheel_w, -wheel_h / 2.0, wheel_w, wheel_h),
                wheel_w * 0.35,
                wheel_w * 0.35,
            )

            painter.setPen(QPen(outline, max(1.2, radius * 0.14)))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(body, radius * 0.22, radius * 0.22)

            # Front-direction marker. It makes orientation visible without
            # bringing back the old always-on heading arrow.
            painter.setPen(QPen(outline, max(1.2, radius * 0.13), Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(
                QPointF(radius * 0.30, -radius * 0.28),
                QPointF(radius * 0.62, 0.0),
            )
            painter.drawLine(
                QPointF(radius * 0.62, 0.0),
                QPointF(radius * 0.30, radius * 0.28),
            )

        else:
            painter.setPen(QPen(outline, float(outline_width)))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QRectF(-radius, -radius, 2.0 * radius, 2.0 * radius))

        if label:
            # Keep IDs upright even though drone/wheeled shapes rotate with
            # the robot heading.
            painter.save()
            painter.rotate(math.degrees(float(theta)))
            painter.setFont(QFont("Segoe UI", int(label_font_size), QFont.Bold))
            painter.setPen(QPen(outline))
            painter.drawText(
                QRectF(-radius, -radius, 2.0 * radius, 2.0 * radius),
                Qt.AlignCenter,
                str(label),
            )
            painter.restore()

        painter.restore()

    def draw_frontier_reasoning_overlay(self, painter: QPainter) -> None:
        """Draw the selected frontier's live geometric inputs beside the decision."""
        if (
            not self.frontier_reasoning_overlay_enabled
            or not self.frontier_reasoning_simulation_paused
            or not self.frontier_reasoning_decision
        ):
            return
        data = self.frontier_reasoning_decision
        robot = data.get("robot")
        frontier = data.get("frontier")
        if robot is None or frontier is None:
            return
        rx, ry = float(robot[0]), float(robot[1])
        fx, fy = float(frontier[0]), float(frontier[1])
        rsx, rsy = self.world_to_screen(rx, ry)
        fsx, fsy = self.world_to_screen(fx, fy)
        distance = float(data.get("distance", math.hypot(fx - rx, fy - ry)))
        terms = list(data.get("terms", ()))[:4]

        painter.save()
        accent = QColor(245, 166, 35, 235)
        painter.setPen(QPen(accent, 2.0, Qt.DashLine))
        painter.drawLine(QPointF(rsx, rsy), QPointF(fsx, fsy))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))

        def label(x: float, y: float, text: str) -> None:
            metrics = painter.fontMetrics()
            rect = metrics.boundingRect(text).adjusted(-6, -4, 6, 4)
            box = QRectF(x, y, rect.width(), rect.height())
            painter.fillRect(box, QColor(20, 27, 38, 220))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(box, Qt.AlignCenter, text)

        label(rsx + 12, rsy - 30, f"R ({rx:.2f}, {ry:.2f})")
        label(fsx + 12, fsy - 30, f"F ({fx:.2f}, {fy:.2f})")
        mid_x, mid_y = (rsx + fsx) / 2.0, (rsy + fsy) / 2.0
        label(mid_x + 8, mid_y - 22, f"d(R,F) = {distance:.2f} m")
        if terms:
            label(fsx + 12, fsy + 8, "  |  ".join(str(term) for term in terms))
        painter.restore()

    def draw_frontier_clusters(self, painter: QPainter) -> None:
        """Color the actual connected frontier components exported by the planner."""
        if (
            not self.frontier_reasoning_overlay_enabled
            or not self.frontier_reasoning_simulation_paused
            or not self.frontier_reasoning_cluster_view_enabled
            or not self.frontier_reasoning_clusters
        ):
            return
        palette = (
            QColor(0, 174, 239), QColor(238, 75, 138), QColor(84, 190, 121),
            QColor(156, 102, 204), QColor(255, 145, 45), QColor(32, 191, 184),
            QColor(225, 196, 35), QColor(95, 122, 230),
        )
        painter.save()
        for index, cluster in enumerate(self.frontier_reasoning_clusters):
            color = QColor(palette[index % len(palette)])
            fill = QColor(color)
            fill.setAlpha(48)
            edge = QColor(color)
            edge.setAlpha(125)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(edge, 1.0))
            resolution = max(0.01, float(cluster.get("resolution", 0.25) or 0.25))
            half = resolution * 0.5
            for x, y in cluster.get("points", ()):
                sx0, sy0 = self.world_to_screen(float(x) - half, float(y) - half)
                sx1, sy1 = self.world_to_screen(float(x) + half, float(y) + half)
                rect = QRectF(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.restore()

    def draw_frontier_candidate_inspection(self, painter: QPainter) -> None:
        """Draw a compact focus ring for the candidate browsed in the panel."""
        if (
            not self.frontier_reasoning_overlay_enabled
            or not self.frontier_reasoning_simulation_paused
            or not self.frontier_reasoning_inspection
        ):
            return
        frontier = self.frontier_reasoning_inspection.get("frontier")
        if frontier is None:
            return
        sx, sy = self.world_to_screen(float(frontier[0]), float(frontier[1]))
        index = int(self.frontier_reasoning_inspection.get("index", 0))
        count = int(self.frontier_reasoning_inspection.get("count", 0))
        painter.save()
        glow = QColor(255, 196, 46, 70)
        accent = QColor(255, 174, 0, 245)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(QRectF(sx - 22, sy - 22, 44, 44))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(accent, 4.0))
        painter.drawEllipse(QRectF(sx - 14, sy - 14, 28, 28))
        painter.setBrush(QBrush(QColor(20, 27, 38, 225)))
        painter.setPen(QPen(QColor("white"), 1.5))
        badge = QRectF(sx + 11, sy - 25, 42, 20)
        painter.drawRoundedRect(badge, 5, 5)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(badge, Qt.AlignCenter, f"{index}/{count}")
        painter.restore()

    def draw_cursor_coordinates(self, painter: QPainter) -> None:
        """Draw the world coordinate beside the mouse while it is over the map."""
        if (
            not self.cursor_coordinates_enabled
            or self.cursor_coordinate_position is None
            or self.cursor_coordinate_world is None
        ):
            return
        mouse_x, mouse_y = self.cursor_coordinate_position
        world_x, world_y = self.cursor_coordinate_world
        text = f"({world_x:.2f}, {world_y:.2f})"
        plot = self.plot_rect()
        painter.save()
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text).adjusted(-7, -4, 7, 4)
        box_x = mouse_x + 14.0
        box_y = mouse_y + 14.0
        if box_x + text_rect.width() > plot.right():
            box_x = mouse_x - text_rect.width() - 14.0
        if box_y + text_rect.height() > plot.bottom():
            box_y = mouse_y - text_rect.height() - 14.0
        box = QRectF(box_x, box_y, text_rect.width(), text_rect.height())
        painter.setPen(QPen(QColor(255, 255, 255, 210), 1.0))
        painter.setBrush(QBrush(QColor(20, 27, 38, 225)))
        painter.drawRoundedRect(box, 5, 5)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignCenter, text)
        painter.restore()

    def draw_goal_and_robot(self, painter: QPainter):
        history_position, _history_total = self._nav_debug_history_position
        if self.navigation_debug_enabled and history_position is not None and self._nav_debug_snapshot is not None:
            snapshot = self._nav_debug_snapshot
            rx, ry = self.world_to_screen(snapshot.robot_pose.x, snapshot.robot_pose.y)
            body_px = max(5.0, float(snapshot.safety.robot_radius) * self.pixels_per_meter())
            self.draw_robot_icon(
                painter,
                rx,
                ry,
                body_px,
                QColor(BLUE),
                theta=float(snapshot.robot_pose.theta),
            )
            return

        x, y, theta, _ = self.current_robot_pose()
        gx, gy = self.current_goal_xy()

        rx, ry = self.world_to_screen(x, y)
        gx_s, gy_s = self.world_to_screen(gx, gy)

        # Goal marker: only in Goal Seeking mode. In exploration mode the
        # GUI final goal G is not executable (the exploration planner picks
        # its own frontier targets -- see navigation_modes.py's docstring),
        # so showing it there was misleading.
        if is_goal_seeking_planner(self.config.exploration_planner):
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(GREEN_LIGHT)))
            painter.drawEllipse(QRectF(gx_s - 15, gy_s - 15, 30, 30))
            painter.setBrush(QBrush(QColor(GREEN)))
            painter.drawEllipse(QRectF(gx_s - 8, gy_s - 8, 16, 16))
            painter.setBrush(QBrush(QColor("white")))
            painter.drawEllipse(QRectF(gx_s - 3, gy_s - 3, 6, 6))
            painter.restore()

        # Exploration target markers belong to the planned-route layer. In
        # multi-robot mode each robot owns its own F marker; do not draw a
        # single shared F because that makes the robots look coupled.
        if self.config.show_path and self.robots and "Multiple" in self.config.agent_mode:
            painter.save()
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            for target_index, target in enumerate(self.multi_exploration_targets):
                if target is None:
                    continue
                tx, ty = float(target[0]), float(target[1])
                if math.hypot(tx - gx, ty - gy) <= max(0.20, self.config.goal_tolerance):
                    continue
                route = (
                    self.multi_planned_path_points[target_index]
                    if target_index < len(self.multi_planned_path_points)
                    else []
                )
                if route and math.hypot(
                    float(route[-1][0]) - tx, float(route[-1][1]) - ty
                ) <= max(0.20, self.config.goal_tolerance):
                    # The remaining-route layer already draws the endpoint as F.
                    continue
                tx_s, ty_s = self.world_to_screen(tx, ty)
                color = robot_color(target_index)
                radius = self._multi_frontier_marker_radius_px(target_index)
                painter.setPen(QPen(QColor("white"), 2.0))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QRectF(tx_s - radius, ty_s - radius, 2 * radius, 2 * radius))
                painter.setPen(QPen(QColor("white")))
                painter.drawText(
                    QRectF(tx_s - radius, ty_s - radius, 2 * radius, 2 * radius),
                    Qt.AlignCenter,
                    f"F{target_index + 1}",
                )
            painter.restore()
        elif self.config.show_path and self.exploration_target_xy is not None:
            tx, ty = self.exploration_target_xy
            route_represents_target = bool(self.planned_path_points) and math.hypot(
                float(self.planned_path_points[-1][0]) - tx,
                float(self.planned_path_points[-1][1]) - ty,
            ) <= max(0.20, self.config.goal_tolerance)
            if (
                not route_represents_target
                and math.hypot(tx - gx, ty - gy) > max(0.20, self.config.goal_tolerance)
            ):
                tx_s, ty_s = self.world_to_screen(tx, ty)
                painter.save()
                painter.setPen(QPen(QColor("white"), 2.0))
                painter.setBrush(QBrush(QColor(146, 62, 160)))
                painter.drawEllipse(QRectF(tx_s - 10, ty_s - 10, 20, 20))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QPen(QColor("white")))
                painter.drawText(QRectF(tx_s - 10, ty_s - 10, 20, 20), Qt.AlignCenter, "F")
                painter.restore()

        # Multi-robot preview: before the simulation starts, show every robot
        # start pose and allow click-drag placement. The runtime multi-robot
        # controller is a separate implementation step; this keeps configuration
        # stable first.
        if self.robot is None and "Multiple" in self.config.agent_mode:
            painter.save()
            px_per_meter = self.pixels_per_meter()
            selected_index = max(0, min(int(self.config.selected_robot_index), int(self.config.robot_count) - 1))

            for index, robot_cfg in enumerate(normalized_robot_start_configs(self.config)):
                sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
                body_px = max(7.0, float(robot_cfg.body_radius) * px_per_meter)
                is_selected = index == selected_index

                if is_selected:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(225, 126, 38, 45)))
                    painter.drawEllipse(QRectF(sx - body_px - 8, sy - body_px - 8, 2 * (body_px + 8), 2 * (body_px + 8)))

                self.draw_robot_icon(
                    painter,
                    sx,
                    sy,
                    body_px,
                    robot_color(index),
                    theta=float(robot_cfg.theta),
                    label=str(index + 1),
                    outline_width=2.4 if is_selected else 1.8,
                    label_font_size=8,
                )

            painter.restore()
            return

        # Runtime multi-robot drawing. This is the first executable multi-robot
        # baseline: every robot is visible and moves as an independent agent.
        # The heading arrow and the executed-trail line were dropped (were
        # gated behind the removed "Robot Orders" toggle) -- the Navigation
        # Debug overlay's heading ray is the single-robot replacement; no
        # multi-robot equivalent exists yet.
        if self.robots and "Multiple" in self.config.agent_mode:
            painter.save()

            for index, robot in enumerate(self.robots):
                sx, sy = self.world_to_screen(float(robot.x), float(robot.y))
                color = robot_color(index)
                body_px = self._multi_robot_body_radius_px(index)

                self.draw_robot_icon(
                    painter,
                    sx,
                    sy,
                    body_px,
                    color,
                    theta=float(getattr(robot, "theta", 0.0)),
                    label=str(index + 1),
                    outline_width=2.2,
                    label_font_size=8,
                )

            painter.restore()
            return

        # Robot marker: always visible. Its size follows body_radius. The
        # heading arrow was dropped -- the Navigation Debug overlay draws a
        # heading ray instead (see draw_navigation_debug_heading_rays()),
        # only when that layer is active, instead of an always-on red arrow.
        px_per_meter = self.pixels_per_meter()
        body_px = max(5.0, float(self.config.body_radius) * px_per_meter)
        self.draw_robot_icon(
            painter,
            rx,
            ry,
            body_px,
            QColor(BLUE),
            theta=float(theta),
        )

    def _navigation_debug_pick_live_terms(self, snapshot):
        """Pick whichever safety check is most relevant to show as the live
        formula next to the robot: a currently-blocked predicted-motion
        check is the most urgent, then the always-live active-segment
        check (computed every tick there is a target), else a clear
        predicted-motion result, else nothing (idle / no target)."""
        predicted = snapshot.predicted_motion.collision
        if not predicted.unavailable and predicted.value is not None and predicted.value.blocked:
            return "predicted", predicted.value
        active = snapshot.safety.active_segment
        if not active.unavailable and active.value is not None:
            return "active segment", active.value
        if not predicted.unavailable and predicted.value is not None:
            return "predicted", predicted.value
        return None, None

    def draw_navigation_debug_heading_rays(self, painter: QPainter):
        """World-space rays/arc around the robot answering "why did it
        turn": current heading (white), desired heading toward the active
        target (cyan), and the angular error arc between them. Every angle
        drawn is read straight off the snapshot (robot_pose.theta,
        controller.desired_heading) -- nothing here is computed."""
        snapshot = self._nav_debug_snapshot
        if snapshot is None:
            return

        px_per_meter = self.pixels_per_meter()
        rx, ry = self.world_to_screen(snapshot.robot_pose.x, snapshot.robot_pose.y)
        ray_len = max(28.0, snapshot.safety.safety_radius * px_per_meter * 1.8)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        theta = snapshot.robot_pose.theta
        hx = rx + ray_len * math.cos(theta)
        hy = ry - ray_len * math.sin(theta)
        painter.setPen(QPen(QColor(255, 255, 255), 2.4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(rx, ry), QPointF(hx, hy))
        painter.setPen(QPen(QColor(40, 40, 40)))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QPointF(hx + 3, hy), f"θ={math.degrees(theta):.0f}°")

        if not snapshot.controller.desired_heading.unavailable:
            theta_t = snapshot.controller.desired_heading.value
            dx = rx + ray_len * math.cos(theta_t)
            dy = ry - ray_len * math.sin(theta_t)
            painter.setPen(QPen(QColor(0, 188, 212), 2.2, Qt.DashLine, Qt.RoundCap))
            painter.drawLine(QPointF(rx, ry), QPointF(dx, dy))
            painter.setPen(QPen(QColor(0, 140, 165)))
            painter.drawText(QPointF(dx + 3, dy), f"θt={math.degrees(theta_t):.0f}°")

            # Angular-error arc between the two rays.
            arc_r = ray_len * 0.55
            start_deg = math.degrees(theta)
            span_deg = math.degrees(theta_t - theta)
            span_deg = (span_deg + 180.0) % 360.0 - 180.0
            painter.setPen(QPen(QColor(230, 140, 20), 1.6, Qt.SolidLine))
            painter.drawArc(
                QRectF(rx - arc_r, ry - arc_r, arc_r * 2, arc_r * 2),
                int(-start_deg * 16),
                int(-span_deg * 16),
            )

        painter.restore()

    def draw_navigation_debug_waypoint_line(self, painter: QPainter):
        """World-space line from the robot to the active waypoint, with the
        real distance already computed by the controller (goal_metrics())
        -- not recomputed here."""
        snapshot = self._nav_debug_snapshot
        if snapshot is None or snapshot.path.active_segment is None:
            return
        start, end = snapshot.path.active_segment
        if end is None:
            return

        sx, sy = self.world_to_screen(*start)
        ex, ey = self.world_to_screen(*end)
        painter.save()
        painter.setPen(QPen(QColor(0, 188, 212, 160), 1.4, Qt.DashDotLine))
        painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))
        painter.setPen(QPen(QColor(255, 255, 255), 2.0))
        painter.setBrush(QBrush(QColor(0, 188, 212)))
        painter.drawEllipse(QRectF(ex - 5, ey - 5, 10, 10))
        if not snapshot.controller.distance_to_goal.unavailable:
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(0, 120, 140))
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            painter.drawText(QPointF(mx + 4, my - 4), f"d={snapshot.controller.distance_to_goal.value:.2f}m")
        painter.restore()

    def _navigation_debug_label_color(self, color: QColor) -> QColor:
        """This label's backing plate (see below) is always a light,
        translucent white -- deliberately, so the map stays visible through
        it -- regardless of the app's own light/dark theme. Composited over
        a dark canvas that translucent white reads as a mid-grey rather than
        near-white, so the semantic colors below (tuned for a near-white
        backing) lose contrast in dark mode. Darken them for dark mode only;
        the hue/meaning (grey=neutral, green=live/ok, orange=rotate/history,
        red=blocked) never changes, and light mode is untouched."""
        if self._theme_mode == ThemeMode.DARK:
            return color.darker(150)
        return color

    def draw_navigation_debug_robot_label(self, painter: QPainter):
        """Compact floating readout anchored above the robot, following it
        every frame like a nameplate -- short formulas only (the full
        breakdown lives in the docked NavigationReasoningWindow).
        Nothing here is computed: every value is read straight off the
        snapshot. Background is translucent and the label is offset past
        the safety-radius ring so it never covers the robot or the map
        underneath it.
        """
        snapshot = self._nav_debug_snapshot
        if snapshot is None:
            return

        lines: list[str] = []
        colors: list[QColor] = []
        accent = self._navigation_debug_label_color(QColor(TEXT_MUTED))

        mode_line = snapshot.tracking_mode or snapshot.navigation_state
        lines.append(f"{mode_line} · {snapshot.decision_kind}")
        colors.append(self._navigation_debug_label_color(QColor(TEXT_MUTED)))

        if not snapshot.controller.heading_error.unavailable and not snapshot.rotate_threshold.unavailable:
            eth = math.degrees(snapshot.controller.heading_error.value)
            thr = math.degrees(snapshot.rotate_threshold.value)
            rotate = abs(eth) > thr
            accent = self._navigation_debug_label_color(QColor(ORANGE) if rotate else QColor(GREEN))
            lines.append(f"|eθ|={abs(eth):.1f}° {'>' if rotate else '≤'} thr={thr:.1f}°")
            lines.append(f"ROTATE={rotate}")
            colors.append(accent)
            colors.append(accent)

        checker_label, terms = self._navigation_debug_pick_live_terms(snapshot)
        if terms is not None and terms.blocked:
            distance_text = "n/a" if terms.distance.unavailable else f"{terms.distance.value:.2f}m"
            accent = self._navigation_debug_label_color(QColor(RED))
            lines.append(f"{checker_label}: BLOCKED d={distance_text}<r={terms.required_clearance:.2f}m")
            colors.append(accent)

        position, total = self._nav_debug_history_position
        if position is not None:
            view_color = self._navigation_debug_label_color(QColor(ORANGE))
            view_text = f"HISTORY {position}/{total}"
        else:
            view_color = self._navigation_debug_label_color(QColor(GREEN))
            view_text = "LIVE"
        lines.append(view_text)
        colors.append(view_color)

        px_per_meter = self.pixels_per_meter()
        rx, ry = self.world_to_screen(snapshot.robot_pose.x, snapshot.robot_pose.y)
        safety_r_px = snapshot.safety.safety_radius * px_per_meter

        painter.save()
        painter.setFont(QFont("Segoe UI", 7))
        metrics = painter.fontMetrics()
        line_height = metrics.height() + 1
        width = max(metrics.horizontalAdvance(line) for line in lines) + 14
        height = line_height * len(lines) + 7

        # Anchored above and to the right of the safety-radius ring -- past
        # the robot's own footprint, like a nameplate that follows it
        # without ever covering it.
        label_x = rx + safety_r_px * 0.35 + 6
        label_y = ry - safety_r_px - height - 8
        rect = QRectF(label_x, label_y, width, height)

        path = QPainterPath()
        path.addRoundedRect(rect, 5, 5)
        # Deliberately light: translucent fill so the map/route underneath
        # stays visible through the label, not an opaque card.
        painter.fillPath(path, QColor(255, 255, 255, 120))
        painter.setPen(QPen(accent, 1.0))
        painter.drawPath(path)

        for i, (line, color) in enumerate(zip(lines, colors)):
            painter.setPen(color)
            painter.drawText(QPointF(rect.left() + 6, rect.top() + 5 + line_height * (i + 1) - 3), line)

        # Thin leader line from the robot to the label, like a callout.
        painter.setPen(QPen(accent, 1.0, Qt.DotLine))
        painter.drawLine(QPointF(rx, ry - safety_r_px * 0.3), QPointF(rect.left(), rect.bottom()))
        painter.restore()

    # The full field breakdown ("NAVIGATION REASONING") now lives in the
    # standalone NavigationReasoningWindow (see navigation_reasoning_
    # window.py) instead of a fixed card drawn on top of the canvas -- a
    # side panel does not overlap the map/title/FPS. main_window.py forwards snapshot/event/
    # history-position pushes to it directly; the canvas keeps only the
    # compact near-robot label and the world-space annotations.

