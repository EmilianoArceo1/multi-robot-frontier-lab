"""
Phase 4: SimulationCanvas must render only the team's DISCOVERED hazard
belief during live simulation -- never the omniscient ground-truth
HazardField.

    - draw_fires()/set_hazard_snapshot()/_build_hazard_pixmap() (ground
      truth) are kept intact for legacy/potential editor use and their own
      existing tests (test_fire_hazards.py, test_theme_palette.py), but are
      no longer called from the live paint loop.
    - draw_discovered_hazard()/set_discovered_hazard_frame()/
      _build_discovered_hazard_pixmap() are the new runtime path: a cell
      produces a pixel only when HazardBeliefFrame.observed is True for it.

Same testing approach as test_canvas_render_cache.py: a real
SimulationCanvas instance (needs a QApplication), never .show()'d. Pixel
inspection via QImage at known grid-aligned coordinates, cache-object
identity, and frame contents -- never a full-screenshot comparison.
"""
from __future__ import annotations

import inspect

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.app.theme import ThemeMode
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.simulation.hazard_service import RuntimeHazardService

_app = QApplication.instance() or QApplication([])

_BOUNDS = (0.0, 5.0, 0.0, 5.0)
_RESOLUTION = 1.0  # -> 5x5 grid, cell (row, col) centers at integers + 0.5


def _make_canvas(width: int = 200, height: int = 200) -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(width, height)
    return canvas


def _make_belief(robot_count: int = 1) -> HazardBelief:
    return HazardBelief(GridGeometry(_BOUNDS, _RESOLUTION), robot_count=robot_count)


def _payload(belief: HazardBelief) -> dict:
    return {"frame": belief.snapshot(), "bounds": _BOUNDS, "resolution": _RESOLUTION}


def _draw_once(canvas: SimulationCanvas) -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_discovered_hazard(painter)
    painter.end()


def _pixel_alpha(pixmap: QPixmap, row: int, col: int) -> int:
    """Grid row 0 is the world's lower edge, flipped to the image's bottom
    row by _build_discovered_hazard_pixmap() -- mirror that here."""
    image = pixmap.toImage()
    return image.pixelColor(col, image.height() - 1 - row).alpha()


def _pixel_rgb(pixmap: QPixmap, row: int, col: int) -> tuple[int, int, int]:
    image = pixmap.toImage()
    color = image.pixelColor(col, image.height() - 1 - row)
    return (color.red(), color.green(), color.blue())


# ---------------------------------------------------------------------------
# 1. Ground truth hazard that was never observed produces a transparent frame.
# ---------------------------------------------------------------------------


def test_unobserved_ground_truth_produces_a_transparent_frame():
    canvas = _make_canvas()
    belief = _make_belief()  # nothing ever observed
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is None, (
        "no observed cells at all must never produce a heatmap pixmap"
    )


# ---------------------------------------------------------------------------
# 2/14. FireSource is never a rendering input for runtime.
# ---------------------------------------------------------------------------


def test_render_functions_never_reference_fire_source_or_ground_truth():
    draw_source = inspect.getsource(SimulationCanvas.draw_discovered_hazard)
    build_source = inspect.getsource(SimulationCanvas._build_discovered_hazard_pixmap)
    combined = draw_source + build_source

    # Checked as actual code patterns (constructor/attribute/call), not bare
    # substrings -- both methods' own docstrings mention these names in
    # prose (explaining what must NOT happen) without violating the rule.
    for forbidden in ("FireSource(", "hazard_service.", ".field.", ".sources("):
        assert forbidden not in combined, (
            f"draw_discovered_hazard()/_build_discovered_hazard_pixmap() must never "
            f"contain {forbidden!r} -- runtime rendering must not be omniscient"
        )


# ---------------------------------------------------------------------------
# 3/4. Observed hot cells appear; observed safe cells produce no heatmap.
# ---------------------------------------------------------------------------


def test_observed_hot_cell_appears():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas)

    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) > 0


def test_observed_safe_cell_produces_no_heatmap():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.0], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas)

    # observed=True but value=0.0 -- the frame is non-empty (something was
    # observed) so a pixmap IS built, but that specific cell stays alpha 0.
    assert canvas._discovered_hazard_pixmap_cache is not None
    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) == 0


# ---------------------------------------------------------------------------
# 5/6. Unobserved cells stay transparent; partial observation shows only the
# observed area.
# ---------------------------------------------------------------------------


def test_unobserved_cells_remain_transparent_next_to_an_observed_one():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.9], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas)
    pixmap = canvas._discovered_hazard_pixmap_cache

    assert _pixel_alpha(pixmap, 0, 0) == 0
    assert _pixel_alpha(pixmap, 4, 4) == 0


def test_partial_observation_shows_only_the_observed_area():
    canvas = _make_canvas()
    belief = _make_belief()
    # A 2x2 observed block, hot enough to be visible.
    belief.observe_cells([1, 1, 2, 2], [1, 2, 1, 2], [0.7, 0.7, 0.7, 0.7], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas)
    pixmap = canvas._discovered_hazard_pixmap_cache

    for row in range(5):
        for col in range(5):
            alpha = _pixel_alpha(pixmap, row, col)
            if row in (1, 2) and col in (1, 2):
                assert alpha > 0, f"observed hot cell ({row},{col}) must be visible"
            else:
                assert alpha == 0, f"unobserved cell ({row},{col}) must stay transparent"


# ---------------------------------------------------------------------------
# 7/8. Removal semantics: removing a fire outside the FoV keeps the last
# visualization; re-observing after removal clears it.
# ---------------------------------------------------------------------------


def test_removing_fire_outside_fov_keeps_the_last_visualization():
    """Removing a FireSource never touches HazardBelief (see hazard_service.
    RuntimeHazardService's own contract/tests) -- pushing the SAME,
    unchanged belief frame again must render identically."""
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    payload = _payload(belief)
    canvas.set_discovered_hazard_frame(payload)
    _draw_once(canvas)
    alpha_before = _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2)
    assert alpha_before > 0

    # "Fire removed outside the FoV": ground truth changes elsewhere, but
    # the belief (and therefore the frame already pushed) is untouched --
    # simulate by simply re-pushing the identical, unchanged frame.
    canvas.set_discovered_hazard_frame(payload)
    _draw_once(canvas)

    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) == alpha_before


def test_reobserving_after_removal_clears_the_visualization():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) > 0

    # Re-observing after the fire was removed -- ground truth is now 0.0 at
    # that cell, ovserved stays True.
    belief.observe_cells([2], [2], [0.0], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)

    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) == 0


# ---------------------------------------------------------------------------
# 9. Attribution-only changes (observed_by_robot) never alter the visual.
# ---------------------------------------------------------------------------


def test_attribution_only_change_does_not_alter_the_visual_result():
    canvas = _make_canvas()
    belief = _make_belief(robot_count=2)
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    alpha_before = _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2)
    rgb_before = _pixel_rgb(canvas._discovered_hazard_pixmap_cache, 2, 2)
    revision_before = belief.revision

    # Robot 1 observes the SAME cell with the SAME value -- only
    # observed_by_robot changes; team values/observed do not.
    belief.observe_cells([2], [2], [0.8], robot_index=1)
    assert belief.revision > revision_before  # sanity: attribution did bump revision
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)

    assert _pixel_alpha(canvas._discovered_hazard_pixmap_cache, 2, 2) == alpha_before
    assert _pixel_rgb(canvas._discovered_hazard_pixmap_cache, 2, 2) == rgb_before


# ---------------------------------------------------------------------------
# 10/11. Cache reuse and invalidation keyed on revision.
# ---------------------------------------------------------------------------


def test_repeating_the_same_revision_reuses_the_cache():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    cache_after_first_draw = canvas._discovered_hazard_pixmap_cache
    assert cache_after_first_draw is not None

    # Push an equal (same revision/bounds/resolution) frame again.
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is cache_after_first_draw


def test_new_revision_invalidates_the_cache():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    cache_after_first_draw = canvas._discovered_hazard_pixmap_cache

    belief.observe_cells([3], [3], [0.6], robot_index=0)  # bumps revision
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is not cache_after_first_draw


# ---------------------------------------------------------------------------
# 12. Changing the viewport preserves alignment without rebuilding the cache.
# ---------------------------------------------------------------------------


def test_changing_viewport_does_not_rebuild_the_cache():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    cache_before = canvas._discovered_hazard_pixmap_cache
    cache_key_before = canvas._discovered_hazard_pixmap_cache_key

    canvas.resize(500, 400)  # viewport/transform change, same frame
    canvas.invalidate_view_transform_caches()
    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is cache_before
    assert canvas._discovered_hazard_pixmap_cache_key == cache_key_before


# ---------------------------------------------------------------------------
# 13. Theme changes never alter hazard semantics -- the cache is not even
# theme-keyed, so it survives a theme switch untouched.
# ---------------------------------------------------------------------------


def test_theme_change_preserves_hazard_semantics():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)
    cache_before = canvas._discovered_hazard_pixmap_cache
    rgb_before = _pixel_rgb(cache_before, 2, 2)

    canvas.set_theme_mode(ThemeMode.DARK)
    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is cache_before, (
        "the hazard heatmap cache must not be theme-keyed -- these are "
        "semantic, theme-independent colors"
    )
    assert _pixel_rgb(canvas._discovered_hazard_pixmap_cache, 2, 2) == rgb_before


# ---------------------------------------------------------------------------
# 15. HISTORY without a historical belief frame hides the layer entirely.
# ---------------------------------------------------------------------------


def test_history_without_a_historical_belief_frame_hides_the_layer():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    # Enter HISTORY (a later phase will add a real historical hazard-belief
    # frame to NavigationDebugSnapshot; today there is none at all).
    canvas.navigation_debug_enabled = True
    canvas._nav_debug_history_position = (2, 10)
    canvas._nav_debug_snapshot = object()  # any non-None "selected" frame

    _draw_once(canvas)

    assert canvas._discovered_hazard_pixmap_cache is None, (
        "browsing history with no historical hazard-belief frame must hide "
        "the layer -- never fall back to the live frame or ground truth"
    )


# ---------------------------------------------------------------------------
# 16. Explicit ground-truth editing capability (locate/remove sources) is
# preserved -- editor_mode never touches fire, and toggle/removal still
# work through the same live canvas interaction as before.
# ---------------------------------------------------------------------------


def test_editor_mode_never_handles_fire_and_toggle_removal_still_works():
    canvas = _make_canvas()
    assert canvas.editor_mode is False  # default: fire interaction is live-canvas only

    source_text = inspect.getsource(SimulationCanvas.mousePressEvent)
    editor_branch = source_text.split("if self.editor_mode and not self.robot")[1].split("hit = self.robot_index_at_screen_position")[0]
    assert "fireToggleRequested" not in editor_branch, (
        "the editor_mode branch must never emit fireToggleRequested -- "
        "fire placement/removal stays a live-canvas-only interaction"
    )

    # The actual locate/remove capability is unaffected by Phase 4 (ground-
    # truth FireSource lookups never touched hazard_service.py here).
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    added = service.add_fire((2.5, 2.5)).source
    assert service.sources() == (added,)
    removed = service.remove_fire_near((2.5, 2.5))
    assert removed.source == added
    assert service.sources() == ()


# ---------------------------------------------------------------------------
# 17. Occupancy/explored/FoV state is unaffected by the new render path.
# ---------------------------------------------------------------------------


def test_occupancy_explored_and_fov_state_unaffected_by_discovered_hazard_render():
    canvas = _make_canvas()
    canvas.set_mapped_obstacle_points([(1.0, 1.0), (2.0, 2.0)])
    canvas.explored_area_polygons = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]
    mapped_before = list(canvas.mapped_obstacle_points)
    explored_before = list(canvas.explored_area_polygons)

    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    _draw_once(canvas)

    assert canvas.mapped_obstacle_points == mapped_before
    assert canvas.explored_area_polygons == explored_before


# ---------------------------------------------------------------------------
# 18. The legacy ground-truth path keeps working independently, with its own
# payload shape never mixed with the discovered-hazard one.
# ---------------------------------------------------------------------------


def test_legacy_hazard_snapshot_is_independent_of_the_discovered_hazard_frame():
    canvas = _make_canvas()

    canvas.set_hazard_snapshot(
        {"version": 1, "bounds": _BOUNDS, "resolution": _RESOLUTION, "grid": None, "sources": ()}
    )
    assert canvas._hazard_snapshot is not None
    assert canvas._discovered_hazard_frame is None  # setting one never sets the other

    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    assert canvas._discovered_hazard_frame is not None
    assert canvas._hazard_snapshot is not None  # still holds the earlier ground-truth push
