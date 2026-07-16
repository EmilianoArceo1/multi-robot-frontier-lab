"""
Pure unit tests for HazardBelief/HazardBeliefFrame/HazardBeliefUpdate
(robotics_sim.environment.hazard_belief) -- no engine, no HazardField, no
FoV rasterization. This is the team's discovered-only hazard layer: what
robots have actually observed, never the omniscient ground truth.

FoV -> row/col/value rasterization is a Phase 2 concern
(RuntimeHazardService.observe_visible_polygon()); observe_cells() here is
exercised directly with hand-picked cell coordinates.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import (
    HazardBelief,
    HazardBeliefFrame,
    HazardBeliefUpdate,
)

_BOUNDS = (0.0, 5.0, 0.0, 5.0)
_RESOLUTION = 1.0  # -> 5x5 grid, cell (row, col) centers at integers + 0.5


def _make_geometry(bounds: tuple[float, float, float, float] = _BOUNDS) -> GridGeometry:
    return GridGeometry(bounds, _RESOLUTION)


def _make_belief(robot_count: int = 1) -> HazardBelief:
    return HazardBelief(_make_geometry(), robot_count=robot_count)


# ---------------------------------------------------------------------------
# 1. Initial state is completely unobserved.
# ---------------------------------------------------------------------------


def test_initial_state_is_fully_unobserved():
    belief = _make_belief(robot_count=2)

    frame = belief.snapshot()

    assert frame.values.shape == (5, 5)
    assert frame.observed.shape == (5, 5)
    assert frame.observed_by_robot.shape == (2, 5, 5)
    assert not frame.observed.any()
    assert not frame.observed_by_robot.any()
    assert (frame.values == 0.0).all()
    assert frame.revision == 0


# ---------------------------------------------------------------------------
# 2. A visible safe cell becomes observed=True with value=0.
# ---------------------------------------------------------------------------


def test_visible_safe_cell_is_observed_with_zero_hazard():
    belief = _make_belief()

    belief.observe_cells([1], [1], [0.0], robot_index=0)
    frame = belief.snapshot()

    assert frame.observed[1, 1] == True  # noqa: E712
    assert frame.values[1, 1] == 0.0


# ---------------------------------------------------------------------------
# 3. A visible hot cell copies the ground-truth value as float32.
# ---------------------------------------------------------------------------


def test_visible_hot_cell_stores_float32_value():
    belief = _make_belief()

    belief.observe_cells([2], [3], [0.75], robot_index=0)
    frame = belief.snapshot()

    assert frame.values.dtype == np.float32
    assert frame.observed[2, 3] == True  # noqa: E712
    assert float(frame.values[2, 3]) == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Values below 0 or above 1 are clamped into [0, 1].
# ---------------------------------------------------------------------------


def test_values_outside_unit_range_are_clamped():
    belief = _make_belief()

    belief.observe_cells([0, 1], [0, 1], [-0.5, 5.0], robot_index=0)
    frame = belief.snapshot()

    assert frame.values[0, 0] == 0.0
    assert frame.values[1, 1] == 1.0
    assert float(frame.values.min()) >= 0.0
    assert float(frame.values.max()) <= 1.0


# ---------------------------------------------------------------------------
# 5. Cells not included in the call remain untouched.
# ---------------------------------------------------------------------------


def test_cells_outside_the_observation_are_not_modified():
    belief = _make_belief()

    belief.observe_cells([0], [0], [0.9], robot_index=0)
    frame = belief.snapshot()

    assert frame.observed[1, 1] == False  # noqa: E712
    assert frame.values[1, 1] == 0.0
    # Every other cell in the 5x5 grid besides (0, 0) stayed unobserved.
    assert int(frame.observed.sum()) == 1


# ---------------------------------------------------------------------------
# 6. Individual robot attribution is preserved.
# ---------------------------------------------------------------------------


def test_robot_attribution_is_preserved():
    belief = _make_belief(robot_count=2)

    belief.observe_cells([0], [0], [0.5], robot_index=0)
    frame = belief.snapshot()

    assert frame.observed_by_robot[0, 0, 0] == True  # noqa: E712
    assert frame.observed_by_robot[1, 0, 0] == False  # noqa: E712


# ---------------------------------------------------------------------------
# 7. Team belief fuses observations from multiple robots.
# ---------------------------------------------------------------------------


def test_team_belief_fuses_multiple_robots():
    belief = _make_belief(robot_count=2)

    belief.observe_cells([0], [0], [0.4], robot_index=0)
    belief.observe_cells([1], [1], [0.6], robot_index=1)
    frame = belief.snapshot()

    assert frame.observed[0, 0] and frame.observed[1, 1]
    assert float(frame.values[0, 0]) == pytest.approx(0.4, abs=1e-6)
    assert float(frame.values[1, 1]) == pytest.approx(0.6, abs=1e-6)
    assert frame.observed_by_robot[0, 0, 0] and not frame.observed_by_robot[1, 0, 0]
    assert frame.observed_by_robot[1, 1, 1] and not frame.observed_by_robot[0, 1, 1]


# ---------------------------------------------------------------------------
# 8-10. Revision changes only when real state changes.
# ---------------------------------------------------------------------------


def test_revision_changes_on_first_observation():
    belief = _make_belief()
    before = belief.revision

    update = belief.observe_cells([0], [0], [0.5], robot_index=0)

    assert belief.revision > before
    assert update.changed is True
    assert update.newly_observed_cells == 1


def test_revision_does_not_change_on_identical_repeat_observation():
    belief = _make_belief()
    belief.observe_cells([0], [0], [0.5], robot_index=0)
    revision_after_first = belief.revision

    update = belief.observe_cells([0], [0], [0.5], robot_index=0)

    assert belief.revision == revision_after_first
    assert update.changed is False
    assert update.newly_observed_cells == 0
    assert update.changed_value_cells == 0
    assert update.newly_attributed_cells == 0


def test_revision_changes_when_a_different_robot_attributes_the_same_cell():
    belief = _make_belief(robot_count=2)
    belief.observe_cells([0], [0], [0.5], robot_index=0)
    revision_before = belief.revision

    update = belief.observe_cells([0], [0], [0.5], robot_index=1)

    assert belief.revision > revision_before
    assert update.changed is True
    assert update.newly_observed_cells == 0
    assert update.changed_value_cells == 0
    assert update.newly_attributed_cells == 1


# ---------------------------------------------------------------------------
# 11-12. snapshot() returns immutable, independently-owned copies.
# ---------------------------------------------------------------------------


def test_snapshot_returns_immutable_copies_not_shared_memory():
    belief = _make_belief()
    belief.observe_cells([0], [0], [0.5], robot_index=0)

    frame = belief.snapshot()
    belief.observe_cells([0], [0], [0.9], robot_index=0)  # mutate belief after the snapshot

    assert float(frame.values[0, 0]) == pytest.approx(0.5, abs=1e-6), (
        "a live HazardBelief mutation must never retroactively change a frame already taken"
    )


def test_snapshot_arrays_are_not_writeable():
    belief = _make_belief(robot_count=2)
    belief.observe_cells([0], [0], [0.5], robot_index=0)

    frame = belief.snapshot()

    assert frame.values.flags.writeable is False
    assert frame.observed.flags.writeable is False
    assert frame.observed_by_robot.flags.writeable is False
    with pytest.raises(ValueError):
        frame.values[0, 0] = 1.0
    with pytest.raises(ValueError):
        frame.observed[0, 0] = True
    with pytest.raises(ValueError):
        frame.observed_by_robot[0, 0, 0] = True


# ---------------------------------------------------------------------------
# 13-14. restore() recreates the exact prior belief, content and revision.
# ---------------------------------------------------------------------------


def _populated_belief(robot_count: int = 2) -> HazardBelief:
    belief = _make_belief(robot_count=robot_count)
    belief.observe_cells([0, 2], [0, 3], [0.3, 0.8], robot_index=0)
    belief.observe_cells([2], [3], [0.8], robot_index=1)
    belief.observe_cells([4], [4], [0.0], robot_index=1)
    return belief


def test_restore_recreates_exact_hazard_belief_content():
    original = _populated_belief()
    frame = original.snapshot()

    restored = _make_belief(robot_count=2)
    restored.restore(frame)
    restored_frame = restored.snapshot()

    assert np.array_equal(restored_frame.values, frame.values)
    assert np.array_equal(restored_frame.observed, frame.observed)
    assert np.array_equal(restored_frame.observed_by_robot, frame.observed_by_robot)


def test_restore_recreates_exact_revision():
    original = _populated_belief()
    frame = original.snapshot()
    assert frame.revision > 0

    restored = _make_belief(robot_count=2)
    restored.restore(frame)

    assert restored.revision == frame.revision


# ---------------------------------------------------------------------------
# 15. restore() rejects incompatible shapes (grid geometry or robot_count).
# ---------------------------------------------------------------------------


def test_restore_rejects_incompatible_grid_shape():
    belief = _make_belief(robot_count=1)
    other = HazardBelief(_make_geometry((0.0, 3.0, 0.0, 3.0)), robot_count=1)  # 3x3, not 5x5
    other.observe_cells([0], [0], [0.5], robot_index=0)
    mismatched_frame = other.snapshot()

    with pytest.raises(ValueError):
        belief.restore(mismatched_frame)


def test_restore_rejects_incompatible_robot_count():
    belief = _make_belief(robot_count=1)
    other = _make_belief(robot_count=2)
    other.observe_cells([0], [0], [0.5], robot_index=0)
    mismatched_frame = other.snapshot()

    with pytest.raises(ValueError):
        belief.restore(mismatched_frame)


# ---------------------------------------------------------------------------
# 16-17. clear() wipes everything, and is a true no-op the second time.
# ---------------------------------------------------------------------------


def test_clear_removes_all_observed_state():
    belief = _populated_belief()

    belief.clear()
    frame = belief.snapshot()

    assert not frame.observed.any()
    assert not frame.observed_by_robot.any()
    assert (frame.values == 0.0).all()


def test_clear_twice_does_not_bump_revision_again():
    belief = _populated_belief()

    belief.clear()
    revision_after_first_clear = belief.revision
    belief.clear()

    assert belief.revision == revision_after_first_clear


# ---------------------------------------------------------------------------
# Extra contract checks from REQUISITOS not covered by the numbered list.
# ---------------------------------------------------------------------------


def test_robot_count_must_be_at_least_one():
    with pytest.raises(ValueError):
        HazardBelief(_make_geometry(), robot_count=0)


def test_invalid_robot_index_raises_value_error():
    belief = _make_belief(robot_count=2)

    with pytest.raises(ValueError):
        belief.observe_cells([0], [0], [0.5], robot_index=2)
    with pytest.raises(ValueError):
        belief.observe_cells([0], [0], [0.5], robot_index=-1)


def test_mismatched_row_col_value_shapes_raise_value_error():
    belief = _make_belief()

    with pytest.raises(ValueError):
        belief.observe_cells([0, 1], [0], [0.5, 0.5], robot_index=0)
    with pytest.raises(ValueError):
        belief.observe_cells([0], [0, 1], [0.5], robot_index=0)


def test_out_of_bounds_cell_indices_raise_value_error():
    belief = _make_belief()  # 5x5 grid: valid rows/cols are 0..4

    with pytest.raises(ValueError):
        belief.observe_cells([5], [0], [0.5], robot_index=0)
    with pytest.raises(ValueError):
        belief.observe_cells([0], [-1], [0.5], robot_index=0)


# ---------------------------------------------------------------------------
# Module hygiene: no Qt/engine/planner/HazardField/FireSource imports.
# ---------------------------------------------------------------------------


def test_module_imports_nothing_forbidden():
    import robotics_sim.environment.hazard_belief as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden_prefixes = (
        "PySide6",
        "PyQt5",
        "PyQt6",
        "qtpy",
        "robotics_sim.app",
        "robotics_sim.simulation",
        "robotics_sim.planning",
        "robotics_sim.environment.hazard_field",
    )
    offending = [
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    assert offending == [], f"hazard_belief.py imports forbidden modules: {offending}"


def test_dataclasses_are_frozen():
    import dataclasses

    assert dataclasses.fields(HazardBeliefFrame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        HazardBeliefFrame(
            values=np.zeros((1, 1), dtype=np.float32),
            observed=np.zeros((1, 1), dtype=bool),
            observed_by_robot=np.zeros((1, 1, 1), dtype=bool),
            revision=0,
        ).revision = 1

    with pytest.raises(dataclasses.FrozenInstanceError):
        HazardBeliefUpdate(
            changed=False, newly_observed_cells=0, changed_value_cells=0, newly_attributed_cells=0
        ).changed = True
