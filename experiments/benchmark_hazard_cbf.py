"""Runnable benchmark for the Observed Hazard OGM-HOCBF safety filter.

This is a BENCHMARK, not a pytest test: no pass/fail threshold, no
assertions -- it prints a table and exits. Run directly:

    python experiments/benchmark_hazard_cbf.py

Measures, for each (resolution, robot_count) combination, over the
simulator's real default world bounds (WORLD_X_MIN/MAX, WORLD_Y_MIN/MAX),
using the accepted PRODUCTION safety-filter configuration (pyramid_levels=1
-- see SimulationConfig.hazard_cbf_pyramid_levels and the multiscale audit
finding in test_hazard_hocbf_filter.py for why multi-level was rejected):

  - one real SDF (re)build (a fresh HazardSafetyRuntime per resolution, so
    the expensive rebuild happens exactly once per resolution, not once per
    robot_count -- this matches production, where one shared runtime serves
    every robot off the same team belief);
  - 100 filter_control() calls with that SDF reused (cycling through all of
    that row's robots, so "per robot" below is the mean cost of filtering
    ONE robot's control on a given tick);
  - how many belief cells crossed the block threshold;
  - approximate memory of the cached pyramid's per-level NumPy arrays.

Resolution guidance (post-audit decision, see the "profile" column below):

  - 0.50 m is the OFFICIAL resolution for demos and experiments -- fast
    rebuilds, representative blocked-cell counts.
  - 0.25 m is usable for OFFLINE analysis only (rebuild latency measured in
    hundreds of ms -- noticeable as a stall if it happens live, tolerable
    for a non-interactive/batch run).
  - 0.10 m is NOT VIABLE for interactive use with the current brute-force
    distance transform (robotics_sim/environment/hazard_distance_field.py):
    a single rebuild measured in the audit took ~22 SECONDS over the
    simulator's default world. This result is intentionally not omitted or
    softened -- it is the reason a proper O(N) distance transform (e.g.
    Felzenszwalt-Huttenlocher) is a prerequisite before offering resolutions
    finer than 0.25 m in any interactive/live configuration.
  - 1.00 m is fast but coarse -- useful for quick smoke checks, not
    representative of realistic hazard footprints.

No new dependencies are used (numpy + stdlib only).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotics_sim.core.limits import RobotLimits  # noqa: E402
from robotics_sim.core.state import RobotState  # noqa: E402
from robotics_sim.environment.grid_geometry import GridGeometry  # noqa: E402
from robotics_sim.environment.hazard_belief import HazardBelief  # noqa: E402
from robotics_sim.simulation.config import (  # noqa: E402
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
)
from robotics_sim.simulation.hazard_safety_runtime import HazardSafetyRuntime  # noqa: E402

_BOUNDS = (WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX)
_HAZARD_CENTER = (0.0, 0.0)
_HAZARD_RADIUS_M = 1.5
_BLOCK_THRESHOLD = 0.55
_REUSED_FILTER_CALLS = 100
_RESOLUTIONS = (1.0, 0.5, 0.25, 0.10)
_ROBOT_COUNTS = (1, 4, 8)
_LEVEL_ARRAY_NAMES = ("value", "gradient_x", "gradient_y", "hessian_xx", "hessian_xy", "hessian_yy")

# Post-audit decision: production uses pyramid_levels=1 (level 0) only.
_PRODUCTION_PYRAMID_LEVELS = 1

# Post-audit resolution classification -- see module docstring. Not derived
# from the measured build_ms below on purpose: this is a fixed, reviewed
# policy decision, not something that should silently reclassify itself if a
# future machine happens to measure a different number.
_RESOLUTION_PROFILE = {
    1.0: "fast",
    0.5: "recommended",
    0.25: "high rebuild latency",
    0.10: "unsupported for interactive use",
}


def _build_belief(resolution: float) -> tuple[HazardBelief, GridGeometry]:
    geometry = GridGeometry(_BOUNDS, resolution)
    belief = HazardBelief(geometry, robot_count=1)
    center_x, center_y = _HAZARD_CENTER
    rows: list[int] = []
    cols: list[int] = []
    for dy in np.arange(-_HAZARD_RADIUS_M, _HAZARD_RADIUS_M + 1e-9, resolution):
        for dx in np.arange(-_HAZARD_RADIUS_M, _HAZARD_RADIUS_M + 1e-9, resolution):
            if dx * dx + dy * dy > _HAZARD_RADIUS_M**2:
                continue
            cell = geometry.world_to_grid(center_x + dx, center_y + dy)
            if cell is not None:
                rows.append(cell.row)
                cols.append(cell.col)
    belief.observe_cells(rows=rows, cols=cols, values=[0.9] * len(rows), robot_index=0)
    return belief, geometry


def _robot_states(count: int) -> list[RobotState]:
    """`count` robots approaching the hazard from evenly spaced directions,
    each close enough to be within activation distance -- a representative
    "everyone near the hazard" load, not an idle scene."""
    states = []
    center_x, center_y = _HAZARD_CENTER
    approach_distance = 2.0
    for i in range(count):
        angle = 2.0 * np.pi * i / max(1, count)
        x = center_x + approach_distance * np.cos(angle)
        y = center_y + approach_distance * np.sin(angle)
        theta = angle + np.pi  # heading back toward the hazard center
        states.append(RobotState(x=x, y=y, theta=theta, v=1.5))
    return states


def _pyramid_array_bytes(frame) -> int:
    total = 0
    for level in frame.levels:
        for name in _LEVEL_ARRAY_NAMES:
            total += getattr(level, name).nbytes
    return total


def _run_resolution(resolution: float) -> list[dict]:
    belief, geometry = _build_belief(resolution)
    belief_frame = belief.snapshot()
    limits = RobotLimits(max_speed=2.0, max_acceleration=2.0, max_angular_speed=2.5)
    nominal = np.array([[0.0], [0.0]])

    max_robots = max(_ROBOT_COUNTS)
    all_states = _robot_states(max_robots)

    runtime = HazardSafetyRuntime(
        block_threshold=_BLOCK_THRESHOLD,
        margin=0.20,
        activation_distance=3.0,
        k1=2.0,
        k2=2.0,
        pyramid_levels=_PRODUCTION_PYRAMID_LEVELS,
        smoothing_sigma_cells=0.75,
        acceleration_weight=1.0,
        angular_weight=0.35,
    )

    # Exactly one real SDF rebuild for this resolution.
    runtime.filter_control(
        belief_frame=belief_frame,
        geometry=geometry,
        state=all_states[0],
        limits=limits,
        nominal_control=nominal,
        safety_radius=0.35,
    )
    build_ms = runtime.field_last_build_ms
    blocked_rows, _blocked_cols = belief.blocked_cells(_BLOCK_THRESHOLD)
    blocked_cells = int(blocked_rows.size)
    sdf_bytes = _pyramid_array_bytes(runtime.field_frame)

    rows = []
    for robot_count in _ROBOT_COUNTS:
        states = all_states[:robot_count]
        filter_start = time.perf_counter()
        for i in range(_REUSED_FILTER_CALLS):
            runtime.filter_control(
                belief_frame=belief_frame,
                geometry=geometry,
                state=states[i % robot_count],
                limits=limits,
                nominal_control=nominal,
                safety_radius=0.35,
            )
        total_reused_ms = (time.perf_counter() - filter_start) * 1000.0
        filter_ms_per_robot = total_reused_ms / _REUSED_FILTER_CALLS

        rows.append(
            {
                "resolution": resolution,
                "profile": _RESOLUTION_PROFILE.get(resolution, "unclassified"),
                "robot_count": robot_count,
                "blocked_cells": blocked_cells,
                "build_ms": build_ms,
                "filter_ms_per_robot": filter_ms_per_robot,
                "sdf_kb": sdf_bytes / 1024.0,
            }
        )
    return rows


def main() -> None:
    header = (
        f"{'res(m)':>7} {'profile':>32} {'robots':>7} {'blocked':>8} {'build_ms':>10} "
        f"{'filter_ms/robot':>16} {'sdf_KB':>8}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    # resolution=0.10 is deliberately NOT skipped or averaged away here --
    # it is the row that demonstrates why it is classified "unsupported for
    # interactive use" (see module docstring), and it takes noticeably
    # longer to reach than the other three; rows print as each finishes so
    # the run is not silent for ~20+ seconds.
    for resolution in _RESOLUTIONS:
        for row in _run_resolution(resolution):
            print(
                f"{row['resolution']:>7.2f} {row['profile']:>32} {row['robot_count']:>7d} "
                f"{row['blocked_cells']:>8d} {row['build_ms']:>10.3f} "
                f"{row['filter_ms_per_robot']:>16.4f} {row['sdf_kb']:>8.1f}"
            )


if __name__ == "__main__":
    main()
