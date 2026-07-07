from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from robotics_interfaces.observations import Point2D

RobotCommandStatus = Literal["ASSIGNED", "HOLD", "BRAKE", "FAILED"]


@dataclass(frozen=True)
class RobotCommand:
    """Optional richer output a plugin may return alongside targets.

    A plugin that only owns TARGET_GENERATION/TASK_ALLOCATION only needs to
    set target (and optionally heading_rad). path and control_xy are for
    plugins that also own PATH_PLANNING or CONTROL (see
    robotics_interfaces.plugins.PluginRuntimeProfile) — the runtime should not
    expect them from a plain target allocator.
    """

    robot_id: int
    status: RobotCommandStatus
    target: Point2D | None = None
    heading_rad: float | None = None
    path: tuple[Point2D, ...] = ()
    control_xy: Point2D | None = None
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
