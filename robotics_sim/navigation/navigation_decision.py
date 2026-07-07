"""
NavigationDecision — value types returned by RobotAgent.step().

The engine reads a NavigationDecision and translates it into robot control,
planner calls, path replacement, or future runtime parameter updates.

This file intentionally stays independent from Qt, canvas, engine internals,
robot physics implementation, and concrete planner implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


Point2D = tuple[float, float]


NavigationDecisionKind = Literal[
    # Current runtime decisions.
    "FOLLOW_PATH",
    "BRAKE",
    "HOLD",
    "REQUEST_PLAN",
    "PREFETCH_NEXT_TARGET",
    "ACCEPT_PENDING_PATH",
    "REPLAN_FOR_SAFETY",
    # Forward-compatible decision levels for algorithm-driven refactors.
    "SET_EXPLORATION_TARGET",
    "SET_PATH_PLAN",
    "APPLY_CONTROL_COMMAND",
    "REQUEST_PARAMETER_PATCH",
]


ParameterScope = Literal[
    "robot",
    "sensor",
    "planner",
    "mapping",
    "coordination",
    "communication",
    "simulation",
    "algorithm",
]


@dataclass(frozen=True)
class ControlCommand:
    """Low-level control request produced by a future control-level algorithm.

    The current engine does not consume this yet. It exists so that future
    policies such as APF/CBF/MPC/learned controllers can request control-level
    actions without importing or mutating the physical Robot object directly.
    """

    robot_id: int | None = None
    linear_velocity: float | None = None
    angular_velocity: float | None = None
    acceleration: float | None = None
    angular_acceleration: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ParameterPatch:
    """Request to modify a runtime parameter through the engine host.

    A policy should never mutate simulator attributes directly. It asks for a
    patch; the runtime validates and applies, clamps, or rejects it.

    Examples of parameter_path:
        robot.max_speed
        robot.safety_radius
        sensor.range_m
        sensor.fov_rad
        planner.target_exclusion_radius
        communication.bandwidth_kbps
        algorithm.mmpf.repulsion_gain
    """

    scope: ParameterScope
    parameter_path: str
    value: Any
    robot_id: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class NavigationDecision:
    """Immutable decision packet produced by RobotAgent.step().

    Current engine fields
    ---------------------
    kind:
        What the engine should do this frame.
    target:
        Relevant xy coordinate: goal, frontier, safety target, etc.
    reason:
        Human-readable explanation for logs and metrics.
    brake:
        True when the engine should brake this frame.
    replace_path:
        True when the active path should be swapped out.
    force_new_target:
        True when frontier hysteresis must not reuse the previous target.

    Forward-compatible fields
    -------------------------
    path:
        Explicit path plan from a future path-level algorithm.
    control:
        Explicit control command from a future control-level algorithm.
    parameter_patches:
        Runtime parameter changes requested by a future adaptive algorithm.
    metadata:
        Small debug/context payload for metrics and tracing. The engine should
        treat it as read-only and optional.
    """

    kind: NavigationDecisionKind
    target: Point2D | None = None
    reason: str = ""
    brake: bool = False
    replace_path: bool = False
    force_new_target: bool = False

    # Future algorithm decision payloads. These default to empty so existing
    # engine call sites keep working unchanged.
    path: tuple[Point2D, ...] = ()
    control: ControlCommand | None = None
    parameter_patches: tuple[ParameterPatch, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory helpers — keep call sites readable and typo-free.
# Existing helpers preserve the current runtime behavior.
# ---------------------------------------------------------------------------


def follow(target: Point2D, reason: str = "") -> NavigationDecision:
    """Robot should keep following its current active waypoint sequence."""
    return NavigationDecision(kind="FOLLOW_PATH", target=target, reason=reason)


def brake(reason: str = "") -> NavigationDecision:
    """Engine should apply braking control this frame."""
    return NavigationDecision(kind="BRAKE", reason=reason, brake=True)


def hold(reason: str = "") -> NavigationDecision:
    """Stop and stay at the current position. No fallback to G."""
    return NavigationDecision(kind="HOLD", reason=reason)


def request_plan(
    target: Point2D,
    reason: str = "",
    *,
    do_brake: bool = False,
    force_new_target: bool = False,
) -> NavigationDecision:
    """Engine should compute a new path to *target*.

    force_new_target=True signals that the previous frontier was just reached
    and the planner must not use hysteresis to return it again.
    """
    return NavigationDecision(
        kind="REQUEST_PLAN",
        target=target,
        reason=reason,
        brake=do_brake,
        force_new_target=force_new_target,
    )


def prefetch_next_target(
    target: Point2D,
    reason: str = "",
) -> NavigationDecision:
    """Engine should compute the next path in the background without stopping."""
    return NavigationDecision(kind="PREFETCH_NEXT_TARGET", target=target, reason=reason)


def accept_pending_path(reason: str = "") -> NavigationDecision:
    """Engine should replace the current path with the agent's pending_path."""
    return NavigationDecision(kind="ACCEPT_PENDING_PATH", reason=reason, replace_path=True)


def replan_for_safety(
    target: Point2D | None = None,
    reason: str = "",
) -> NavigationDecision:
    """Safety replan: engine replans immediately and brakes while computing."""
    return NavigationDecision(kind="REPLAN_FOR_SAFETY", target=target, reason=reason, brake=True)


# ---------------------------------------------------------------------------
# Forward-compatible helpers. The current engine may ignore these until the
# algorithm host is wired in. They are intentionally harmless now.
# ---------------------------------------------------------------------------


def set_exploration_target(
    target: Point2D,
    reason: str = "",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> NavigationDecision:
    """Future target-level algorithm selected a target but not a path."""
    return NavigationDecision(
        kind="SET_EXPLORATION_TARGET",
        target=target,
        reason=reason,
        metadata=metadata or {},
    )


def set_path_plan(
    path: tuple[Point2D, ...] | list[Point2D],
    reason: str = "",
    *,
    target: Point2D | None = None,
    replace_path: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> NavigationDecision:
    """Future path-level algorithm returned a full waypoint sequence."""
    normalized_path = tuple((float(x), float(y)) for x, y in path)
    inferred_target = target
    if inferred_target is None and normalized_path:
        inferred_target = normalized_path[-1]
    return NavigationDecision(
        kind="SET_PATH_PLAN",
        target=inferred_target,
        reason=reason,
        replace_path=replace_path,
        path=normalized_path,
        metadata=metadata or {},
    )


def apply_control_command(
    control: ControlCommand,
    reason: str = "",
    *,
    brake_if_missing: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> NavigationDecision:
    """Future control-level algorithm returned a direct control command."""
    return NavigationDecision(
        kind="APPLY_CONTROL_COMMAND",
        reason=reason or control.reason,
        brake=brake_if_missing,
        control=control,
        metadata=metadata or {},
    )


def request_parameter_patch(
    patches: tuple[ParameterPatch, ...] | list[ParameterPatch],
    reason: str = "",
    *,
    metadata: Mapping[str, Any] | None = None,
) -> NavigationDecision:
    """Future adaptive algorithm requests runtime parameter changes."""
    return NavigationDecision(
        kind="REQUEST_PARAMETER_PATCH",
        reason=reason,
        parameter_patches=tuple(patches),
        metadata=metadata or {},
    )
