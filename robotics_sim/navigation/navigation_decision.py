"""
NavigationDecision — the value type returned by RobotAgent.step().

The engine reads the decision and translates it into robot control and planner
calls. No Qt, no rendering, no Robot physics knowledge inside this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NavigationDecisionKind = Literal[
    "FOLLOW_PATH",
    "BRAKE",
    "HOLD",
    "REQUEST_PLAN",
    "PREFETCH_NEXT_TARGET",
    "ACCEPT_PENDING_PATH",
    "REPLAN_FOR_SAFETY",
]


@dataclass(frozen=True)
class NavigationDecision:
    """
    Immutable decision packet produced by RobotAgent.step().

    Attributes:
        kind:         what the engine should do this frame.
        target:       relevant xy coordinate (goal, frontier, etc.).
        reason:       human-readable explanation for logging / metrics.
        brake:        True when the engine must apply braking this frame.
        replace_path: True when the active path should be swapped out.
    """

    kind: NavigationDecisionKind
    target: tuple[float, float] | None = None
    reason: str = ""
    brake: bool = False
    replace_path: bool = False
    force_new_target: bool = False


# ---------------------------------------------------------------------------
# Factory helpers — keep call sites readable and typo-free.
# ---------------------------------------------------------------------------

def follow(target: tuple[float, float], reason: str = "") -> NavigationDecision:
    """Robot should keep following its current active waypoint sequence."""
    return NavigationDecision(kind="FOLLOW_PATH", target=target, reason=reason)


def brake(reason: str = "") -> NavigationDecision:
    """Engine should apply braking control this frame."""
    return NavigationDecision(kind="BRAKE", reason=reason, brake=True)


def hold(reason: str = "") -> NavigationDecision:
    """Stop and stay at the current position. No fallback to G."""
    return NavigationDecision(kind="HOLD", reason=reason)


def request_plan(
    target: tuple[float, float],
    reason: str = "",
    *,
    do_brake: bool = False,
    force_new_target: bool = False,
) -> NavigationDecision:
    """Engine should compute a new path to *target*.

    force_new_target=True signals that the previous frontier was just
    reached and the planner must not use hysteresis to return it again.
    """
    return NavigationDecision(
        kind="REQUEST_PLAN",
        target=target,
        reason=reason,
        brake=do_brake,
        force_new_target=force_new_target,
    )


def prefetch_next_target(
    target: tuple[float, float],
    reason: str = "",
) -> NavigationDecision:
    """Engine should compute the next path *in the background* without stopping."""
    return NavigationDecision(kind="PREFETCH_NEXT_TARGET", target=target, reason=reason)


def accept_pending_path(reason: str = "") -> NavigationDecision:
    """Engine should replace the current path with the agent's pending_path."""
    return NavigationDecision(kind="ACCEPT_PENDING_PATH", reason=reason, replace_path=True)


def replan_for_safety(
    target: tuple[float, float] | None = None,
    reason: str = "",
) -> NavigationDecision:
    """Safety replan: engine replans immediately and brakes while computing."""
    return NavigationDecision(kind="REPLAN_FOR_SAFETY", target=target, reason=reason, brake=True)
