"""
Navigation-mode helpers.

Do not use DEFAULT_EXPLORATION_PLANNER to decide whether a robot should go to
the GUI goal. The default planner can be an exploration planner.

Official rule:
    "Goal seeking" is the only mode where the GUI final goal G is executable.
    Every other exploration_planner value is an exploration mode.
"""

from __future__ import annotations

GOAL_SEEKING_PLANNER = "Goal seeking"


def normalize_planner_name(name: str | None) -> str:
    return str(name or "").strip()


def is_goal_seeking_planner(name: str | None) -> bool:
    return normalize_planner_name(name) == GOAL_SEEKING_PLANNER


def is_exploration_planner(name: str | None) -> bool:
    return not is_goal_seeking_planner(name)
