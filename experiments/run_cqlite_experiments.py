"""Run the three CQLite paper scenarios as deterministic native proxies.

This runner exercises the real ``algorithms.cqlite`` plugin for ten seeded,
decision-level episodes.  It is useful for regression/comparison and writes
machine-readable aggregate metrics.  It is not a Gazebo/ROS SLAM benchmark:
the interactive ``examples/cqlite_*.sim`` presets are the executable mapping
experiments, while this script intentionally replaces motion and sensing with
instantaneous visits to deterministic coverage waypoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Any, Iterable

from algorithms.cqlite.plugin import CQLITE_COORDINATOR, CQLitePlugin
from robotics_interfaces import CoordinationRequest, ExplorationCandidate, RobotCoordinationState


_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = (_ROOT / "experiments" / "results").resolve()
_PRESETS = (
    _ROOT / "examples" / "cqlite_house_3.sim",
    _ROOT / "examples" / "cqlite_bookstore_3.sim",
    _ROOT / "examples" / "cqlite_bookstore_6.sim",
)

# Table I is reference data transcribed from the supplied paper, never output
# under the native-results key and never used to grade the implementation.
PUBLISHED_TABLE_I: dict[str, dict[str, dict[str, list[float]]]] = {
    "house_3": {
        "RRT": {"mapping_time_s": [1208, 52], "path_length_m": [592, 11], "exploration_percent": [87, 4], "overlap_percent": [51, 5], "map_ssim": [0.73, 0.12], "cpu_percent": [112, 22], "ram_mb": [824, 19], "communication_mb": [2.2, 0.08]},
        "DRL": {"mapping_time_s": [924, 67], "path_length_m": [604, 19], "exploration_percent": [91, 3], "overlap_percent": [46, 6], "map_ssim": [0.89, 0.08], "cpu_percent": [79, 18], "ram_mb": [1264, 41], "communication_mb": [2.4, 0.06]},
        "CQLite": {"mapping_time_s": [1029, 59], "path_length_m": [543, 9], "exploration_percent": [95, 3], "overlap_percent": [28, 2], "map_ssim": [0.91, 0.06], "cpu_percent": [42, 8], "ram_mb": [665, 24], "communication_mb": [0.6, 0.02]},
    },
    "bookstore_3": {
        "RRT": {"mapping_time_s": [347, 32], "path_length_m": [278, 26], "exploration_percent": [90, 5], "overlap_percent": [57, 8], "map_ssim": [0.68, 0.21], "cpu_percent": [67, 18], "ram_mb": [624, 16], "communication_mb": [1.3, 0.06]},
        "DRL": {"mapping_time_s": [323, 21], "path_length_m": [235, 29], "exploration_percent": [93, 2], "overlap_percent": [51, 9], "map_ssim": [0.71, 0.13], "cpu_percent": [65, 15], "ram_mb": [819, 33], "communication_mb": [1.8, 0.04]},
        "CQLite": {"mapping_time_s": [317, 19], "path_length_m": [147, 21], "exploration_percent": [97, 2], "overlap_percent": [31, 6], "map_ssim": [0.89, 0.08], "cpu_percent": [34, 9], "ram_mb": [432, 21], "communication_mb": [0.4, 0.01]},
    },
    "bookstore_6": {
        "RRT": {"mapping_time_s": [212, 18], "path_length_m": [223, 12], "exploration_percent": [93, 4], "overlap_percent": [47, 7], "map_ssim": [0.71, 0.17], "cpu_percent": [68, 21], "ram_mb": [452, 19], "communication_mb": [1.1, 0.04]},
        "DRL": {"mapping_time_s": [265, 29], "path_length_m": [196, 17], "exploration_percent": [94, 5], "overlap_percent": [39, 8], "map_ssim": [0.73, 0.15], "cpu_percent": [47, 16], "ram_mb": [724, 38], "communication_mb": [1.3, 0.05]},
        "CQLite": {"mapping_time_s": [197, 13], "path_length_m": [121, 11], "exploration_percent": [98, 2], "overlap_percent": [21, 6], "map_ssim": [0.93, 0.10], "cpu_percent": [26, 9], "ram_mb": [319, 18], "communication_mb": [0.2, 0.01]},
    },
}


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _inside_obstacle(
    point: tuple[float, float], obstacles: Iterable[Iterable[float]], margin: float = 0.28
) -> bool:
    x, y = point
    for raw in obstacles:
        ox, oy, width, height = (float(value) for value in raw)
        if ox - margin <= x <= ox + width + margin and oy - margin <= y <= oy + height + margin:
            return True
    return False


def _coverage_waypoints(payload: dict[str, Any], rng: random.Random) -> list[tuple[float, float]]:
    camera = payload["camera"]
    x_min = float(camera["center_x"]) - float(camera["width"]) / 2.0
    x_max = float(camera["center_x"]) + float(camera["width"]) / 2.0
    y_min = float(camera["center_y"]) - float(camera["height"]) / 2.0
    y_max = float(camera["center_y"]) + float(camera["height"]) / 2.0
    obstacles = payload["map"]["obstacles"]
    spacing = 0.80
    points: list[tuple[float, float]] = []
    x = x_min + 0.55
    while x <= x_max - 0.55 + 1e-9:
        y = y_min + 0.55
        while y <= y_max - 0.55 + 1e-9:
            point = (round(x, 3), round(y, 3))
            if not _inside_obstacle(point, obstacles):
                points.append(point)
            y += spacing
        x += spacing
    rng.shuffle(points)
    area = float(camera["width"]) * float(camera["height"])
    limit = 64 if area > 150.0 else 40
    return points[: min(limit, len(points))]


def _scenario_key(path: Path) -> str:
    if "house" in path.stem:
        return "house_3"
    return "bookstore_6" if path.stem.endswith("_6") else "bookstore_3"


def run_trial(preset: Path, seed: int) -> dict[str, float | int]:
    payload = json.loads(preset.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    waypoints = _coverage_waypoints(payload, rng)
    remaining = list(waypoints)
    robots_data = payload["multi_robot"]["robots"]
    positions = [(float(item["x"]), float(item["y"])) for item in robots_data]
    headings = [float(item["theta"]) for item in robots_data]
    parameters = dict(payload["coordination"]["parameters"])
    parameters.update(
        {
            "grid_resolution": float(payload["map"]["grid_resolution"]),
            "cqlite_use_path_service": False,
            "min_frontier_travel_distance": 0.05,
            "target_exclusion_radius": 0.20,
            "reservation_resolution": 0.20,
        }
    )

    plugin = CQLitePlugin()
    paths = [0.0 for _ in positions]
    explored_by_robot: list[list[tuple[float, float]]] = [[] for _ in positions]
    overlap_visits = 0
    visits = 0
    last_debug: dict[str, Any] = {}
    iterations = 0

    while remaining and iterations < len(waypoints) + 4:
        states = tuple(
            RobotCoordinationState(
                robot_id=robot_id,
                xy=position,
                safety_radius=float(robots_data[robot_id]["safety_radius"]),
                sensor_range=float(robots_data[robot_id]["vision"]),
                vision_model=str(payload["sensor"]["type"]),
                theta=headings[robot_id],
                current_target=None,
                is_active=True,
            )
            for robot_id, position in enumerate(positions)
        )
        proposals = {
            robot_id: tuple(
                ExplorationCandidate(
                    target=point,
                    source="cqlite_native_decision_episode",
                    information_gain=1.0 + ((index * 37 + seed * 13) % 17) / 17.0,
                    metadata={"coverage_waypoint_index": index},
                )
                for index, point in enumerate(remaining)
            )
            for robot_id in range(len(states))
        }
        request = CoordinationRequest(
            robot_states=states,
            robots_to_assign=tuple(range(len(states))),
            proposals_by_robot=proposals,
            parameters=parameters,
            time_s=float(iterations),
        )
        result = plugin.assign(request)
        last_debug = dict(result.debug)
        selected: list[tuple[int, tuple[float, float]]] = []
        for assignment in result.assignments:
            if assignment.status == "ASSIGNED" and assignment.target is not None:
                selected.append((assignment.robot_id, assignment.target))
        if not selected:
            break

        for robot_id, target in selected:
            if any(
                _distance(target, prior) <= float(parameters["cqlite_overlap_radius"])
                for other_id, prior_points in enumerate(explored_by_robot)
                if other_id != robot_id
                for prior in prior_points
            ):
                overlap_visits += 1
            visits += 1
            paths[robot_id] += _distance(positions[robot_id], target)
            positions[robot_id] = target
            explored_by_robot[robot_id].append(target)
            remaining = [point for point in remaining if _distance(point, target) > 1e-9]
        iterations += 1

    speed = float(parameters["cqlite_nominal_speed"])
    communication = last_debug.get("communication", {})
    covered = len(waypoints) - len(remaining)
    return {
        "seed": seed,
        "decision_iterations": iterations,
        "mapping_time_proxy_s": max(paths, default=0.0) / max(speed, 1e-9),
        "total_path_length_m": sum(paths),
        "exploration_percent": 100.0 * covered / max(len(waypoints), 1),
        "overlap_visit_percent": 100.0 * overlap_visits / max(visits, 1),
        "communication_payload_kb": float(communication.get("payload_bytes_cumulative", 0)) / 1000.0,
        "communication_messages": int(communication.get("messages_cumulative", 0)),
        "map_merge_requests": int(communication.get("map_merge_requests_cumulative", 0)),
    }


def _aggregate(trials: list[dict[str, float | int]]) -> dict[str, Any]:
    metric_names = [key for key in trials[0] if key != "seed"]
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [float(trial[name]) for trial in trials]
        summary[name] = {"mean": mean(values), "std_population": pstdev(values)}
    return summary


def run_matrix(trial_count: int = 10, seed_base: int = 0) -> dict[str, Any]:
    if trial_count < 1:
        raise ValueError("trial_count must be at least 1")
    scenarios: dict[str, Any] = {}
    for preset in _PRESETS:
        key = _scenario_key(preset)
        trials = [run_trial(preset, seed_base + offset) for offset in range(trial_count)]
        scenarios[key] = {
            "preset": str(preset.relative_to(_ROOT)).replace("\\", "/"),
            "trial_count": trial_count,
            "aggregate": _aggregate(trials),
            "trials": trials,
        }
    return {
        "schema": "robotics_sim.cqlite_native_proxy.v1",
        "method": CQLITE_COORDINATOR,
        "fidelity": "decision_level_native_proxy_not_gazebo_slam",
        "warning": (
            "Native proxy metrics are not directly comparable with Table I: "
            "they omit SLAM, ROS middleware, image SSIM, CPU/RAM, and physical control time."
        ),
        "source": {
            "doi": "10.1109/LRA.2024.3358095",
            "repository": "https://github.com/herolab-uga/cqlite",
            "commit": "8423b0563215bc29e3ccf6bad17d5ad2b3732f3d",
        },
        "native_results": scenarios,
        "published_table_i_reference": PUBLISHED_TABLE_I,
    }


def _output_path(value: str) -> Path:
    path = Path(value).resolve()
    try:
        path.relative_to(_RESULTS)
    except ValueError as exc:
        raise ValueError(f"--output must be inside {_RESULTS}") from exc
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(_RESULTS / "cqlite_native_proxy_summary.json"),
    )
    args = parser.parse_args(argv)
    try:
        output = _output_path(args.output)
        report = run_matrix(trial_count=args.trials, seed_base=args.seed_base)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CQLite native proxy: {args.trials} trials x {len(_PRESETS)} scenarios")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
