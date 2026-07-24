#!/usr/bin/env python
"""Run one reproducible headless smoke episode and print its result as JSON.

No GUI, no dataset export, no learning integration -- see
robotics_sim/simulation/headless_episode_runner.py for what this drives.

Example:
    python experiments/run_smoke_episode.py \\
        --map 01_open --scenario single_fire --seed 1 --max-time-s 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotics_sim.simulation.headless_episode_runner import (
    HeadlessEpisodeError,
    HeadlessSmokeEpisodeRunner,
    load_smoke_scenario,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/maps/smoke_v0/manifest.json",
        help="Path to the smoke-corpus manifest.json.",
    )
    parser.add_argument("--map", required=True, dest="map_id", help="Map id or .sim filename stem, e.g. 01_open.")
    parser.add_argument("--scenario", required=True, dest="scenario_id", help="Fire scenario id, e.g. single_fire.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-time-s", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        scenario = load_smoke_scenario(
            Path(args.manifest),
            map_id=args.map_id,
            scenario_id=args.scenario_id,
            seed=args.seed,
            max_time_s=args.max_time_s,
            max_steps=args.max_steps,
        )
        runner = HeadlessSmokeEpisodeRunner()
        result = runner.run(scenario, coverage_threshold=args.coverage_threshold)
    except HeadlessEpisodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- surfaced with context, never silently succeeds
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
