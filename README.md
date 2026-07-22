# Multi-Robot Frontier Lab

Research-oriented 2D simulator for multi-robot frontier exploration, FoV-aware planning, coordination strategies, and dynamic replanning.

Developed as part of a research internship at Texas A&M University.

## Current status

The current baseline focuses on an explainable 2D simulation for autonomous exploration in unknown or dynamic environments.

Implemented features include:

- occupancy-grid and belief-map based exploration
- field-of-view-aware frontier selection
- A* and direct path planning
- route simplification
- asynchronous prefetching of exploration targets
- dynamic obstacle sensing
- GUI visualization for robot movement, sensing, and mapped areas

Multi-robot coordination is under active development.

## Research direction

The project follows a modular multi-stage planning structure:

1. target generation
2. task allocation / coordination
3. motion planning
4. local execution and replanning

The next development stage will separate coordination algorithms from the simulator so that independent, reserved, auction-based, region-guided, and learning-based strategies can be compared under the same experimental framework.

## Project structure

```text
robotics_sim/
  app/
  assets/
  control/
  core/
  environment/
  models/
  navigation/
  planning/
  simulation/
  tests/
```

## Run
From the project root:

python main.py

The simulator can also be launched as a module:

python -m robotics_sim.main

## Reproducible experiments

- Nav2D autonomous exploration: load
  `examples/nav2d_tutorial3_single.sim` (single robot) or
  `examples/nav2d_tutorial4_multi.sim` (two-robot wavefront coordination).
  The pinned source, parameter mapping, algorithms, run protocol, and known
  platform differences are documented in
  `docs/experiments/nav2d_replication.md`.
- RSS 2026 uncertainty-guaranteed informative path planning: load
  `examples/rss26_ipp_rbf_smoke.sim` for the dependency-free integration check,
  or use `experiments/rss26_ipp/runner.py` for the pinned official Attentive-GP
  benchmark. The fidelity boundary, exact parameters, outputs, and run protocol
  are documented in `docs/experiments/rss26_uncertainty_guaranteed_ipp.md`.
- CQLite coverage-biased distributed Q-learning: load
  `examples/cqlite_house_3.sim`, `examples/cqlite_bookstore_3.sim`, or
  `examples/cqlite_bookstore_6.sim`. Run the ten-trial decision-level proxy
  with `python -m experiments.run_cqlite_experiments --trials 10`; fidelity,
  published parameters, and metric boundaries are documented in
  `docs/CQLITE_EXPERIMENTS.md`.

## Limitations
This is currently a 2D research simulator. It does not yet model real robot hardware, ROS 2 integration, SLAM uncertainty, communication loss, or full dynamic constraints.

The current goal is to build a stable, explainable baseline before adding more advanced architectures.
