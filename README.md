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

## Run
From the project root:

python main.py

The simulator can also be launched as a module:

python -m robotics_sim.main

## Limitations
This is currently a 2D research simulator. It does not yet model real robot hardware, ROS 2 integration, SLAM uncertainty, communication loss, or full dynamic constraints.

The current goal is to build a stable, explainable baseline before adding more advanced architectures.
