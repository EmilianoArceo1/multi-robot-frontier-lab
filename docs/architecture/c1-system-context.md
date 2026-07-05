# C1 System Context — multi-robot-frontier-lab

## System of Interest

**multi-robot-frontier-lab** is a 2D multi-robot simulation host for evaluating interchangeable exploration, mapping, monitoring, coordination, planning, and control algorithms under a shared physical simulation environment.

The simulator acts as the experimental host. Algorithms are treated as interchangeable decision providers.

## People

### Researcher / Developer

Builds, modifies, tests, debugs, compares, and documents algorithms and simulation behavior.

### Operator / Demo User

Runs the GUI, observes the simulation, changes runtime parameters, and injects dynamic events such as fires or obstacles.

## External Systems

### GitHub Repository

Stores source code, branches, commits, pull requests, documentation, diagrams, tests, architecture decisions, scenario files, and experiment artifacts.

### Configuration and Scenario Files

Define maps, robot count, initial robot poses, sensor parameters, physical parameters, planner settings, algorithm selection, dynamic events, and experiment settings.

### Interchangeable Algorithm Plugins

Decision providers that may implement target generation, task allocation, path planning, local control, map updates, parameter patches, or full-stack coordination.

### Experiment Results, Metrics, and Logs

Generated artifacts used to compare algorithms, including coverage, exploration time, overlap, distance traveled, collisions, deadlocks, response time, computation time, and debug traces.

### Research Papers and Reference Repositories

External scientific and software references used to guide algorithm design and architecture decisions.

## Diagram

```mermaid
flowchart LR
    researcher["Researcher / Developer"]
    operator["Operator / Demo User"]

    github["GitHub Repository"]
    configs["Configuration and Scenario Files"]
    algorithms["Interchangeable Algorithm Plugins"]
    results["Experiment Results, Metrics, and Logs"]
    references["Research Papers and Reference Repositories"]

    system["multi-robot-frontier-lab<br/>2D multi-robot simulation host"]

    researcher -->|"develops, configures, runs, debugs, evaluates"| system
    operator -->|"runs GUI, observes robots, injects events"| system

    system -->|"loads reproducible scenarios and parameters"| configs
    configs -->|"provide maps, robots, sensors, events, algorithm selection"| system

    system <-->|"sends snapshots / receives decisions"| algorithms

    system -->|"generates metrics, logs, traces"| results
    researcher -->|"analyzes and compares"| results

    researcher <-->|"versions code, docs, diagrams, tests, configs, results"| github
    system -->|"codebase and architecture artifacts are versioned"| github

    researcher -->|"studies and extracts ideas"| references
    references -->|"inspire plugin implementations"| algorithms