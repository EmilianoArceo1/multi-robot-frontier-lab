# CQLite experiments (RA-L 2024)

This integration implements the algorithm described in *Communication-Efficient Multi-Robot Exploration Using Coverage-Biased Distributed Q-Learning* (Latif and Parasuraman, RA-L 2024, DOI `10.1109/LRA.2024.3358095`). The paper supplied with this project is the primary algorithm specification. The public repository is pinned at commit `8423b0563215bc29e3ccf6bad17d5ad2b3732f3d`.

## What is implemented

`algorithms/cqlite/plugin.py` maintains one Q table and explored-frontier set per robot. On each decision it:

1. obtains the current frontier actions from the simulator service;
2. updates each candidate with equation (1), using the coverage-biased reward in equation (7);
3. computes overlap probability from equations (5)--(6);
4. partitions actions by travel-time Voronoi ownership from equation (9);
5. selects the maximum Q/travel priority while enforcing team reservations;
6. sends only the newly explored state or selected `(state, Q)` update to robots inside communication range.

The plugin does not exchange full Q tables. The debug result reports neighbor edges, messages, compact-payload bytes, Q-table sizes, Q updates, reward, overlap, travel time, and whether the no-stall Voronoi fallback was used.

Published numeric parameters are pinned in every `.sim` preset: `alpha=0.6`, `gamma=0.95`, step cost `2` per metre, LiDAR range `15 m`, overlap radius `1 m`, nominal speed `0.5 m/s`, angular-speed limit `pi/4 rad/s`, and a `50 m` fully connected communication range inside the published `40--60 m` interval.

The paper defines `rho` and `sigma` but does not publish their experiment values. The native presets therefore label `rho=2.0` and `sigma=0.01` as host assumptions. They are stored under `coordination.parameters` and can be changed without editing code.

## Interactive experiments

Open one preset in the simulator and press **Start**:

- `examples/cqlite_house_3.sim` -- three robots, independent `20 x 12.5 m` house proxy (250 square metres).
- `examples/cqlite_bookstore_3.sim` -- three robots, independent `12.5 x 8 m` bookstore proxy (100 square metres).
- `examples/cqlite_bookstore_6.sim` -- six robots in the same bookstore proxy.

The **Metrics** panel contains the simulator's mapping time, distance, coverage, and multi-robot overlap. The navigation-reasoning robot selector can be used to inspect each robot's chosen frontier/path. CQLite decision telemetry is also retained in `last_coordination_debug` and emitted through the runtime metrics service.

Run each preset ten times to match the paper's trial count. Record mean and standard deviation; do not compare a single interactive run against a ten-trial paper mean.

## Automated native proxy

Run:

```powershell
python -m experiments.run_cqlite_experiments --trials 10
```

The output is written to `experiments/results/cqlite_native_proxy_summary.json`. This runner executes the real CQLite plugin but makes visits instantaneous over deterministic coverage waypoints. Its timing, distance, overlap, and compact communication payload are regression proxies, not Gazebo/ROS SLAM measurements.

## Fidelity boundary

The exact house and AWS bookstore Gazebo world files, random start poses, `rho`, `sigma`, map resolution, and baseline implementations were not published with the paper. The official repository's planner file also does not contain the distributed Q update shown in Algorithm 1; it ends in fixed goal commands and includes unfinished POMDP code. Therefore:

- the three geometries here are independently created area-matched proxies;
- A* is used for CQLite travel-cost queries and execution;
- the simulator's occupancy/frontier services replace `gmapping`, ROS map merge, `move_base`, and DWA;
- an ad-hoc merge request is represented as a HOLD/debug event because this simulator already exposes a shared team belief snapshot;
- the wire payload count measures compact algorithm fields, excluding ROS and transport overhead;
- published RRT/DRL/CQLite Table I values are reference data only and are never mixed with native output metrics;
- map SSIM, ROS CPU/RAM, and exact middleware payload cannot be reproduced by this native simulator.

These limitations make the experiment reproducible and auditable without claiming numerical equivalence to a world and software stack that are unavailable.
