# MARVEL adapter

Paper source: https://arxiv.org/abs/2502.20217

Place the authors' `checkpoint.pth` in `weights/`, or configure an absolute
path through `MARVEL_WEIGHTS_PATH`. Checkpoints are intentionally excluded
from Git.

Inference also requires a PyTorch installation compatible with the authors'
checkpoint. Torch is imported lazily, so the rest of the simulator and plugin
selector continue to work when it is not installed.

The adapter preserves the paper's declared assumptions: CTDE, decentralized
policy execution, perfect inter-robot communication, a shared occupancy map,
36 heading bins, 3 heading candidates, 6 node features, and 128-dimensional
embeddings. Missing or incompatible weights produce an explicit HOLD; no
heuristic is silently substituted for the cited policy.

`backend.py` converts the simulator's existing shared belief-map snapshot into
the graph observation used by the authors: a viewpoint lattice with 5x5
neighbour connectivity, free/unknown frontier extraction, relative position,
utility, guidepost, robot occupancy and informative-heading node features.
The selected waypoint and heading always come from the loaded PolicyNet logits.

Two task-assignment selectors share this implementation and checkpoint:

- `MARVEL CTDE graph-attention policy` is the paper-scale reproduction. It
  defaults to a 10 m sensor, 120 degree FoV, 4 m graph nodes, 0.8 m frontier
  voxels and a 60 m local map.
- `MARVEL CTDE graph-attention policy (scaled environment)` defaults to a 3 m
  sensor and 120 degree FoV. It multiplies every spatial graph length by
  `sensor_range / 10`, giving 1.2 m nodes, a 2.7 m utility range, approximately
  0.24 m frontier voxels and an 18 m local map at the default.

Both selectors reproduce the authors' panoramic known starting region before
the configured directional FoV is used. Range and FoV remain editable. The
scaled version preserves the dimensionless inputs seen by PolicyNet; it is an
explicit simulator adaptation, not a published lower-range benchmark.

Use `examples/marvel_original_scale.sim` to exercise the original selector
with the published sensor/graph parameters in a compact, explicitly labelled
smoke-test environment. It is not presented as the paper's unavailable
generated 90 x 90 m benchmark geometry.
