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
