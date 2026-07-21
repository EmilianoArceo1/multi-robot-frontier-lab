"""NumPy-only core for uncertainty-guaranteed informative path planning.

This package implements the discrete mathematical core described in
Jakkala et al., *Informative Path Planning with Guaranteed Estimation
Uncertainty* (arXiv:2602.05198): Gaussian-process posterior uncertainty,
Theorem-1 binary coverage maps, GreedyCover, and a budgeted GCBCover using
nearest insertion for routing costs.

Fidelity boundary
-----------------
The paper's experiments learn a non-stationary Attentive kernel and use the
planners distributed through SGP-Tools (``sgptools``).  :class:`RBFKernel`
is deliberately a small, stationary, NumPy-only smoke-test/MVP model.  It is
useful for validating the theorem and simulator plumbing, but it
must not be described as a faithful reproduction of the paper's learned Attentive model.
The public ``KernelProtocol`` keeps the planner core open to a future
Attentive/SGP-Tools adapter without importing simulator or GUI code here.
"""

from algorithms.uncertainty_guaranteed_ipp.certificate import (
    UncertaintyCertificate,
    certify_plan,
)
from algorithms.uncertainty_guaranteed_ipp.coverage import (
    BinaryCoverageMap,
    GreedyCoverResult,
    build_binary_coverage_matrix,
    greedy_cover,
)
from algorithms.uncertainty_guaranteed_ipp.gp import (
    GaussianProcessPosterior,
    KernelProtocol,
    RBFKernel,
    gp_posterior,
    posterior_variance,
)
from algorithms.uncertainty_guaranteed_ipp.routing import (
    GCBCoverResult,
    InsertionChoice,
    gcb_cover,
    nearest_insertion_increment,
    nearest_insertion_route,
    route_cost,
)

__all__ = [
    "BinaryCoverageMap",
    "GCBCoverResult",
    "GaussianProcessPosterior",
    "GreedyCoverResult",
    "InsertionChoice",
    "KernelProtocol",
    "RBFKernel",
    "UncertaintyCertificate",
    "build_binary_coverage_matrix",
    "certify_plan",
    "gcb_cover",
    "gp_posterior",
    "greedy_cover",
    "nearest_insertion_increment",
    "nearest_insertion_route",
    "posterior_variance",
    "route_cost",
]
