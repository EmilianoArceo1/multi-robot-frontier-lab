from types import SimpleNamespace

import numpy as np

from robotics_sim.control.wang_ames_barrier_certificate import (
    SAFETY_ALGORITHM_OPTIONS,
    WANG_AMES_BARRIER_CERTIFICATE,
    filter_control,
)
from robotics_sim.simulation.config import SimulationConfig, config_from_sim_payload, config_to_sim_payload


def _robot(x, y, theta, v, *, acceleration=1.0, omega=2.0):
    return SimpleNamespace(
        x=x,
        y=y,
        theta=theta,
        v=v,
        limits=SimpleNamespace(max_acceleration=acceleration, max_angular_speed=omega),
    )


def test_selector_exposes_only_the_cited_certificate():
    assert SAFETY_ALGORITHM_OPTIONS == (WANG_AMES_BARRIER_CERTIFICATE,)


def test_nominal_control_passes_when_robots_are_separating():
    ego = _robot(0.0, 0.0, np.pi, 0.5)
    other = _robot(3.0, 0.0, 0.0, 0.5)
    nominal = np.array([[0.2], [0.1]])
    result = filter_control(ego=ego, others=(ego, other), nominal_control=nominal, safety_distance=0.7)
    assert result.feasible
    assert np.allclose(result.control, nominal)


def test_head_on_approach_is_modified_or_braked():
    ego = _robot(0.0, 0.0, 0.0, 1.0)
    other = _robot(0.8, 0.0, np.pi, 1.0)
    result = filter_control(
        ego=ego,
        others=(ego, other),
        nominal_control=np.array([[1.0], [0.0]]),
        safety_distance=0.7,
    )
    assert result.active
    assert result.control[0, 0] <= 0.0


def test_safety_algorithm_round_trips_in_sim_payload():
    config = SimulationConfig(safety_algorithm=WANG_AMES_BARRIER_CERTIFICATE)
    restored = config_from_sim_payload(config_to_sim_payload(config))
    assert restored.safety_algorithm == WANG_AMES_BARRIER_CERTIFICATE
