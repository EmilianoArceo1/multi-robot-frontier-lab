"""Runtime-level tests for the Observed Hazard OGM-HOCBF safety filter.

Covers:
  - robotics_sim.simulation.hazard_safety_runtime.HazardSafetyRuntime, using
    a real RuntimeHazardService (ground truth FireSource + observed-only
    HazardBelief) -- no engine, no Qt.
  - its wiring into robotics_sim.simulation.engine.SimulationControllerMixin
    (apply_hazard_safety_filter / ensure_hazard_safety_runtime), using the
    same duck-typed-fake-engine pattern already used by
    test_safety_hard_stop.py and test_multi_robot_route_validation.py, plus
    the same inspect.getsource() call-ordering proof already used by
    test_plugin_runtime_ownership.py's safety-veto-bypass test.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from robot import Robot
from robotics_sim.core.limits import RobotLimits
from robotics_sim.core.state import RobotState
from robotics_sim.simulation.config import SimulationConfig
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.hazard_safety_runtime import HazardSafetyRuntime
from robotics_sim.simulation.hazard_service import RuntimeHazardService

_BOUNDS = (0.0, 10.0, 0.0, 10.0)
_RESOLUTION = 0.5
_FIRE_POSITION = (8.5, 5.5)
# Bounding polygon big enough to cover the fire's footprint (default
# radius=2.0) around _FIRE_POSITION.
_OBSERVING_POLYGON = [(6.0, 3.0), (10.0, 3.0), (10.0, 8.0), (6.0, 8.0)]


def _service() -> RuntimeHazardService:
    return RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1, block_threshold=0.55)


def _runtime(**overrides) -> HazardSafetyRuntime:
    kwargs = dict(
        block_threshold=0.55,
        margin=0.20,
        activation_distance=3.0,
        k1=2.0,
        k2=2.0,
        pyramid_levels=1,
        smoothing_sigma_cells=0.75,
        acceleration_weight=1.0,
        angular_weight=0.35,
    )
    kwargs.update(overrides)
    return HazardSafetyRuntime(**kwargs)


def _limits() -> RobotLimits:
    return RobotLimits(max_acceleration=2.0, max_angular_speed=2.5)


def _approaching_state() -> RobotState:
    return RobotState(x=7.0, y=5.5, theta=0.0, v=2.0)  # heading straight at the fire, close enough to intervene


def _filter_once(service: RuntimeHazardService, runtime: HazardSafetyRuntime, *, state=None, control=None):
    return runtime.filter_control(
        belief_frame=service.belief.snapshot(),
        geometry=service.belief.geometry,
        state=state or _approaching_state(),
        limits=_limits(),
        nominal_control=np.array([[0.0], [0.0]]) if control is None else control,
        safety_radius=0.35,
    )


# ---------------------------------------------------------------------------
# 23. An unobserved FireSource must not activate the CBF.
# ---------------------------------------------------------------------------


def test_unobserved_fire_source_does_not_activate_filter():
    service = _service()
    service.add_fire(_FIRE_POSITION)  # ground truth only, never observed

    result = _filter_once(service, _runtime())

    assert result.active is False
    assert result.constraint_count == 0
    assert np.array_equal(result.control, np.array([[0.0], [0.0]]))


# ---------------------------------------------------------------------------
# 24. After observing the region, the field contains the hazard and the
#     filter can intervene.
# ---------------------------------------------------------------------------


def test_after_observing_region_field_contains_hazard_and_can_intervene():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    observation = service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)
    assert observation.newly_blocked_cells > 0

    result = _filter_once(service, _runtime())

    assert result.active is True
    assert result.constraint_count >= 1


# ---------------------------------------------------------------------------
# 25. A hazard now out of every robot's FoV, but observed before, keeps
#     influencing the filter (memory of out-of-view obstacles).
# ---------------------------------------------------------------------------


def test_hazard_out_of_fov_after_observation_still_influences_filter():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    # No further observe_visible_polygon() call -- simulates the hazard
    # cell no longer being inside anyone's sensor FoV this tick. The belief
    # (not any live sensor read) is what the filter consults.
    result = _filter_once(service, _runtime())

    assert result.active is True


# ---------------------------------------------------------------------------
# 26. Ground truth removed but not re-observed -> belief stays conservative.
# ---------------------------------------------------------------------------


def test_hazard_removed_from_ground_truth_without_reobservation_stays_conservative():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    removed = service.remove_fire_near(_FIRE_POSITION)
    assert removed.changed
    assert service.field.sources() == ()

    result = _filter_once(service, _runtime())

    assert result.active is True, "belief must stay conservative until the region is re-observed"


# ---------------------------------------------------------------------------
# 27. Re-observing a now-safe region clears the hazard from the field.
# ---------------------------------------------------------------------------


def test_reobserving_safe_region_clears_hazard_from_field():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)
    service.remove_fire_near(_FIRE_POSITION)

    reobservation = service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)
    assert reobservation.newly_unblocked_cells > 0

    result = _filter_once(service, _runtime())

    assert result.active is False


# ---------------------------------------------------------------------------
# 31. The SDF is not rebuilt every simulation tick when the mask is unchanged.
# ---------------------------------------------------------------------------


def test_field_is_not_rebuilt_every_tick_when_mask_unchanged():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    runtime = _runtime()
    for _ in range(5):
        _filter_once(service, runtime)

    assert runtime.field_rebuild_count == 1
    assert runtime.field_reuse_count == 4


# ---------------------------------------------------------------------------
# Performance instrumentation: field_*_build_ms only reflect real rebuilds;
# filter_*_ms reflect every filter_control() call; every metric stays
# non-negative.
# ---------------------------------------------------------------------------


def test_build_ms_metrics_only_reflect_real_rebuilds():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    runtime = _runtime()

    _filter_once(service, runtime)  # first call: a real rebuild
    assert runtime.field_rebuild_count == 1
    assert runtime.field_reuse_count == 0
    total_after_rebuild = runtime.field_total_build_ms
    assert runtime.field_last_build_ms >= 0.0
    assert total_after_rebuild >= 0.0
    assert runtime.field_max_build_ms >= runtime.field_last_build_ms

    for _ in range(4):  # same mask every time -> pure reuse, no rebuild
        _filter_once(service, runtime)

    assert runtime.field_rebuild_count == 1
    assert runtime.field_reuse_count == 4
    # Reuse calls must not add to the build-time accumulators at all.
    assert runtime.field_total_build_ms == total_after_rebuild
    assert runtime.field_max_build_ms == total_after_rebuild  # only one rebuild ever happened


def test_filter_ms_metrics_are_recorded_every_call_and_non_negative():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    runtime = _runtime()

    for _ in range(5):
        _filter_once(service, runtime)

    assert runtime.filter_last_ms >= 0.0
    assert runtime.filter_total_ms >= 0.0
    assert runtime.filter_max_ms >= 0.0
    # The running max must always be >= the most recent call's own time.
    assert runtime.filter_max_ms >= runtime.filter_last_ms
    # Accumulating across 5 calls, not overwriting: total >= max of any one call.
    assert runtime.filter_total_ms >= runtime.filter_max_ms


def test_all_runtime_metrics_are_non_negative():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    runtime = _runtime()
    for state in (_approaching_state(), RobotState(x=0.5, y=0.5, theta=0.0, v=0.0), _approaching_state()):
        _filter_once(service, runtime, state=state)

    for attribute in (
        "activation_count",
        "intervention_count",
        "infeasible_count",
        "invalid_initial_condition_count",
        "field_rebuild_count",
        "field_reuse_count",
        "maximum_intervention_norm",
        "field_last_build_ms",
        "field_total_build_ms",
        "field_max_build_ms",
        "filter_last_ms",
        "filter_total_ms",
        "filter_max_ms",
    ):
        assert getattr(runtime, attribute) >= 0.0, attribute


# ---------------------------------------------------------------------------
# 32. Metrics are counted exactly once per filter_control() call.
# ---------------------------------------------------------------------------


def test_metrics_are_counted_once_per_call():
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    runtime = _runtime()

    _filter_once(service, runtime, state=_approaching_state())
    assert runtime.activation_count == 1
    assert runtime.intervention_count <= 1
    assert runtime.field_rebuild_count == 1

    far_state = RobotState(x=0.5, y=0.5, theta=0.0, v=0.0)
    _filter_once(service, runtime, state=far_state)
    assert runtime.activation_count == 1, "far away and outside activation distance: no new activation"
    assert runtime.field_reuse_count == 1


# ---------------------------------------------------------------------------
# 28. hazard_cbf_enabled=False -> the previous control is preserved exactly.
# ---------------------------------------------------------------------------


def test_disabled_flag_preserves_control_exactly():
    fake = SimpleNamespace(config=SimpleNamespace(hazard_cbf_enabled=False))
    fake.apply_hazard_safety_filter = SimulationControllerMixin.apply_hazard_safety_filter.__get__(fake)

    control = np.array([[0.4], [0.1]])
    result = fake.apply_hazard_safety_filter(object(), control)

    assert result is control


# ---------------------------------------------------------------------------
# 29. Single-robot engine wiring: apply_hazard_safety_filter runs after the
#     nominal control and before predicted_motion_report(); robot.update()
#     never accidentally receives the un-filtered nominal control.
# ---------------------------------------------------------------------------


def test_single_robot_wiring_runs_filter_before_safety_veto():
    source = inspect.getsource(SimulationControllerMixin.simulation_step)
    legacy_marker = source.index("Legacy fallback")
    new_flow_source = source[:legacy_marker]
    legacy_source = source[legacy_marker:]

    for segment in (new_flow_source, legacy_source):
        nominal_call = segment.index("nominal_control_safe(")
        filter_assignment = "self.last_control = self.apply_hazard_safety_filter(self.robot, self.last_control)"
        filter_call = segment.index(filter_assignment, nominal_call)
        veto_call = segment.index("predicted_motion_report(", filter_call)
        update_call = segment.index("self.robot.update(self.last_control, dt)", veto_call)
        assert nominal_call < filter_call < veto_call < update_call


def test_engine_hazard_safety_hook_is_passthrough_for_aerial_robots():
    """Observed fire is information, so it never changes aerial control."""
    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    robot = Robot(x=7.0, y=5.5, theta=0.0, v=2.0, max_acceleration=2.0, max_angular_speed=2.5)

    fake = SimpleNamespace(
        config=SimpleNamespace(
            hazard_cbf_enabled=True,
            hazard_cbf_margin=0.20,
            hazard_cbf_activation_distance=3.0,
            hazard_cbf_k1=2.0,
            hazard_cbf_k2=2.0,
            hazard_cbf_pyramid_levels=1,
            hazard_cbf_sdf_smoothing_sigma_cells=0.75,
            hazard_cbf_acceleration_weight=1.0,
            hazard_cbf_angular_weight=0.35,
            hazard_block_threshold=0.55,
        ),
        hazard_service=service,
    )
    fake.safety_radius_for_robot = lambda robot=None: 0.35
    fake.ensure_hazard_safety_runtime = SimulationControllerMixin.ensure_hazard_safety_runtime.__get__(fake)
    fake.apply_hazard_safety_filter = SimulationControllerMixin.apply_hazard_safety_filter.__get__(fake)

    nominal = np.array([[0.0], [0.0]])
    filtered = fake.apply_hazard_safety_filter(robot, nominal)

    assert filtered is nominal


# ---------------------------------------------------------------------------
# 30. Multi-robot engine wiring: select_runtime_control_source() runs first,
#     the HOCBF filter runs after it, predicted_motion_report() (the safety
#     veto) runs after the filter -- a CONTROL-owning plugin cannot skip it.
# ---------------------------------------------------------------------------


def test_multi_robot_wiring_runs_filter_after_control_source_selection():
    source = inspect.getsource(SimulationControllerMixin.simulation_step_multi)

    control_source_call = source.index("select_runtime_control_source(")
    filter_assignment = "control = self.apply_hazard_safety_filter(robot, control)"
    filter_call = source.index(filter_assignment, control_source_call)
    veto_call = source.index("predicted_motion_report(", filter_call)
    update_call = source.index("robot.update(control, dt)", veto_call)

    assert control_source_call < filter_call < veto_call < update_call


# ---------------------------------------------------------------------------
# Production posture (post-audit decision): the real, unmodified engine
# wiring (ensure_hazard_safety_runtime()), fed a REAL, unmodified
# SimulationConfig(), must build the runtime with exactly one pyramid
# level. Level 0 alone is the accepted production configuration -- imposing
# every pyramid level as simultaneous hard constraints was rejected for
# conservatism/spurious infeasibility (see the multiscale audit finding
# tests in test_hazard_hocbf_filter.py, kept as evidence for why the
# default is 1, not removed).
# ---------------------------------------------------------------------------


def test_engine_wiring_with_real_config_defaults_builds_single_pyramid_level():
    config = SimulationConfig()
    assert config.hazard_cbf_pyramid_levels == 1, "production default must be a single (level 0) pyramid level"

    fake = SimpleNamespace(config=config)
    fake.ensure_hazard_safety_runtime = SimulationControllerMixin.ensure_hazard_safety_runtime.__get__(fake)

    runtime = fake.ensure_hazard_safety_runtime()
    assert runtime.pyramid_levels == 1

    service = _service()
    service.add_fire(_FIRE_POSITION)
    service.observe_visible_polygon(_OBSERVING_POLYGON, robot_index=0)

    result = runtime.filter_control(
        belief_frame=service.belief.snapshot(),
        geometry=service.belief.geometry,
        state=_approaching_state(),
        limits=_limits(),
        nominal_control=np.array([[0.0], [0.0]]),
        safety_radius=0.35,
    )

    assert result.active is True
    assert runtime.field_frame is not None
    assert len(runtime.field_frame.levels) == 1


def test_engine_wiring_falls_back_to_single_level_when_config_lacks_pyramid_levels_attribute():
    """A config object with no ``hazard_cbf_pyramid_levels`` attribute at all
    (e.g. an old/duck-typed config predating this field) must fall back to
    a single pyramid level -- matching the production default, not the
    rejected multi-level experimental posture."""
    config = SimpleNamespace()  # deliberately missing hazard_cbf_pyramid_levels
    assert not hasattr(config, "hazard_cbf_pyramid_levels")

    fake = SimpleNamespace(config=config)
    fake.ensure_hazard_safety_runtime = SimulationControllerMixin.ensure_hazard_safety_runtime.__get__(fake)

    runtime = fake.ensure_hazard_safety_runtime()

    assert runtime.pyramid_levels == 1
