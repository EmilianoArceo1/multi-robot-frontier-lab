"""Tests for TransitionAssemblyInput's cross-validation and
LearningTransitionAssembler.build()."""

from __future__ import annotations

import pytest

from robotics_interfaces.learning import (
    CandidateKind,
    CandidateSetSpec,
    CriticState,
    HoldPolicy,
    TerminationReason,
)
from robotics_interfaces.learning.transitions import RewardComponent
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import (
    DecisionSelectionBatch,
    RobotActionSelection,
    RobotRewardOutcome,
    TransitionAssemblyInput,
    TransitionOutcomeBatch,
)

NORMALIZATION = FeatureNormalizationConfig(
    distance_scale=10.0,
    information_gain_scale=5.0,
    travel_cost_scale=20.0,
    safety_cost_scale=2.0,
    overlap_cost_scale=4.0,
    heading_cost_scale=1.0,
    sensor_range_scale=8.0,
    safety_radius_scale=1.0,
    fire_window_radius_cells=1,
)


def make_geometry() -> GridGeometry:
    return GridGeometry(bounds=(0.0, 10.0, 0.0, 10.0), resolution=1.0)


def make_candidate_spec(max_candidates: int = 8) -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=max_candidates,
        max_headings_per_candidate=1,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_robot(robot_id: int = 0, xy=(1.0, 1.0)) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id, xy=xy, safety_radius=0.5, sensor_range=4.0,
        vision_model="cone", theta=0.0,
    )


def make_candidate(
    target=(4.0, 6.0), heading_rad=None, enabled=True, reachable=True
) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(
            target=target, source="frontier", information_gain=1.0, heading_rad=heading_rad
        ),
        kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=enabled, reachable=reachable,
    )


def build_decision(
    geometry, robots_spec, episode_id="ep-assembler", decision_step=0, time_s=0.0, candidate_spec=None
):
    """robots_spec: tuple of (robot_id, candidates) pairs."""
    candidate_spec = candidate_spec or make_candidate_spec()
    robot_captures = tuple(
        RobotActorCaptureInput(
            robot=make_robot(rid), candidates=candidates, graph_edges=(),
            visible_teammates=(), hazard_belief=HazardBelief(geometry).snapshot(),
        )
        for rid, candidates in robots_spec
    )
    frame = RuntimeActorFrame(
        episode_id=episode_id, decision_step=decision_step, time_s=time_s,
        robots=robot_captures, grid_geometry=geometry, normalization=NORMALIZATION,
        candidate_spec=candidate_spec,
    )
    assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=candidate_spec
        ),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return assembler.build(frame)


def make_critic_state(decision_step=0, time_s=0.0) -> CriticState:
    return CriticState(
        schema_version="0.1.0", decision_step=decision_step, time_s=time_s,
        global_feature_names=("coverage",), global_features=(0.5,),
        per_robot_feature_names=(), per_robot_features={},
    )


def make_reward_component(name="new_coverage", raw=1.0, weight=0.5) -> RewardComponent:
    return RewardComponent(name=name, raw_value=raw, applied_weight=weight, weighted_value=raw * weight)


def make_selections(episode_id, decision_step, robot_ids, action_index=0):
    return DecisionSelectionBatch(
        episode_id=episode_id, decision_step=decision_step,
        selections=tuple(
            RobotActionSelection(robot_id=rid, action_index=action_index, issued_at_step=decision_step)
            for rid in robot_ids
        ),
    )


def make_outcome(episode_id, decision_step, robot_ids, terminated=False, truncated=False, reason=None):
    if reason is None:
        reason = TerminationReason.RUNNING if not (terminated or truncated) else TerminationReason.MAX_STEPS
    return TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=decision_step,
        rewards=tuple(
            RobotRewardOutcome(robot_id=rid, components=(make_reward_component(),))
            for rid in robot_ids
        ),
        terminated=terminated, truncated=truncated, termination_reason=reason,
    )


class TestRobotCount:
    def test_single_robot(self):
        geometry = make_geometry()
        current = build_decision(geometry, ((0, (make_candidate(),)),))
        next_ = build_decision(geometry, ((0, (make_candidate(),)),), decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,)),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert set(transition.actor_observations) == {0}
        assert set(transition.selected_actions) == {0}

    def test_two_robots(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)), (1, (make_candidate(),)))
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0, 1)),
            outcome=make_outcome("ep-assembler", 0, (0, 1)),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert set(transition.actor_observations) == {0, 1}
        assert set(transition.next_actor_observations) == {0, 1}


class TestOrderPreserved:
    def test_robot_order_preserved(self):
        geometry = make_geometry()
        spec = ((5, (make_candidate(),)), (2, (make_candidate(),)))
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (5, 2)),
            outcome=make_outcome("ep-assembler", 0, (5, 2)),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert list(transition.actor_observations) == [5, 2]
        assert list(transition.selected_actions) == [5, 2]
        assert list(transition.next_actor_observations) == [5, 2]


class TestHeadingSelection:
    def test_action_with_heading(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(heading_rad=0.4),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,)),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.selected_actions[0].heading_index == 0

    def test_action_without_heading(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(heading_rad=None),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,)),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.selected_actions[0].heading_index is None


class TestSelectedActionResolution:
    def test_selected_action_nonexistent_fails(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,), action_index=99),
            outcome=make_outcome("ep-assembler", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        with pytest.raises(KeyError):
            LearningTransitionAssembler().build(build_input)

    def test_selected_action_disabled_fails(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(enabled=False, reachable=True),)),)
        current = build_decision(geometry, spec)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,), action_index=0),
            outcome=make_outcome("ep-assembler", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        with pytest.raises(ValueError):
            LearningTransitionAssembler().build(build_input)


class TestSelectionMismatch:
    def test_missing_selection_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)), (1, (make_candidate(),)))
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0,)),  # missing robot 1
                outcome=make_outcome("ep-assembler", 0, (0, 1)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_extra_selection_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0, 1)),  # robot 1 not in current
                outcome=make_outcome("ep-assembler", 0, (0,)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_duplicate_selection_rejected(self):
        with pytest.raises(ValueError):
            DecisionSelectionBatch(
                episode_id="ep-assembler", decision_step=0,
                selections=(
                    RobotActionSelection(robot_id=0, action_index=0, issued_at_step=0),
                    RobotActionSelection(robot_id=0, action_index=1, issued_at_step=0),
                ),
            )


class TestRewardMismatch:
    def test_missing_reward_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)), (1, (make_candidate(),)))
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0, 1)),
                outcome=make_outcome("ep-assembler", 0, (0,)),  # missing robot 1
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_extra_reward_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0,)),
                outcome=make_outcome("ep-assembler", 0, (0, 1)),  # robot 1 not in current
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_duplicate_reward_component_names_rejected(self):
        with pytest.raises(ValueError):
            RobotRewardOutcome(
                robot_id=0,
                components=(
                    make_reward_component(name="new_coverage"),
                    make_reward_component(name="new_coverage"),
                ),
            )

    def test_reward_total_calculated_correctly(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        outcome = TransitionOutcomeBatch(
            episode_id="ep-assembler", decision_step=0,
            rewards=(
                RobotRewardOutcome(
                    robot_id=0,
                    components=(
                        make_reward_component(name="new_coverage", raw=1.0, weight=0.5),
                        make_reward_component(name="path_cost", raw=-2.0, weight=0.1),
                    ),
                ),
            ),
            terminated=True, truncated=False, termination_reason=TerminationReason.MAX_STEPS,
        )
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=outcome,
            next_decision=None,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.reward_total_by_robot[0] == pytest.approx(0.5 + (-0.2))


class TestTerminalHandling:
    def test_non_terminal_with_next_decision_succeeds(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,), terminated=False, truncated=False),
            next_decision=next_,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.terminated is False
        assert transition.next_actor_observations

    def test_non_terminal_without_next_decision_fails(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0,)),
                outcome=make_outcome("ep-assembler", 0, (0,), terminated=False, truncated=False),
                next_decision=None,
                critic_state=make_critic_state(),
            )

    def test_terminal_without_next_decision_succeeds(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.terminated is True
        assert transition.next_actor_observations == {}

    def test_terminal_with_next_decision_fails(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0,)),
                outcome=make_outcome("ep-assembler", 0, (0,), terminated=True),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_truncated_without_next_decision_succeeds(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,), truncated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.truncated is True
        assert transition.next_actor_observations == {}

    def test_terminated_and_truncated_simultaneously_fails(self):
        with pytest.raises(ValueError):
            TransitionOutcomeBatch(
                episode_id="ep-assembler", decision_step=0,
                rewards=(),
                terminated=True, truncated=True,
                termination_reason=TerminationReason.MAX_STEPS,
            )


class TestMismatchDetection:
    def test_episode_id_mismatch(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec, episode_id="ep-A")
        next_ = build_decision(geometry, spec, episode_id="ep-A", decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-B", 0, (0,)),
                outcome=make_outcome("ep-A", 0, (0,)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_decision_step_mismatch(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec, decision_step=0)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 5, (0,)),
                outcome=make_outcome("ep-assembler", 0, (0,)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_next_step_equal_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec, decision_step=3)
        next_ = build_decision(geometry, spec, decision_step=3)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 3, (0,)),
                outcome=make_outcome("ep-assembler", 3, (0,)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_next_step_decreasing_rejected(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec, decision_step=3)
        next_ = build_decision(geometry, spec, decision_step=1)
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 3, (0,)),
                outcome=make_outcome("ep-assembler", 3, (0,)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )

    def test_next_decision_robot_order_mismatch_rejected(self):
        geometry = make_geometry()
        current = build_decision(
            geometry, ((0, (make_candidate(),)), (1, (make_candidate(),))), decision_step=0
        )
        next_ = build_decision(
            geometry, ((1, (make_candidate(),)), (0, (make_candidate(),))), decision_step=1
        )
        with pytest.raises(ValueError):
            TransitionAssemblyInput(
                current_decision=current,
                selections=make_selections("ep-assembler", 0, (0, 1)),
                outcome=make_outcome("ep-assembler", 0, (0, 1)),
                next_decision=next_,
                critic_state=make_critic_state(),
            )


class TestCriticState:
    def test_critic_state_passed_through_separately(self):
        geometry = make_geometry()
        spec = ((0, (make_candidate(),)),)
        current = build_decision(geometry, spec)
        critic_state = CriticState(
            schema_version="0.1.0", decision_step=0, time_s=0.0,
            global_feature_names=("coverage",), global_features=(0.75,),
            per_robot_feature_names=(), per_robot_features={},
        )
        build_input = TransitionAssemblyInput(
            current_decision=current,
            selections=make_selections("ep-assembler", 0, (0,)),
            outcome=make_outcome("ep-assembler", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=critic_state,
        )
        transition = LearningTransitionAssembler().build(build_input)
        assert transition.critic_state == critic_state
        assert transition.critic_state.global_features == (0.75,)
