"""Pure host-side assembly: TransitionAssemblyInput -> LearningTransition.

``LearningTransitionAssembler`` never computes rewards (it only sums
already-computed ``RewardComponent.weighted_value``), never generates
candidates, never executes physical actions, and never re-derives the
validation that ``TransitionAssemblyInput`` already performed at
construction -- it trusts a validated input and delegates action
resolution entirely to ``DecisionCaptureBatch.resolve_action``, the real
catalog's public API.

``TransitionAssemblyInput.critic_state`` is required, matching
``LearningTransition.critic_state``'s own required ``CriticState`` field in
the current ObservationSpec contract -- no placeholder or derived
CriticState is ever constructed here.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

from robotics_interfaces.learning.observations import ActorObservation
from robotics_interfaces.learning.transitions import LearningTransition
from robotics_interfaces.learning.versioning import TRANSITION_SPEC_VERSION
from robotics_sim.learning.transition_inputs import TransitionAssemblyInput


class LearningTransitionAssembler:
    """Builds one LearningTransition from a validated TransitionAssemblyInput."""

    def build(self, build_input: TransitionAssemblyInput) -> LearningTransition:
        if not isinstance(build_input, TransitionAssemblyInput):
            raise TypeError(
                f"build_input must be a TransitionAssemblyInput, got "
                f"{type(build_input).__name__}"
            )

        current_decision = build_input.current_decision
        robot_order = tuple(o.robot_id for o in current_decision.actor_batch.observations)

        selections_by_robot = {s.robot_id: s for s in build_input.selections.selections}
        rewards_by_robot = {r.robot_id: r for r in build_input.outcome.rewards}

        actor_observations: dict[int, ActorObservation] = {}
        selected_actions = {}
        reward_components_by_robot = {}
        reward_total_by_robot: dict[int, float] = {}

        for robot_id in robot_order:
            actor_observations[robot_id] = current_decision.get_observation(robot_id)

            selection = selections_by_robot[robot_id]
            # Resolution -- and rejection of a nonexistent, disabled, or
            # unreachable action -- happens entirely through the catalog's
            # own public API; nothing here re-implements that check.
            selected_actions[robot_id] = current_decision.resolve_action(
                robot_id, selection.action_index, selection.issued_at_step
            )

            reward_outcome = rewards_by_robot[robot_id]
            reward_components_by_robot[robot_id] = reward_outcome.components
            reward_total_by_robot[robot_id] = sum(
                component.weighted_value for component in reward_outcome.components
            )

        is_terminal = build_input.outcome.terminated or build_input.outcome.truncated
        if is_terminal:
            next_actor_observations: dict[int, ActorObservation] = {}
        else:
            next_decision = build_input.next_decision
            next_actor_observations = {
                robot_id: next_decision.get_observation(robot_id) for robot_id in robot_order
            }

        return LearningTransition(
            schema_version=TRANSITION_SPEC_VERSION,
            episode_id=current_decision.actor_batch.episode_id,
            decision_step=current_decision.actor_batch.decision_step,
            actor_observations=actor_observations,
            critic_state=build_input.critic_state,
            selected_actions=selected_actions,
            reward_components_by_robot=reward_components_by_robot,
            reward_total_by_robot=reward_total_by_robot,
            next_actor_observations=next_actor_observations,
            terminated=build_input.outcome.terminated,
            truncated=build_input.outcome.truncated,
            termination_reason=build_input.outcome.termination_reason,
        )
