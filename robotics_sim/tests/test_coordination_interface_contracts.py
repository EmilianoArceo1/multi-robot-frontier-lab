from __future__ import annotations

from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_interfaces.services import CoordinationServices, FrontierProvider


class FakeFrontierProvider:
    def candidates_for_robot(self, robot, world, blocked_targets=()):
        return (
            ExplorationCandidate(
                target=(robot.xy[0] + 1.0, robot.xy[1]),
                source="fake_provider",
                information_gain=5.0,
            ),
        )


def _robot(robot_id: int = 0) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


def test_request_can_carry_world_and_services_without_robotics_sim_dependency():
    world = WorldSnapshot(
        explored_points=((0.0, 0.0),),
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
    )
    services = CoordinationServices(frontier_provider=FakeFrontierProvider())

    request = CoordinationRequest(
        robot_states=(_robot(),),
        robots_to_assign=(0,),
        world=world,
        services=services,
    )

    assert request.world == world
    assert isinstance(request.services.frontier_provider, FrontierProvider)


def test_candidate_proposal_remains_backward_compatible():
    proposal = CandidateProposal(
        robot_id=7,
        target=(1.0, 2.0),
        score=9.5,
        information_gain=10.0,
        travel_cost=0.5,
        reason="test",
    )

    candidate = proposal.as_candidate()

    assert candidate.target == (1.0, 2.0)
    assert candidate.metadata["robot_id"] == 7
    assert candidate.metadata["score"] == 9.5
    assert candidate.utility == 9.5
