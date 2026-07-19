from Rock_AI.actions.action_encoder import ActionEncoder
from Rock_AI.actions.farmer_action import BreedPairAction
from Rock_AI.actions.action_schema import ActionObservationSchema
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment
from Rock_AI.neat.neat_recurrent_network import RecurrentEvaluationConfig
from Rock_AI.neat.neat_topology_helper import (
    RECURRENT_TOPOLOGY_VERSION, RecurrentConnectionGene, RecurrentNodeGene,
    RecurrentTopologyArtifact, TopologyResourceLimits,
)
from Rock_AI.policies.recurrent_neat_farmer_policy import RecurrentNeatFarmerPolicy
from Rock_AI.logging.public_world_event_record import PublicWorldEventRecord


def make_policy():
    schema = ActionObservationSchema()
    names = schema.feature_names + tuple(f"{name}.visible" for name in schema.feature_names)
    input_ids = tuple(-(index + 1) for index in range(len(names)))
    artifact = RecurrentTopologyArtifact(
        RECURRENT_TOPOLOGY_VERSION, "full-test", 1, 1, 1, "player", input_ids, (0,), names,
        ("action_score",), (RecurrentNodeGene(0, "output", 0.0, 1.0, "tanh", "sum"),),
        (RecurrentConnectionGene(input_ids[0], 0, 1.0, True),),
        RecurrentEvaluationConfig(1).to_dict(), TopologyResourceLimits().to_dict(),
        {"policy_kind": "full_farmer"},
    )
    return RecurrentNeatFarmerPolicy(artifact, checkpoint_id="test")


def test_candidate_comparison_does_not_mutate_memory_and_commit_happens_once():
    environment = MultiFarmEconomyEnvironment(seed=90)
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    policy = make_policy()
    observation = environment.observe(farm_id, policy.state)
    before = policy.export_state()
    decision = policy.rank_actions(observation)
    assert policy.export_state() == before
    assert decision.model_trace["candidate_evaluations_committed"] == 0
    result = environment.execute(decision.selected)
    policy.commit_selected(decision.selected, result)
    assert policy.state.decision_count == 1
    delayed = PublicWorldEventRecord("event", environment.world.turn, "bid_resolved", "A later public result", payload={"price": 4})
    policy.commit_visible_resolution(decision.selected, delayed)
    assert policy.state.decision_count == 2


def test_breed_encoding_and_hash_are_parent_order_symmetric():
    environment = MultiFarmEconomyEnvironment(seed=91)
    environment.reset()
    world = environment.world
    farm_id = sorted(world.farms)[0]
    farm = world.farm(farm_id)
    breed = next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, BreedPairAction) and not row.action.potion_keys)
    reversed_action = BreedPairAction(farm_id, world.turn, breed.action.parent_b_id, breed.action.parent_a_id, ())
    left = farm.get_rock(breed.action.parent_a_id)
    right = farm.get_rock(breed.action.parent_b_id)
    reversed_candidate = ActionEncoder().encode(reversed_action, actor=farm, world=world, objective=farm.profile, rock_a=right, rock_b=left)
    assert reversed_candidate.values == breed.values
    assert reversed_candidate.candidate_hash == breed.candidate_hash
