from Rock_AI.actions.farmer_action import ImportRandomRockAction, ImportRequestedRockAction
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_random_import_is_not_generated_until_action_commits():
    environment = MultiFarmEconomyEnvironment(seed=30)
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    before_ids = set(environment.world.owner_by_rock_id)
    candidate = next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, ImportRandomRockAction))
    assert set(environment.world.owner_by_rock_id) == before_ids
    result = environment.execute(candidate)
    assert result.success
    assert len(environment.world.owner_by_rock_id) == len(before_ids) + 1
    assert result.public_payload["revealed_rock_id"] not in before_ids


def test_requested_import_uses_public_quote_and_reveals_after_commit():
    environment = MultiFarmEconomyEnvironment(seed=31)
    environment.reset()
    farm_id = sorted(environment.world.farms)[0]
    candidate = next(row for row in environment.legal_candidates(farm_id) if isinstance(row.action, ImportRequestedRockAction))
    assert "revealed_rock_id" not in candidate.metadata
    result = environment.execute(candidate)
    assert result.success and result.public_payload["cost"] == candidate.action.quoted_cost
