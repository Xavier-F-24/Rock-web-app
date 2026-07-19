from Rock_AI.actions import ActionObservationSchema, ImportRandomRockAction
from Rock_AI.actions.action_encoder import ActionEncoder
from Rock_AI.actions.action_hash import canonical_action_hash
from Rock_AI.policies.market_action_policy_adapter import LegalFarmerActionGenerator
from Rock_World import create_starter_world


def test_action_schema_is_fixed_and_contains_no_hidden_genetics():
    schema = ActionObservationSchema()
    assert schema.feature_count > 50
    assert not any("genotype" in name or "death_gene" in name or "allele" in name for name in schema.feature_names)


def test_candidates_have_aligned_fixed_width_and_canonical_hashes():
    world = create_starter_world(seed=12)
    farm_id = sorted(world.farms)[0]
    rows = LegalFarmerActionGenerator().generate(world, farm_id)
    widths = {len(row.values) for row in rows}
    assert widths == {ActionObservationSchema().feature_count}
    assert len({row.candidate_hash for row in rows}) == len(rows)
    row = rows[0]
    assert row.candidate_hash == canonical_action_hash(
        row.action, observation_schema_version=1, action_schema_version=1,
        normalizer_version=1, public_rule_version=world.rule_version,
        objective_values=row.values[-6:], encoded_values=row.values, masks=row.visibility_mask,
    )
