import copy

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.environments.multi_farm_economy_environment import MultiFarmEconomyEnvironment


def test_private_opponent_money_and_hidden_genotype_do_not_change_observation():
    environment = MultiFarmEconomyEnvironment(seed=22)
    environment.reset()
    actor_id, opponent_id = sorted(environment.world.farms)[:2]
    baseline = environment.observe(actor_id)
    modified = copy.deepcopy(environment.world)
    opponent = modified.farm(opponent_id)
    opponent.money += 9999
    rock = next(iter(opponent.rocks.values()))
    pair = next(iter(rock.genotype.genes.values()))
    pair.allele_a = genetics.Allele(pair.allele_a.value + 1000)
    environment.world = modified
    changed = environment.observe(actor_id)
    assert changed.economy.observation_hash == baseline.economy.observation_hash
    assert tuple(row.candidate_hash for row in changed.legal_candidates) == tuple(row.candidate_hash for row in baseline.legal_candidates)


def test_each_farm_receives_a_separate_private_observation():
    environment = MultiFarmEconomyEnvironment(seed=23)
    environment.reset()
    rows = [environment.observe(farm_id) for farm_id in sorted(environment.world.farms)]
    assert len({row.economy.actor_farm_id for row in rows}) == 3
    assert len({row.economy.observation_hash for row in rows}) == 3
    assert all(not hasattr(opponent, "money") for row in rows for opponent in row.economy.opponents)
