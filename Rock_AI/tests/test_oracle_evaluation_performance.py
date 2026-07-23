from types import SimpleNamespace

import Rock_AI.agents.oracle_breeding_agent as oracle_module
from Rock_AI.agents.breeding_agent_helper import BreedPairAction
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.genetics_evaluator import GeneticsEvaluator
from Rock_GameState.rock_game_state_helper import GameMaster


def test_expression_results_are_reused_without_changing_exact_distributions():
    game = GameMaster(seed=630)
    parent_a = game.get_rock(1)
    parent_b = game.get_rock(2)
    evaluator = GeneticsEvaluator()

    first = evaluator.evaluate_all_genes(
        parent_a,
        parent_b,
        mutation_chance=0.01,
    )
    first_cache = evaluator.phenotype_cache_info()
    first_distributions = evaluator.distribution_cache_info()
    second = evaluator.evaluate_all_genes(
        parent_a,
        parent_b,
        mutation_chance=0.01,
    )
    second_cache = evaluator.phenotype_cache_info()
    second_distributions = evaluator.distribution_cache_info()

    assert first == second
    assert first_cache["misses"] > 0
    assert second_cache["misses"] == first_cache["misses"]
    assert second_cache["hits"] == first_cache["hits"]
    assert second_cache["size"] == first_cache["size"]
    assert first_distributions["misses"] == 1
    assert second_distributions["hits"] == 1
    assert second_distributions["size"] == 1


def test_oracle_reports_each_candidate_and_deterministic_selection(monkeypatch):
    game = GameMaster(seed=631)
    rules = EncodedBreedingRules()
    candidates = []
    rocks = sorted(game.rocks.values(), key=lambda rock: int(rock.id))
    for left, parent_a in enumerate(rocks):
        for parent_b in rocks[left + 1:]:
            if game.breeding_master.validate_breeding_pair(
                parent_a,
                parent_b,
                game=game,
                warn_relatedness=False,
            )["valid"]:
                candidates.append(BreedPairAction(parent_a.id, parent_b.id))
    assert len(candidates) >= 2

    estimate = SimpleNamespace(mean=1.0)

    class FastEvaluator:
        def evaluate_pair(self, parent_a, parent_b, **kwargs):
            return SimpleNamespace(
                parent_ids=(parent_a.id, parent_b.id),
                expectation=SimpleNamespace(
                    expected_child_value=estimate,
                    expected_survivor_count=estimate,
                    expected_maximum_child_value=estimate,
                ),
            )

    def fake_score(evaluation, objective_profile):
        score = float(sum(int(value) for value in evaluation.parent_ids))
        return SimpleNamespace(score=score, contributions={"test": score})

    monkeypatch.setattr(oracle_module, "score_pair_evaluation", fake_score)
    events = []
    agent = OracleBreedingAgent(
        evaluator=FastEvaluator(),
        trial_count=1,
        progress_callback=events.append,
    )
    agent.reset(900)
    observation = SimpleNamespace(
        farm=game,
        generation=0,
        remaining_breeding_actions=3,
        breeding_rules=rules,
    )

    selected = agent.choose_action(observation, tuple(candidates))

    candidate_events = [
        event for event in events if event["event"] == "candidate_completed"
    ]
    assert events[0]["event"] == "decision_started"
    assert events[-1]["event"] == "decision_completed"
    assert [event["completed"] for event in candidate_events] == list(
        range(1, len(candidates) + 1)
    )
    assert all(event["total"] == len(candidates) for event in events)
    assert events[-1]["decision_index"] == 0
    assert events[-1]["parent_ids"] == (
        selected.parent_a_id,
        selected.parent_b_id,
    )
