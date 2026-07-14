"""Analytical and simulation-backed genetics evaluation."""

from Rock_AI.evaluation.breeding_expectation_helper import BreedingExpectationEvaluator
from Rock_AI.evaluation.genetics_evaluator import GeneticsEvaluator
from Rock_AI.evaluation.pair_evaluator import PairEvaluation, PairEvaluator, PairUtilityWeights
from Rock_AI.evaluation.pair_ranker_metrics import calculate_pair_ranker_metrics

__all__ = [
    "BreedingExpectationEvaluator",
    "BreedingAgentEvaluator",
    "BreedingTournament",
    "GeneticsEvaluator",
    "PairEvaluation",
    "PairEvaluator",
    "PairUtilityWeights",
    "calculate_pair_ranker_metrics",
]


def __getattr__(name: str):
    if name == "BreedingAgentEvaluator":
        from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator

        return BreedingAgentEvaluator
    if name == "BreedingTournament":
        from Rock_AI.evaluation.breeding_tournament_helper import BreedingTournament

        return BreedingTournament
    raise AttributeError(name)
