"""Run and persist one deterministic breeding-only agent campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from Rock_AI.agents.breeding_agent_helper import get_objective_profile
from Rock_AI.agents.heuristic_breeding_agent import HeuristicBreedingAgent
from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.logging.episode_storage_helper import save_episode_records
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy


def _predictor_candidate(ranker_checkpoint: str | Path) -> Path | None:
    ranker_path = Path(ranker_checkpoint)
    candidates = (
        ranker_path.parents[1] / "breeding_predictor_smoke" / "best.pt"
        if len(ranker_path.parents) > 1 else Path("missing"),
        Path("training_runs/breeding_predictor_smoke/best.pt"),
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def create_agent(
    name: str,
    *,
    objective_name: str = "balanced",
    neural_checkpoint: str | None = None,
    predictor_checkpoint: str | None = None,
    oracle_trials: int = 50,
):
    objective = get_objective_profile(objective_name)
    if name == "random":
        return RandomBreedingAgent(objective)
    if name == "heuristic":
        return HeuristicBreedingAgent(objective)
    if name == "oracle":
        return OracleBreedingAgent(objective, trial_count=oracle_trials)
    if name == "neural":
        if neural_checkpoint is None:
            raise ValueError("A neural ranker checkpoint is required for the neural agent")
        predictor = Path(predictor_checkpoint) if predictor_checkpoint else _predictor_candidate(neural_checkpoint)
        try:
            policy = NeuralPairRankingPolicy.load(
                neural_checkpoint,
                predictor_checkpoint=predictor,
            )
        except ValueError as error:
            if "requires a breeding-predictor" in str(error) and predictor is None:
                raise ValueError(
                    "This ranker requires --predictor-checkpoint; no compatible default was found"
                ) from error
            raise
        return NeuralBreedingAgent(policy, objective)
    raise ValueError(f"Unknown agent {name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=("random", "heuristic", "neural", "oracle"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--predictor-checkpoint")
    parser.add_argument("--objective", default="balanced")
    parser.add_argument("--oracle-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-generations", type=int, default=7)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = create_agent(
        args.agent,
        objective_name=args.objective,
        neural_checkpoint=args.checkpoint,
        predictor_checkpoint=args.predictor_checkpoint,
        oracle_trials=args.oracle_trials,
    )
    evaluator = BreedingAgentEvaluator(
        BreedingCampaignConfig(
            max_decisions=args.max_decisions,
            max_generations=args.max_generations,
        )
    )
    record = evaluator.run_episode(agent, seed=args.seed)
    save_episode_records(args.output, [record])
    print(
        f"agent={agent.name} termination={record.termination_reason} "
        f"generations={record.total_generations} breeds={record.total_breeding_decisions} "
        f"objective={record.final_farm_summary['objective_utility']:.3f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
