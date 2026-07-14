"""Paired evaluation CLI for breeding-only agents."""

from __future__ import annotations

import argparse

from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_tournament_helper import (
    BreedingTournament,
    format_tournament_table,
)
from Rock_AI.scripts.run_breeding_agent_episode import create_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", required=True, choices=("random", "heuristic", "neural", "oracle"))
    parser.add_argument("--neural-checkpoint")
    parser.add_argument("--predictor-checkpoint")
    parser.add_argument("--objective", default="balanced")
    parser.add_argument("--oracle-trials", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-generations", type=int, default=7)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser


def run_from_args(args) -> dict:
    agents = [
        create_agent(
            name,
            objective_name=args.objective,
            neural_checkpoint=args.neural_checkpoint,
            predictor_checkpoint=args.predictor_checkpoint,
            oracle_trials=args.oracle_trials,
        )
        for name in args.agents
    ]
    tournament = BreedingTournament(
        BreedingCampaignConfig(
            max_decisions=args.max_decisions,
            max_generations=args.max_generations,
        )
    )
    records, summary = tournament.run(agents, episodes=args.episodes, seed=args.seed)
    tournament.save(args.output, records, summary)
    print(format_tournament_table(summary))
    print(f"output={args.output}")
    return summary


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
