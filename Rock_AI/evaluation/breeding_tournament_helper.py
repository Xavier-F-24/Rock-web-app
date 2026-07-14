"""Fair, paired multi-agent breeding tournament runner."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import numpy as np

from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_AI.agents.breeding_agent_helper import BreedingAgent
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_agent_evaluator import BreedingAgentEvaluator
from Rock_AI.logging.episode_record import EpisodeRecord
from Rock_AI.logging.episode_storage_helper import save_episode_records


METRIC_NAMES = (
    "final_total_farm_value",
    "final_active_rock_value",
    "final_maximum_rock_value",
    "average_rock_value",
    "surviving_offspring",
    "mutation_count",
    "genotype_diversity",
    "phenotype_diversity",
    "rare_trait_count",
    "valid_decisions",
    "invalid_decisions_attempted",
    "early_stop_count",
    "cumulative_pair_evaluator_utility",
    "objective_utility",
)


def _describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(mean(values)) if values else 0.0,
        "standard_deviation": float(pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)) if values else 0.0,
        "first_quartile": float(np.quantile(array, 0.25)) if values else 0.0,
        "third_quartile": float(np.quantile(array, 0.75)) if values else 0.0,
    }


class BreedingTournament:
    def __init__(self, environment_config: BreedingCampaignConfig | None = None):
        self.environment_config = environment_config or BreedingCampaignConfig()
        self.evaluator = BreedingAgentEvaluator(self.environment_config)

    def run(
        self,
        agents: Iterable[BreedingAgent],
        *,
        episodes: int,
        seed: int,
        rules: EncodedBreedingRules | None = None,
    ) -> tuple[list[EpisodeRecord], dict]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        agents = list(agents)
        if not agents:
            raise ValueError("At least one agent is required")
        records: list[EpisodeRecord] = []
        by_episode: dict[int, list[EpisodeRecord]] = {}
        for episode_index in range(episodes):
            episode_seed = int(seed) + episode_index * 10_000
            initial_game = GameMaster(
                seed=episode_seed,
                max_generation=self.environment_config.max_generations,
                max_pairs_per_generation=self.environment_config.max_pairs_per_generation,
            )
            by_episode[episode_seed] = []
            for agent in agents:
                record = self.evaluator.run_episode(
                    agent,
                    seed=episode_seed,
                    initial_farm=copy.deepcopy(initial_game),
                    rules=rules,
                )
                records.append(record)
                by_episode[episode_seed].append(record)
        summary = self.summarize(records, by_episode)
        return records, summary

    @staticmethod
    def summarize(records: list[EpisodeRecord], by_episode: dict[int, list[EpisodeRecord]]) -> dict:
        grouped: dict[str, list[EpisodeRecord]] = {}
        for record in records:
            grouped.setdefault(record.agent_configuration["agent_id"], []).append(record)
        aggregate = {}
        for agent_name, agent_records in grouped.items():
            aggregate[agent_name] = {
                "episode_count": len(agent_records),
                "metrics": {
                    metric: _describe([float(record.final_farm_summary[metric]) for record in agent_records])
                    for metric in METRIC_NAMES
                },
                "runtime_seconds": _describe([record.runtime_seconds for record in agent_records]),
                "termination_reasons": dict(Counter(record.termination_reason for record in agent_records)),
            }
        wins = Counter()
        ties = 0
        for episode_records in by_episode.values():
            best = max(float(record.final_farm_summary["objective_utility"]) for record in episode_records)
            winners = [
                record.agent_configuration["agent_id"]
                for record in episode_records
                if abs(float(record.final_farm_summary["objective_utility"]) - best) <= 1e-9
            ]
            if len(winners) > 1:
                ties += 1
            for winner in winners:
                wins[winner] += 1 / len(winners)
        for agent_name in aggregate:
            aggregate[agent_name]["win_rate"] = wins[agent_name] / max(1, len(by_episode))
        paired = {}
        names = sorted(grouped)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                differences = []
                for episode_records in by_episode.values():
                    values = {
                        record.agent_configuration["agent_id"]: float(record.final_farm_summary["objective_utility"])
                        for record in episode_records
                    }
                    differences.append(values[left] - values[right])
                paired[f"{left}_minus_{right}"] = _describe(differences)
        return {
            "agents": aggregate,
            "paired_objective_utility_differences": paired,
            "episode_count": len(by_episode),
            "tie_episode_count": ties,
            "win_definition": "highest final objective_utility; ties split one win equally",
        }

    @staticmethod
    def save(output_directory: str | Path, records: list[EpisodeRecord], summary: dict) -> Path:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        save_episode_records(output / "episodes.jsonl", records)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return output


def format_tournament_table(summary: dict) -> str:
    header = f"{'agent':<14} {'episodes':>8} {'objective':>12} {'active$':>10} {'win%':>8} {'invalid':>9} {'sec/ep':>9}"
    lines = [header, "-" * len(header)]
    for name, data in sorted(summary["agents"].items()):
        metrics = data["metrics"]
        lines.append(
            f"{name:<14} {data['episode_count']:>8d} "
            f"{metrics['objective_utility']['mean']:>12.3f} "
            f"{metrics['final_active_rock_value']['mean']:>10.2f} "
            f"{data['win_rate'] * 100:>7.1f}% "
            f"{metrics['invalid_decisions_attempted']['mean']:>9.2f} "
            f"{data['runtime_seconds']['mean']:>9.3f}"
        )
    return "\n".join(lines)
