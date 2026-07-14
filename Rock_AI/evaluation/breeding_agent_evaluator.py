"""Single-agent episode execution and deterministic replay."""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping

from Rock_AI.agents.breeding_agent_helper import (
    BreedingAgent,
    FarmerObjectiveProfile,
    NoAction,
    action_from_dict,
)
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.evaluation.breeding_agent_metrics import (
    calculate_farm_metrics,
    calculate_final_objective_utility,
)
from Rock_AI.logging.episode_record import EpisodeRecord


class BreedingAgentEvaluator:
    def __init__(self, environment_config: BreedingCampaignConfig | None = None):
        self.environment_config = environment_config or BreedingCampaignConfig()

    def run_episode(
        self,
        agent: BreedingAgent,
        *,
        seed: int,
        initial_farm: object | None = None,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
    ) -> EpisodeRecord:
        started = time.perf_counter()
        agent_seed = int(seed) + 1_000_003
        agent.reset(agent_seed)
        environment = BreedingCampaignEnvironment(
            seed=seed,
            config=self.environment_config,
            objective_profile=agent.objective_profile,
        )
        observation = environment.reset(
            seed,
            initial_farm=copy.deepcopy(initial_farm),
            rules=rules,
            objective_profile=agent.objective_profile,
        )
        environment.state.episode_id = f"{agent.name}-{seed}"
        while not environment.state.terminated:
            legal_actions = environment.legal_actions()
            try:
                action = agent.choose_action(observation, legal_actions)
            except Exception as error:  # The arena records agent failure without mutating a pair.
                environment.state.errors.append(f"Agent error: {error}")
                action = NoAction("agent_exception")
            environment.step(
                action,
                agent_name=agent.name,
                agent_seed=agent_seed,
                decision_context=agent.last_decision_context,
            )
            if not environment.state.terminated:
                observation = environment.observation()
        final_summary = calculate_farm_metrics(environment.game)
        final_summary.update(
            {
                "mutation_count": environment.state.mutation_count,
                "valid_decisions": environment.state.valid_decisions,
                "invalid_decisions_attempted": environment.state.invalid_decisions,
                "early_stop_count": environment.state.early_stop_count,
                "cumulative_pair_evaluator_utility": environment.state.cumulative_pair_utility,
                "objective_utility": calculate_final_objective_utility(
                    final_summary,
                    agent.objective_profile,
                    mutation_count=environment.state.mutation_count,
                ),
            }
        )
        breed_decisions = sum(
            decision.selected_action.get("action_type") == "breed_pair" and decision.error is None
            for decision in environment.state.decisions
        )
        return EpisodeRecord(
            episode_id=environment.state.episode_id,
            initial_seed=int(seed),
            agent_seed=agent_seed,
            agent_configuration=agent.configuration(),
            environment_configuration=self.environment_config.to_dict(),
            breeding_rules=environment.state.rules.to_dict(),
            initial_farm_summary=environment.state.initial_farm_summary,
            decisions=environment.state.decisions,
            final_farm_summary=final_summary,
            termination_reason=environment.state.termination_reason or "unknown",
            total_generations=environment.game.generation,
            total_breeding_decisions=int(breed_decisions),
            runtime_seconds=time.perf_counter() - started,
            errors=environment.state.errors,
            warnings=environment.state.warnings,
        )

    def replay_episode(
        self,
        record: EpisodeRecord,
        *,
        initial_farm: object | None = None,
    ) -> EpisodeRecord:
        config = BreedingCampaignConfig(**record.environment_configuration)
        objective_values = dict(record.agent_configuration["objective_profile"])
        objective = FarmerObjectiveProfile(**objective_values)
        environment = BreedingCampaignEnvironment(
            seed=record.initial_seed,
            config=config,
            objective_profile=objective,
        )
        environment.reset(
            record.initial_seed,
            initial_farm=copy.deepcopy(initial_farm),
            rules=record.breeding_rules,
            objective_profile=objective,
        )
        environment.state.episode_id = record.episode_id
        started = time.perf_counter()
        for decision in record.decisions:
            if environment.state.terminated:
                break
            environment.step(
                action_from_dict(decision.selected_action),
                agent_name=decision.agent_name,
                agent_seed=record.agent_seed,
                decision_context={
                    "ranked_candidate_pairs": decision.ranked_candidate_pairs,
                    "scores": decision.scores,
                    "predictor_outputs": decision.predictor_outputs,
                    "selected_score": decision.scores.get("neural_score")
                    or decision.scores.get("oracle_pair_utility")
                    or decision.scores.get("heuristic_score"),
                    "pair_evaluator_utility": decision.scores.get("oracle_pair_utility"),
                },
            )
        final_summary = calculate_farm_metrics(environment.game)
        final_summary.update(
            {
                "mutation_count": environment.state.mutation_count,
                "valid_decisions": environment.state.valid_decisions,
                "invalid_decisions_attempted": environment.state.invalid_decisions,
                "early_stop_count": environment.state.early_stop_count,
                "cumulative_pair_evaluator_utility": environment.state.cumulative_pair_utility,
                "objective_utility": calculate_final_objective_utility(
                    final_summary, objective, mutation_count=environment.state.mutation_count
                ),
            }
        )
        return EpisodeRecord(
            episode_id=record.episode_id,
            initial_seed=record.initial_seed,
            agent_seed=record.agent_seed,
            agent_configuration=record.agent_configuration,
            environment_configuration=record.environment_configuration,
            breeding_rules=record.breeding_rules,
            initial_farm_summary=environment.state.initial_farm_summary,
            decisions=environment.state.decisions,
            final_farm_summary=final_summary,
            termination_reason=environment.state.termination_reason or "unknown",
            total_generations=environment.game.generation,
            total_breeding_decisions=sum(
                decision.selected_action.get("action_type") == "breed_pair" and decision.error is None
                for decision in environment.state.decisions
            ),
            runtime_seconds=time.perf_counter() - started,
            errors=environment.state.errors,
            warnings=environment.state.warnings,
        )
