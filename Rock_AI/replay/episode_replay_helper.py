"""Reconstruct episodes once, then navigate with per-decision snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from Rock_AI.agents.breeding_agent_helper import FarmerObjectiveProfile, action_from_dict
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
)
from Rock_AI.environments.rock_training_environment import EnvironmentSnapshot
from Rock_AI.evaluation.breeding_agent_metrics import calculate_farm_metrics
from Rock_AI.logging.agent_decision_record import AgentDecisionRecord
from Rock_AI.logging.episode_record import EpisodeRecord
from Rock_AI.runtime.runtime_event_helper import RuntimeEvent, RuntimeEventType

from .replay_cursor_helper import ReplayCursor
from .replay_validation_helper import (
    ReplayValidationReport,
    compare_metric_summaries,
)


@dataclass(frozen=True)
class ReplayFrame:
    position: int
    decision_index: int | None
    selected_action: dict[str, Any] | None
    farm_summary: dict[str, float | int]
    snapshot: EnvironmentSnapshot


class EpisodeReplay:
    def __init__(
        self,
        environment: BreedingCampaignEnvironment,
        frames: list[ReplayFrame],
        validation: ReplayValidationReport,
    ):
        self.environment = environment
        self.frames = frames
        self.validation = validation
        self.cursor = ReplayCursor(len(frames))

    @classmethod
    def from_episode_record(
        cls,
        record: EpisodeRecord,
        *,
        initial_farm: object | None = None,
    ) -> "EpisodeReplay":
        objective = FarmerObjectiveProfile(**record.agent_configuration["objective_profile"])
        config = BreedingCampaignConfig(**record.environment_configuration)
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
        frames = [
            ReplayFrame(0, None, None, calculate_farm_metrics(environment.game), environment.snapshot())
        ]
        validation = ReplayValidationReport()
        for position, decision in enumerate(record.decisions, start=1):
            if environment.state.terminated:
                validation.add(decision.decision_index, "termination", "runnable", environment.state.termination_reason)
                break
            context = {
                "ranked_candidate_pairs": decision.ranked_candidate_pairs,
                "scores": decision.scores,
                "predictor_outputs": decision.predictor_outputs,
                "selected_score": decision.scores.get("neural_score")
                or decision.scores.get("oracle_pair_utility")
                or decision.scores.get("heuristic_score"),
                "pair_evaluator_utility": decision.scores.get("oracle_pair_utility"),
            }
            environment.step(
                action_from_dict(decision.selected_action),
                agent_name=decision.agent_name,
                agent_seed=record.agent_seed,
                decision_context=context,
            )
            summary = calculate_farm_metrics(environment.game)
            expected = (
                decision.immediate_post_action_farm_metrics
                or decision.post_action_farm_metrics
            )
            compare_metric_summaries(validation, decision.decision_index, expected, summary)
            frames.append(
                ReplayFrame(
                    position,
                    decision.decision_index,
                    decision.selected_action,
                    summary,
                    environment.snapshot(),
                )
            )
        final_actual = calculate_farm_metrics(environment.game)
        final_expected = {
            name: value
            for name, value in record.final_farm_summary.items()
            if name in final_actual
        }
        compare_metric_summaries(validation, len(record.decisions), final_expected, final_actual)
        return cls(environment, frames, validation)

    @classmethod
    def from_runtime_events(
        cls,
        events: Iterable[RuntimeEvent],
        *,
        seed: int,
        environment_configuration: dict[str, Any],
        breeding_rules: dict[str, Any],
        objective_profile: FarmerObjectiveProfile,
        initial_farm: object | None = None,
    ) -> "EpisodeReplay":
        all_events = list(events)
        decisions = [
            event for event in all_events if event.event_type == RuntimeEventType.DECISION_STARTED
        ]
        post_metrics_by_decision = {}
        for event in all_events:
            if event.post_action_metrics is not None:
                post_metrics_by_decision[event.decision_index] = event.post_action_metrics
        synthetic = EpisodeRecord(
            episode_id=f"event-replay-{seed}",
            initial_seed=seed,
            agent_seed=seed + 1_000_003,
            agent_configuration={
                "agent_id": "event-replay",
                "agent_type": "RecordedActions",
                "objective_profile": objective_profile.to_dict(),
            },
            environment_configuration=environment_configuration,
            breeding_rules=breeding_rules,
            initial_farm_summary={},
            decisions=[
                AgentDecisionRecord(
                    episode_id=f"event-replay-{seed}",
                    decision_index=index,
                    generation=event.generation,
                    agent_name="event-replay",
                    observation_summary={},
                    legal_action_count=int(event.payload.get("legal_action_count", 0)),
                    selected_action=event.payload["selected_action"],
                    selected_parent_ids=(
                        tuple(event.payload["selected_parent_ids"])
                        if event.payload.get("selected_parent_ids") else None
                    ),
                    immediate_post_action_farm_metrics=post_metrics_by_decision.get(
                        event.decision_index, {}
                    ),
                    post_action_farm_metrics=post_metrics_by_decision.get(
                        event.decision_index, {}
                    ),
                )
                for index, event in enumerate(decisions)
            ],
            final_farm_summary={},
            termination_reason="event_stream",
            total_generations=0,
            total_breeding_decisions=0,
            runtime_seconds=0.0,
        )
        return cls.from_episode_record(synthetic, initial_farm=initial_farm)

    @property
    def position(self) -> int:
        return self.cursor.position

    @property
    def current_frame(self) -> ReplayFrame:
        return self.frames[self.cursor.position]

    @property
    def current_game(self):
        return copy.deepcopy(self.current_frame.snapshot.state.game)

    def seek(self, position: int | str) -> ReplayFrame:
        self.cursor.seek(position)
        return self.current_frame

    def next(self) -> ReplayFrame:
        self.cursor.next()
        return self.current_frame

    def previous(self) -> ReplayFrame:
        self.cursor.previous()
        return self.current_frame

    def first(self) -> ReplayFrame:
        self.cursor.first()
        return self.current_frame

    def last(self) -> ReplayFrame:
        self.cursor.last()
        return self.current_frame
