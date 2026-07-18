"""Typed actions, observations, objectives, and common agent behavior."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile


@dataclass(frozen=True)
class BreedPairAction:
    parent_a_id: int | str
    parent_b_id: int | str

    def __post_init__(self) -> None:
        if self.parent_a_id == self.parent_b_id:
            raise ValueError("A BreedPairAction requires two different rock IDs")

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": "breed_pair", **asdict(self)}


@dataclass(frozen=True)
class StopGenerationAction:
    reason: str = "agent_finished_generation"

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": "stop_generation", **asdict(self)}


@dataclass(frozen=True)
class NoAction:
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": "no_action", **asdict(self)}


AgentAction: TypeAlias = BreedPairAction | StopGenerationAction | NoAction


def action_from_dict(data: dict[str, Any]) -> AgentAction:
    action_type = data.get("action_type")
    if action_type == "breed_pair":
        return BreedPairAction(data["parent_a_id"], data["parent_b_id"])
    if action_type == "stop_generation":
        return StopGenerationAction(data.get("reason", "agent_finished_generation"))
    if action_type == "no_action":
        return NoAction(data.get("reason", "no_action"))
    raise ValueError(f"Unknown action type: {action_type!r}")


@dataclass(frozen=True)
class CampaignObservation:
    farm: object
    generation: int
    remaining_breeding_actions: int
    legal_pair_ids: tuple[tuple[int | str, int | str], ...]
    breeding_rules: EncodedBreedingRules
    objective_profile: FarmerObjectiveProfile
    prior_decision_count: int
    queued_pair_ids: tuple[tuple[int | str, int | str], ...]
    farm_summary: dict[str, float | int]
    prior_actions: tuple[dict[str, Any], ...] = ()
    prior_decision_summaries: tuple[dict[str, Any], ...] = ()


DEFAULT_OBJECTIVE_PROFILES = {
    "balanced": FarmerObjectiveProfile(),
    "final_farm_value": FarmerObjectiveProfile(
        immediate_expected_value_weight=2.0,
        maximum_offspring_value_weight=0.25,
        survivor_count_weight=1.0,
        genotype_diversity_weight=0.25,
        phenotype_diversity_weight=0.25,
        rare_trait_weight=0.25,
    ),
    "maximum_rock_value": FarmerObjectiveProfile(
        immediate_expected_value_weight=0.25,
        maximum_offspring_value_weight=3.0,
        survivor_count_weight=0.1,
        genotype_diversity_weight=0.0,
        phenotype_diversity_weight=0.0,
        rare_trait_weight=0.5,
    ),
    "genotype_diversity": FarmerObjectiveProfile(
        immediate_expected_value_weight=0.1,
        maximum_offspring_value_weight=0.0,
        survivor_count_weight=0.25,
        genotype_diversity_weight=6.0,
        phenotype_diversity_weight=1.0,
        rare_trait_weight=1.0,
    ),
    "phenotype_diversity": FarmerObjectiveProfile(
        immediate_expected_value_weight=0.1,
        maximum_offspring_value_weight=0.0,
        survivor_count_weight=0.25,
        genotype_diversity_weight=1.0,
        phenotype_diversity_weight=6.0,
        rare_trait_weight=1.0,
    ),
    "rare_traits": FarmerObjectiveProfile(
        immediate_expected_value_weight=0.1,
        maximum_offspring_value_weight=0.25,
        survivor_count_weight=0.25,
        genotype_diversity_weight=1.0,
        phenotype_diversity_weight=1.0,
        rare_trait_weight=8.0,
    ),
}


def get_objective_profile(name: str) -> FarmerObjectiveProfile:
    try:
        return DEFAULT_OBJECTIVE_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown objective profile {name!r}") from error


class BreedingAgent(ABC):
    def __init__(self, agent_id: str, objective_profile: FarmerObjectiveProfile | None = None):
        self.agent_id = agent_id
        self.objective_profile = objective_profile or DEFAULT_OBJECTIVE_PROFILES["balanced"]
        self.seed = 0
        self.rng = random.Random(0)
        self.last_decision_context: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.agent_id

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.last_decision_context = {}

    @abstractmethod
    def choose_action(
        self,
        observation: CampaignObservation,
        legal_actions: tuple[AgentAction, ...],
    ) -> AgentAction:
        raise NotImplementedError

    def configuration(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": type(self).__name__,
            "objective_profile": self.objective_profile.to_dict(),
        }
