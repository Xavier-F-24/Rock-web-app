"""Authoritative, deterministic control loop for breeding-only campaigns."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_AI.agents.breeding_agent_helper import (
    AgentAction,
    BreedPairAction,
    CampaignObservation,
    NoAction,
    StopGenerationAction,
)
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.environments.breeding_training_environment import BreedingTrainingEnvironment
from Rock_AI.environments.rock_training_environment import RockTrainingEnvironment
from Rock_AI.evaluation.breeding_agent_metrics import calculate_farm_metrics
from Rock_AI.logging.agent_decision_record import AgentDecisionRecord


@dataclass(frozen=True)
class BreedingCampaignConfig:
    max_decisions: int = 100
    max_generations: int = 7
    max_pairs_per_generation: int = 3
    invalid_action_policy: str = "terminate"

    def __post_init__(self) -> None:
        if self.max_decisions <= 0 or self.max_generations <= 0 or self.max_pairs_per_generation <= 0:
            raise ValueError("Campaign limits must be positive")
        if self.invalid_action_policy not in {"terminate", "stop_generation"}:
            raise ValueError("invalid_action_policy must be 'terminate' or 'stop_generation'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BreedingCampaignState:
    game: GameMaster
    rules: EncodedBreedingRules
    objective_profile: FarmerObjectiveProfile
    episode_id: str
    initial_farm_summary: dict[str, float | int]
    decisions: list[AgentDecisionRecord] = field(default_factory=list)
    pending_decision_by_pair: dict[tuple[str, str], int] = field(default_factory=dict)
    decision_count: int = 0
    valid_decisions: int = 0
    invalid_decisions: int = 0
    early_stop_count: int = 0
    mutation_count: int = 0
    cumulative_pair_utility: float = 0.0
    terminated: bool = False
    termination_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CampaignStepResult:
    action: AgentAction
    valid: bool
    children: tuple[genetics.Rock, ...]
    generation_advanced: bool
    terminated: bool
    termination_reason: str | None
    error: str | None = None


def _pair_key(parent_a_id: int | str, parent_b_id: int | str) -> tuple[str, str]:
    return tuple(sorted((str(parent_a_id), str(parent_b_id))))


class BreedingCampaignEnvironment(RockTrainingEnvironment):
    def __init__(
        self,
        seed: int = 0,
        *,
        config: BreedingCampaignConfig | None = None,
        objective_profile: FarmerObjectiveProfile | None = None,
    ):
        super().__init__(seed)
        self.config = config or BreedingCampaignConfig()
        self.default_objective = objective_profile or FarmerObjectiveProfile()

    def _make_game(self, seed: int, initial_farm: object | None) -> GameMaster:
        if initial_farm is None:
            game = GameMaster(
                seed=seed,
                max_generation=self.config.max_generations,
                max_pairs_per_generation=self.config.max_pairs_per_generation,
            )
        elif isinstance(initial_farm, GameMaster):
            game = copy.deepcopy(initial_farm)
            game.max_generation = self.config.max_generations
            game.max_pairs_per_generation = self.config.max_pairs_per_generation
            game.breeding_queue.clear()
            game.game_over = game.generation >= game.max_generation
        else:
            source = getattr(initial_farm, "rocks", initial_farm)
            values = source.values() if isinstance(source, Mapping) else source
            rocks = copy.deepcopy(list(values))
            game = GameMaster(
                seed=seed,
                auto_start=False,
                max_generation=self.config.max_generations,
                max_pairs_per_generation=self.config.max_pairs_per_generation,
            )
            game.rock_list = {int(rock.id): rock for rock in rocks}
            game.next_rock_id = max(game.rock_list, default=0) + 1
        game.seed = seed
        game.rng.seed(seed)
        game.genome_factory.rng.seed(seed + 1)
        game.name_generator.rng.seed(seed + 2)
        game.breeding_master.rng.seed(seed + 3)
        game.market_pods.clear()
        game.pending_market_pod = None
        return game

    @staticmethod
    def _validate_rules(rules: EncodedBreedingRules) -> None:
        if not rules.require_opposite_gender:
            raise ValueError("The authoritative breeding engine requires opposite-sex parents")
        if rules.spore_chance is not None or rules.mitosion_chance is not None:
            raise ValueError("Spore and mitosion activation are genotype-controlled")

    def reset(
        self,
        seed: int | None = None,
        initial_farm: object | None = None,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        objective_profile: FarmerObjectiveProfile | None = None,
    ) -> CampaignObservation:
        super().reset(seed)
        game = self._make_game(self.seed, initial_farm)
        encoded_rules = EncodedBreedingRules.from_config(rules, master=game.breeding_master)
        self._validate_rules(encoded_rules)
        objective = objective_profile or self.default_objective
        summary = calculate_farm_metrics(game)
        self.state = BreedingCampaignState(
            game=game,
            rules=encoded_rules,
            objective_profile=objective,
            episode_id=f"campaign-{self.seed}",
            initial_farm_summary=summary,
        )
        self._check_terminal_state()
        return self.observation()

    @property
    def game(self) -> GameMaster:
        if self.state is None:
            raise RuntimeError("Campaign environment must be reset before use")
        return self.state.game

    def _legal_pairs(self) -> tuple[tuple[int | str, int | str], ...]:
        if self.state is None or self.state.terminated:
            return ()
        queued_ids = {
            rock_id
            for pair in self.game.breeding_queue
            for rock_id in (pair.parent_a_id, pair.parent_b_id)
        }
        if len(self.game.breeding_queue) >= self.game.max_pairs_per_generation:
            return ()
        rocks = sorted(self.game.rocks.values(), key=lambda rock: int(rock.id))
        validator = self.game.breeding_master
        pairs = []
        for left in range(len(rocks)):
            for right in range(left + 1, len(rocks)):
                parent_a, parent_b = rocks[left], rocks[right]
                if parent_a.id in queued_ids or parent_b.id in queued_ids:
                    continue
                result = validator.validate_breeding_pair(
                    parent_a, parent_b, game=self.game, warn_relatedness=False
                )
                if result["valid"]:
                    pairs.append((parent_a.id, parent_b.id))
        return tuple(pairs)

    def legal_actions(self) -> tuple[AgentAction, ...]:
        if self.state.terminated:
            return ()
        breed_actions = tuple(BreedPairAction(left, right) for left, right in self._legal_pairs())
        return (*breed_actions, StopGenerationAction())

    def observation(self) -> CampaignObservation:
        pairs = self._legal_pairs()
        return CampaignObservation(
            farm=self.game,
            generation=self.game.generation,
            remaining_breeding_actions=max(
                0, self.game.max_pairs_per_generation - len(self.game.breeding_queue)
            ),
            legal_pair_ids=pairs,
            breeding_rules=self.state.rules,
            objective_profile=self.state.objective_profile,
            prior_decision_count=self.state.decision_count,
            queued_pair_ids=tuple(
                (pair.parent_a_id, pair.parent_b_id) for pair in self.game.breeding_queue
            ),
            farm_summary=calculate_farm_metrics(self.game),
            prior_actions=tuple(
                decision.selected_action for decision in self.state.decisions[-10:]
            ),
        )

    def _configure_breeding_master(self) -> None:
        master = self.game.breeding_master
        rules = self.state.rules
        master.child_gene_mutation_chance = rules.mutation_chance
        master.child_death_chance = rules.child_death_chance
        master.craisen_death_chance = rules.craisen_chance
        master.clutch_mean = rules.clutch_mean
        master.clutch_std = rules.clutch_std
        master.max_clutch_size = rules.max_clutch_size
        master.spore_death_chance = rules.spore_death_chance
        master.spore_clone_count = rules.spore_clone_count

    def _flush_generation(self) -> list[genetics.Rock]:
        self._configure_breeding_master()
        rules = self.state.rules
        had_override = "potion_settings" in self.game.__dict__
        previous_override = self.game.__dict__.get("potion_settings")
        self.game.potion_settings = lambda _keys: {
            "mutation_chance": rules.mutation_chance,
            "death_chance": rules.child_death_chance,
            "craisen_chance": rules.craisen_chance,
            "spore_death_chance": rules.spore_death_chance,
            "spore_clone_count": rules.spore_clone_count,
            "clutch_reroll": rules.clutch_reroll,
            "clutch_plus_one": rules.clutch_plus_one,
        }
        try:
            children = self.game.breed_queue()
        finally:
            if had_override:
                self.game.potion_settings = previous_override
            else:
                del self.game.__dict__["potion_settings"]
        # Campaign Stop is an explicit generation skip; core GameMaster otherwise
        # advances only when a non-empty clutch exists.
        self.game.generation += 1
        self.game.game_over = self.game.generation >= self.game.max_generation
        self.game.events.append(f"Campaign advanced to generation {self.game.generation}.")
        self._attach_children_to_decisions(children)
        return children

    def _attach_children_to_decisions(self, children: list[genetics.Rock]) -> None:
        grouped: dict[tuple[str, str], list[genetics.Rock]] = {}
        for child in children:
            if len(child.parent_ids) >= 2:
                grouped.setdefault(_pair_key(child.parent_ids[0], child.parent_ids[1]), []).append(child)
        for pair_key, pair_children in grouped.items():
            decision_index = self.state.pending_decision_by_pair.get(pair_key)
            if decision_index is None:
                self.state.warnings.append(f"No decision record found for child pair {pair_key}")
                continue
            record = self.state.decisions[decision_index]
            parent_a = self.game.get_rock(record.selected_parent_ids[0])
            parent_b = self.game.get_rock(record.selected_parent_ids[1])
            record.resulting_child_ids = [child.id for child in pair_children]
            record.resulting_child_values = [float(child.value) for child in pair_children]
            record.mutation_outcomes = [
                BreedingTrainingEnvironment._mutation_details(child, parent_a, parent_b)
                for child in pair_children
            ]
            self.state.mutation_count += sum(
                int(outcome["mutation_count"]) for outcome in record.mutation_outcomes
            )
            record.post_action_farm_metrics = calculate_farm_metrics(self.game)
        self.state.pending_decision_by_pair.clear()

    def _check_terminal_state(self) -> None:
        if self.state.terminated:
            return
        if self.game.game_over or self.game.generation >= self.game.max_generation:
            self.state.terminated = True
            self.state.termination_reason = "final_generation_reached"
        elif self.state.decision_count >= self.config.max_decisions:
            self.state.terminated = True
            self.state.termination_reason = "maximum_decisions_reached"
        elif not self.game.breeding_queue and not self._legal_pairs():
            self.state.terminated = True
            self.state.termination_reason = "no_legal_pairs"

    def step(
        self,
        action: AgentAction,
        *,
        agent_name: str,
        agent_seed: int,
        decision_context: dict[str, Any] | None = None,
    ) -> CampaignStepResult:
        if self.state.terminated:
            raise RuntimeError(f"Episode already terminated: {self.state.termination_reason}")
        if not isinstance(action, (BreedPairAction, StopGenerationAction, NoAction)):
            raise TypeError("action must be a typed AgentAction")
        context = decision_context or {}
        pre_metrics = calculate_farm_metrics(self.game)
        legal_pairs = {_pair_key(*pair) for pair in self._legal_pairs()}
        selected_ids = (
            (action.parent_a_id, action.parent_b_id)
            if isinstance(action, BreedPairAction)
            else None
        )
        record = AgentDecisionRecord(
            episode_id=self.state.episode_id,
            decision_index=self.state.decision_count,
            generation=self.game.generation,
            agent_name=agent_name,
            observation_summary={
                "generation": self.game.generation,
                "remaining_breeding_actions": self.game.max_pairs_per_generation - len(self.game.breeding_queue),
                "queued_pairs": len(self.game.breeding_queue),
            },
            legal_action_count=len(legal_pairs),
            selected_action=action.to_dict(),
            selected_parent_ids=selected_ids,
            ranked_candidate_pairs=list(context.get("ranked_candidate_pairs", [])),
            scores=dict(context.get("scores", {})),
            predictor_outputs=context.get("predictor_outputs"),
            objective_weights=self.state.objective_profile.to_dict(),
            pre_action_farm_metrics=pre_metrics,
            immediate_post_action_farm_metrics=pre_metrics,
            post_action_farm_metrics=pre_metrics,
            environment_seed=self.seed,
            agent_seed=agent_seed,
        )
        self.state.decisions.append(record)
        self.state.decision_count += 1
        children: list[genetics.Rock] = []
        generation_advanced = False
        valid = True
        error = None

        if isinstance(action, BreedPairAction):
            pair_key = _pair_key(action.parent_a_id, action.parent_b_id)
            if pair_key not in legal_pairs:
                valid = False
                error = "Agent selected a pair outside the authoritative legal-action set"
                self.state.invalid_decisions += 1
                record.status = "invalid"
                record.error = error
                self.state.errors.append(error)
                if self.config.invalid_action_policy == "terminate":
                    self.state.terminated = True
                    self.state.termination_reason = "invalid_action"
                else:
                    children = self._flush_generation()
                    generation_advanced = True
            else:
                try:
                    self.game.add_pair_to_queue(action.parent_a_id, action.parent_b_id)
                except (TypeError, ValueError) as exception:
                    valid = False
                    error = str(exception)
                    self.state.invalid_decisions += 1
                    record.status = "invalid"
                    record.error = error
                    self.state.errors.append(error)
                    self.state.terminated = True
                    self.state.termination_reason = "environment_failure"
                else:
                    self.state.valid_decisions += 1
                    self.state.pending_decision_by_pair[pair_key] = len(self.state.decisions) - 1
                    pair_evaluator_utility = context.get("pair_evaluator_utility")
                    if pair_evaluator_utility is not None:
                        self.state.cumulative_pair_utility += float(pair_evaluator_utility)
                    if len(self.game.breeding_queue) >= self.game.max_pairs_per_generation:
                        children = self._flush_generation()
                        generation_advanced = True
        elif isinstance(action, StopGenerationAction):
            if legal_pairs:
                self.state.early_stop_count += 1
            self.state.valid_decisions += 1
            children = self._flush_generation()
            generation_advanced = True
        else:
            self.state.valid_decisions += 1
            self.state.terminated = True
            self.state.termination_reason = f"no_action:{action.reason}"

        if self.state.decision_count >= self.config.max_decisions and not self.state.terminated:
            if self.game.breeding_queue:
                children.extend(self._flush_generation())
                generation_advanced = True
            self.state.terminated = True
            self.state.termination_reason = "maximum_decisions_reached"
        if generation_advanced:
            self._check_terminal_state()
        immediate_metrics = calculate_farm_metrics(self.game)
        record.immediate_post_action_farm_metrics = immediate_metrics
        record.post_action_farm_metrics = immediate_metrics
        record.status = self.state.termination_reason if self.state.terminated else "continue"
        return CampaignStepResult(
            action=action,
            valid=valid,
            children=tuple(children),
            generation_advanced=generation_advanced,
            terminated=self.state.terminated,
            termination_reason=self.state.termination_reason,
            error=error,
        )
