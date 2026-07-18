"""Synchronous command manager for persistent, inspectable agent sessions."""

from __future__ import annotations

import copy
import json
import random
import uuid
from pathlib import Path
from typing import Any, Mapping

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Serialization.rock_serialization_helper import game_from_dict, game_to_dict
from Rock_AI.agents.breeding_agent_helper import (
    BreedPairAction,
    BreedingAgent,
    NoAction,
    StopGenerationAction,
)
from Rock_AI.agents.heuristic_breeding_agent import HeuristicBreedingAgent
from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.agents.neat_breeding_agent import NeatBreedingAgent
from Rock_AI.agents.recurrent_neat_breeding_agent import RecurrentNeatBreedingAgent
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile
from Rock_AI.environments.breeding_campaign_environment import (
    BreedingCampaignConfig,
    BreedingCampaignEnvironment,
    BreedingCampaignState,
)
from Rock_AI.evaluation.breeding_agent_metrics import (
    calculate_farm_metrics,
    calculate_final_objective_utility,
)
from Rock_AI.explanations.candidate_explanation_helper import CandidateExplanation
from Rock_AI.explanations.decision_explanation_helper import (
    DecisionExplanation,
    build_decision_explanation,
    decision_explanation_from_dict,
)
from Rock_AI.logging.agent_decision_record import AgentDecisionRecord
from Rock_AI.logging.episode_record import EpisodeRecord
from Rock_AI.logging.episode_storage_helper import load_episode_records
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy
from Rock_AI.policies.recurrent_neat_pair_ranking_policy import RecurrentNeatPairRankingPolicy
from Rock_AI.policies.neat_pair_ranking_policy import NeatPairRankingPolicy
from Rock_AI.replay.episode_replay_helper import EpisodeReplay

from .agent_session_helper import AgentSession
from .runtime_command_helper import (
    CancelSessionCommand,
    PauseSessionCommand,
    ResetSessionCommand,
    ResumeSessionCommand,
    RunGenerationCommand,
    RunToCompletionCommand,
    RuntimeCommand,
    RuntimeCommandResult,
    SeekReplayCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from .runtime_event_helper import RuntimeEvent, RuntimeEventType
from .runtime_speed_helper import RuntimeSpeedMode, evaluate_pause_conditions
from .runtime_state_helper import (
    AgentRuntimeConfig,
    SessionStatus,
    TERMINAL_SESSION_STATUSES,
)


RUNTIME_SAVE_VERSION = 1

def _implements_breeding_agent_interface(agent: object) -> bool:
    """Accept agents across Streamlit hot reloads without relying on class identity."""
    required_methods = ("choose_action", "reset", "configuration")
    return (
        agent is not None
        and all(callable(getattr(agent, name, None)) for name in required_methods)
        and hasattr(agent, "objective_profile")
        and hasattr(agent, "name")
    )



def _jsonable_rng_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_rng_state(item) for item in value]
    return value


def _tuple_rng_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_rng_state(item) for item in value)
    return value


def _decision_from_dict(data: dict[str, Any]) -> AgentDecisionRecord:
    values = dict(data)
    if values.get("selected_parent_ids") is not None:
        values["selected_parent_ids"] = tuple(values["selected_parent_ids"])
    return AgentDecisionRecord(**values)


def migrate_runtime_save(data: dict[str, Any]) -> dict[str, Any]:
    version = int(data.get("runtime_save_version", 1))
    if version > RUNTIME_SAVE_VERSION:
        raise ValueError(
            f"Runtime save version {version} is newer than supported version {RUNTIME_SAVE_VERSION}"
        )
    migrated = dict(data)
    # Future migrations should advance one version at a time here.
    migrated["runtime_save_version"] = RUNTIME_SAVE_VERSION
    return migrated


class AgentRuntimeManager:
    INTERFACE_VERSION = 3

    def __init__(self):
        self.interface_version = self.INTERFACE_VERSION
        self.sessions: dict[str, AgentSession] = {}

    def create_session(
        self,
        *,
        agent: BreedingAgent,
        environment: BreedingCampaignEnvironment | None = None,
        seed: int = 0,
        objective_profile: FarmerObjectiveProfile | None = None,
        runtime_configuration: AgentRuntimeConfig | None = None,
        initial_farm: object | None = None,
        rules: EncodedBreedingRules | Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        if not _implements_breeding_agent_interface(agent):
            raise TypeError(
                "agent must provide choose_action(), reset(), configuration(), "
                "objective_profile, and name"
            )
        objective = objective_profile or agent.objective_profile
        agent.objective_profile = objective
        selected_environment = environment or BreedingCampaignEnvironment(
            seed=seed, objective_profile=objective
        )
        selected_environment.reset(
            seed,
            initial_farm=copy.deepcopy(initial_farm),
            rules=rules,
            objective_profile=objective,
        )
        agent_seed = int(seed) + 1_000_003
        agent.reset(agent_seed)
        identifier = session_id or f"agent-session-{uuid.uuid4().hex}"
        if identifier in self.sessions:
            raise ValueError(f"Session already exists: {identifier}")
        selected_environment.state.episode_id = identifier
        checkpoint_metadata = self._checkpoint_metadata(agent)
        session = AgentSession(
            session_id=identifier,
            agent=agent,
            environment=selected_environment,
            status=SessionStatus.CREATED,
            runtime_configuration=runtime_configuration or AgentRuntimeConfig(),
            objective_profile=objective,
            environment_seed=int(seed),
            agent_seed=agent_seed,
            checkpoint_metadata=checkpoint_metadata,
            episode_termination_reason=selected_environment.state.termination_reason,
            initial_farm_state=copy.deepcopy(selected_environment.game),
        )
        self.sessions[identifier] = session
        return session

    @staticmethod
    def _checkpoint_metadata(agent: BreedingAgent) -> dict[str, Any]:
        policy = getattr(agent, "policy", None)
        if policy is None:
            return {}
        artifact = getattr(policy, "artifact", None)
        if artifact is not None:
            metadata = {
                "neat_network_artifact_path": getattr(policy, "checkpoint_id", None),
                "information_access": artifact.information_access,
                "observation_schema_version": artifact.observation_schema_version,
                "normalizer_version": artifact.normalizer_version,
                "topology_id": artifact.topology_id,
                "model_type": "recurrent_neat_pair_ranker" if isinstance(agent, RecurrentNeatBreedingAgent) else "neat_pair_ranker",
            }
            if isinstance(agent, RecurrentNeatBreedingAgent):
                metadata["recurrent_neat_network_artifact_path"] = getattr(policy, "checkpoint_id", None)
            return metadata
        checkpoint = getattr(policy, "checkpoint", {})
        return {
            "ranker_checkpoint_path": getattr(policy, "checkpoint_path", None),
            "predictor_checkpoint_path": getattr(policy, "predictor_checkpoint_path", None),
            "encoding_schema_version": checkpoint.get("encoding_schema_version"),
            "dataset_schema_version": checkpoint.get("dataset_schema_version"),
            "game_rules_version": checkpoint.get("game_rules_version"),
            "model_architecture_config": checkpoint.get("model_architecture_config"),
            "epoch": checkpoint.get("epoch"),
            "validation_metrics": checkpoint.get("validation_metrics") or checkpoint.get("metrics"),
            "device": str(getattr(policy, "device", "unknown")),
        }

    def episode_record(self, session_id: str) -> EpisodeRecord:
        """Create a replayable record without mutating the live session."""
        session = self.get_session(session_id)
        if session.environment is None or session.agent is None:
            raise ValueError("Only live agent sessions can produce episode records")
        environment = session.environment
        state = environment.state
        final_summary = calculate_farm_metrics(environment.game)
        final_summary.update(
            {
                "mutation_count": state.mutation_count,
                "valid_decisions": state.valid_decisions,
                "invalid_decisions_attempted": state.invalid_decisions,
                "early_stop_count": state.early_stop_count,
                "cumulative_pair_evaluator_utility": state.cumulative_pair_utility,
                "objective_utility": calculate_final_objective_utility(
                    final_summary,
                    session.objective_profile,
                    mutation_count=state.mutation_count,
                ),
            }
        )
        return EpisodeRecord(
            episode_id=state.episode_id,
            initial_seed=session.environment_seed,
            agent_seed=session.agent_seed,
            agent_configuration=session.agent.configuration(),
            environment_configuration=environment.config.to_dict(),
            breeding_rules=state.rules.to_dict(),
            initial_farm_summary=state.initial_farm_summary,
            decisions=copy.deepcopy(state.decisions),
            final_farm_summary=final_summary,
            termination_reason=state.termination_reason or "in_progress",
            total_generations=environment.game.generation,
            total_breeding_decisions=sum(
                decision.selected_action.get("action_type") == "breed_pair" and decision.error is None
                for decision in state.decisions
            ),
            runtime_seconds=0.0,
            errors=list(state.errors),
            warnings=list(state.warnings),
        )

    def build_replay_from_session(
        self,
        session_id: str,
        *,
        replay_session_id: str | None = None,
    ) -> AgentSession:
        session = self.get_session(session_id)
        replay_session = self.build_replay_session(
            self.episode_record(session_id),
            initial_farm=copy.deepcopy(session.initial_farm_state),
            session_id=replay_session_id,
        )
        replay_session.event_history = copy.deepcopy(session.event_history)
        return replay_session

    def get_session(self, session_id: str) -> AgentSession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown agent session: {session_id}") from error

    def _emit(
        self,
        session: AgentSession,
        event_type: RuntimeEventType,
        summary: str,
        *,
        payload: dict[str, Any] | None = None,
        rock_ids=(),
        pre_metrics=None,
        post_metrics=None,
        decision_index: int | None = None,
        generation: int | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            session_id=session.session_id,
            event_index=len(session.event_history),
            decision_index=(
                session.current_decision_index if decision_index is None else decision_index
            ),
            generation=(session.current_generation if generation is None else generation),
            event_type=event_type,
            summary=summary,
            payload=payload or {},
            rock_ids=tuple(rock_ids),
            pre_action_metrics=pre_metrics,
            post_action_metrics=post_metrics,
            environment_seed=session.environment_seed,
            agent_seed=session.agent_seed,
        )
        session.event_history.append(event)
        session.touch()
        return event

    @staticmethod
    def _result(
        session: AgentSession,
        command: RuntimeCommand,
        before: SessionStatus,
        events: list[RuntimeEvent],
        **kwargs,
    ) -> RuntimeCommandResult:
        return RuntimeCommandResult(
            session_id=session.session_id,
            command_name=type(command).__name__,
            status_before=before,
            status_after=session.status,
            events=tuple(events),
            termination_reason=session.episode_termination_reason,
            **kwargs,
        )

    def _invalid_transition(
        self, session: AgentSession, command: RuntimeCommand, before: SessionStatus, message: str
    ) -> RuntimeCommandResult:
        return self._result(session, command, before, [], error=message)

    def apply(self, session_id: str, command: RuntimeCommand) -> RuntimeCommandResult:
        session = self.get_session(session_id)
        before = session.status
        if isinstance(command, StartSessionCommand):
            return self._start(session, command, before)
        if isinstance(command, PauseSessionCommand):
            return self._pause(session, command, before)
        if isinstance(command, ResumeSessionCommand):
            return self._resume(session, command, before)
        if isinstance(command, StepSessionCommand):
            return self._step_command(session, command, before)
        if isinstance(command, RunGenerationCommand):
            return self._run_generation(session, command, before)
        if isinstance(command, RunToCompletionCommand):
            return self._run_completion(session, command, before)
        if isinstance(command, CancelSessionCommand):
            return self._cancel(session, command, before)
        if isinstance(command, ResetSessionCommand):
            return self._reset(session, command, before)
        if isinstance(command, SeekReplayCommand):
            return self._seek_replay(session, command, before)
        raise TypeError(f"Unsupported runtime command: {type(command).__name__}")

    def _start(self, session, command, before):
        if before != SessionStatus.CREATED:
            return self._invalid_transition(session, command, before, "Start requires CREATED status")
        events = []
        if session.environment.state.terminated:
            session.status = SessionStatus.COMPLETED
            session.episode_termination_reason = session.environment.state.termination_reason
            events.append(self._emit(session, RuntimeEventType.SESSION_COMPLETED, "Session was already complete at startup."))
        else:
            session.status = (
                SessionStatus.WAITING_FOR_STEP
                if session.runtime_configuration.speed.mode == RuntimeSpeedMode.MANUAL_STEP
                else SessionStatus.READY
            )
            events.append(self._emit(session, RuntimeEventType.SESSION_STARTED, "Agent session started."))
        return self._result(session, command, before, events)

    def _pause(self, session, command, before):
        if before not in {SessionStatus.READY, SessionStatus.RUNNING, SessionStatus.WAITING_FOR_STEP}:
            return self._invalid_transition(session, command, before, "Pause requires a runnable session")
        session.status = SessionStatus.PAUSED
        event = self._emit(
            session,
            RuntimeEventType.SESSION_PAUSED,
            f"Session paused: {command.reason}.",
            payload={"reason": command.reason},
        )
        return self._result(session, command, before, [event], should_pause=True, pause_reasons=(command.reason,))

    def _resume(self, session, command, before):
        if before != SessionStatus.PAUSED:
            return self._invalid_transition(session, command, before, "Resume requires PAUSED status")
        session.status = (
            SessionStatus.WAITING_FOR_STEP
            if session.runtime_configuration.speed.mode == RuntimeSpeedMode.MANUAL_STEP
            else SessionStatus.READY
        )
        event = self._emit(session, RuntimeEventType.SESSION_RESUMED, "Agent session resumed.")
        return self._result(session, command, before, [event])

    def _can_execute(self, session: AgentSession) -> str | None:
        if session.status in TERMINAL_SESSION_STATUSES:
            return f"Session is terminal: {session.status.value}"
        if session.status == SessionStatus.CREATED:
            return "Session must be started before execution"
        if session.status == SessionStatus.PAUSED and not session.runtime_configuration.allow_step_while_paused:
            return "Stepping while paused is disabled"
        if session.replay_controller is not None:
            return "Replay sessions accept SeekReplayCommand only"
        return None

    def _execute_one(self, session: AgentSession) -> RuntimeCommandResult:
        before = session.status
        command = StepSessionCommand()
        error = self._can_execute(session)
        if error:
            return self._invalid_transition(session, command, before, error)
        session.status = SessionStatus.RUNNING
        environment = session.environment
        agent = session.agent
        observation = environment.observation()
        legal_actions = environment.legal_actions()
        pre_metrics = calculate_farm_metrics(environment.game)
        pre_statuses = {rock.id: rock.status.value for rock in environment.game.rocks.values()}
        decision_index = environment.state.decision_count
        events: list[RuntimeEvent] = []
        try:
            action = agent.choose_action(observation, legal_actions)
            context = copy.deepcopy(agent.last_decision_context)
        except Exception as exception:
            action = NoAction("agent_exception")
            context = {"warnings": [f"Agent exception: {exception}"]}
            environment.state.errors.append(f"Agent exception: {exception}")
        explanation = build_decision_explanation(
            action,
            farm=environment.game,
            legal_pair_ids=observation.legal_pair_ids,
            objective_profile=session.objective_profile,
            decision_context=context,
            retain_top_candidates=session.runtime_configuration.retain_top_candidates,
            mutation_chance=observation.breeding_rules.mutation_chance,
            close_decision_threshold=session.runtime_configuration.speed.close_decision_threshold,
        )
        session.latest_decision_explanation = explanation
        session.latest_ranked_candidates = list(explanation.top_candidates)
        events.append(
            self._emit(
                session,
                RuntimeEventType.DECISION_STARTED,
                f"Decision {decision_index} selected {action.to_dict()['action_type']}.",
                payload={
                    "selected_action": action.to_dict(),
                    "selected_parent_ids": list(explanation.selected_parent_ids or ()),
                    "legal_action_count": len(observation.legal_pair_ids),
                },
                pre_metrics=pre_metrics,
                decision_index=decision_index,
                generation=observation.generation,
            )
        )
        if explanation.top_candidates:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.CANDIDATE_PAIRS_SCORED,
                    f"Scored {len(observation.legal_pair_ids)} legal candidate pairs.",
                    payload={
                        "retained_candidates": [candidate.to_dict() for candidate in explanation.top_candidates]
                    },
                    decision_index=decision_index,
                    generation=observation.generation,
                )
            )
        if isinstance(action, BreedPairAction):
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.PAIR_SELECTED,
                    f"Selected rocks #{action.parent_a_id} and #{action.parent_b_id}.",
                    payload={"score": explanation.selected_candidate_score},
                    rock_ids=(action.parent_a_id, action.parent_b_id),
                    decision_index=decision_index,
                    generation=observation.generation,
                )
            )
        elif isinstance(action, StopGenerationAction) and not observation.legal_pair_ids:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.NO_LEGAL_ACTIONS,
                    "No legal breeding pairs were available.",
                    decision_index=decision_index,
                    generation=observation.generation,
                )
            )
        mutation_before = environment.state.mutation_count
        step_result = environment.step(
            action,
            agent_name=agent.name,
            agent_seed=session.agent_seed,
            decision_context=context,
        )
        post_metrics = calculate_farm_metrics(environment.game)
        record = environment.state.decisions[-1]
        if isinstance(action, BreedPairAction):
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.BREEDING_EXECUTED,
                    (
                        "Breeding queue executed."
                        if step_result.generation_advanced
                        else "Breeding pair added to the current generation queue."
                    ),
                    payload={"valid": step_result.valid, "generation_advanced": step_result.generation_advanced},
                    rock_ids=(action.parent_a_id, action.parent_b_id),
                    pre_metrics=pre_metrics,
                    post_metrics=post_metrics,
                    decision_index=decision_index,
                    generation=observation.generation,
                )
            )
        child_ids = tuple(child.id for child in step_result.children)
        if child_ids:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.CHILDREN_CREATED,
                    f"Created {len(child_ids)} child rocks.",
                    payload={"child_values": [child.value for child in step_result.children]},
                    rock_ids=child_ids,
                    post_metrics=post_metrics,
                    decision_index=decision_index,
                )
            )
        mutation_count = environment.state.mutation_count - mutation_before
        if mutation_count:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.MUTATION_OCCURRED,
                    f"Detected {mutation_count} inherited-allele mutation events.",
                    payload={"mutation_count": mutation_count},
                    rock_ids=child_ids,
                    decision_index=decision_index,
                )
            )
        status_changes = []
        for rock in environment.game.rocks.values():
            old = pre_statuses.get(rock.id)
            if old is not None and old != rock.status.value:
                status_changes.append({"rock_id": rock.id, "before": old, "after": rock.status.value})
        if status_changes:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.ROCK_STATUS_CHANGED,
                    f"Updated status for {len(status_changes)} rocks.",
                    payload={"changes": status_changes},
                    rock_ids=tuple(change["rock_id"] for change in status_changes),
                    decision_index=decision_index,
                )
            )
        if step_result.generation_advanced:
            events.append(
                self._emit(
                    session,
                    RuntimeEventType.GENERATION_ADVANCED,
                    f"Advanced to generation {environment.game.generation}.",
                    payload={"from_generation": observation.generation, "to_generation": environment.game.generation},
                    pre_metrics=pre_metrics,
                    post_metrics=post_metrics,
                    decision_index=decision_index,
                )
            )
        historical_farm_values = [
            float(record.immediate_post_action_farm_metrics.get("final_active_rock_value", 0.0))
            for record in session.environment.state.decisions[:-1]
            if record.immediate_post_action_farm_metrics
        ]
        previous_peak = max(
            historical_farm_values,
            default=float(session.environment.state.initial_farm_summary.get("final_active_rock_value", 0.0)),
        )
        pause = evaluate_pause_conditions(
            session.runtime_configuration.speed,
            mutation_count=mutation_count,
            rare_trait_increase=float(post_metrics["rare_trait_count"] - pre_metrics["rare_trait_count"]),
            farm_value_record=float(post_metrics["final_active_rock_value"]) > previous_peak,
            maximum_value_increase=float(post_metrics["final_maximum_rock_value"] - pre_metrics["final_maximum_rock_value"]),
            candidate_score_gap=explanation.first_second_score_difference,
            action_completed=True,
            breeding_executed=isinstance(action, BreedPairAction),
            generation_advanced=step_result.generation_advanced,
            warning_or_fallback=bool(explanation.warnings or explanation.fallback_reason),
        )
        if environment.state.terminated:
            session.episode_termination_reason = environment.state.termination_reason
            failed = environment.state.termination_reason in {
                "invalid_action",
                "environment_failure",
                "no_action:agent_exception",
            }
            session.status = SessionStatus.FAILED if failed else SessionStatus.COMPLETED
            event_type = RuntimeEventType.SESSION_FAILED if failed else RuntimeEventType.SESSION_COMPLETED
            events.append(
                self._emit(
                    session,
                    event_type,
                    f"Session ended: {environment.state.termination_reason}.",
                    payload={"termination_reason": environment.state.termination_reason},
                    post_metrics=post_metrics,
                    decision_index=decision_index,
                )
            )
        elif pause.should_pause:
            session.status = SessionStatus.PAUSED
        else:
            session.status = (
                SessionStatus.WAITING_FOR_STEP
                if session.runtime_configuration.speed.mode == RuntimeSpeedMode.MANUAL_STEP
                else SessionStatus.READY
            )
        session.touch()
        return RuntimeCommandResult(
            session_id=session.session_id,
            command_name="StepSessionCommand",
            status_before=before,
            status_after=session.status,
            events=tuple(events),
            decision_explanation=explanation,
            decisions_executed=1,
            generation_advanced=step_result.generation_advanced,
            should_pause=pause.should_pause,
            pause_reasons=pause.pause_reasons,
            termination_reason=session.episode_termination_reason,
            error=step_result.error,
        )

    def _step_command(self, session, command, before):
        result = self._execute_one(session)
        return RuntimeCommandResult(
            **{**result.__dict__, "command_name": type(command).__name__, "status_before": before}
        )

    def _run_generation(self, session, command, before):
        if session.status == SessionStatus.PAUSED:
            return self._invalid_transition(
                session, command, before, "RunGeneration requires ResumeSessionCommand while paused"
            )
        error = self._can_execute(session)
        if error:
            return self._invalid_transition(session, command, before, error)
        initial_generation = session.current_generation
        events = []
        decisions = 0
        pause_reasons = ()
        explanation = None
        while session.current_generation == initial_generation and session.status not in TERMINAL_SESSION_STATUSES:
            result = self._execute_one(session)
            if result.error and result.decisions_executed == 0:
                return self._invalid_transition(session, command, before, result.error)
            events.extend(result.events)
            decisions += result.decisions_executed
            explanation = result.decision_explanation
            if result.should_pause:
                pause_reasons = result.pause_reasons
                break
        return self._result(
            session,
            command,
            before,
            events,
            decision_explanation=explanation,
            decisions_executed=decisions,
            generation_advanced=session.current_generation != initial_generation,
            should_pause=bool(pause_reasons),
            pause_reasons=pause_reasons,
        )

    def _run_completion(self, session, command, before):
        if session.status == SessionStatus.PAUSED:
            return self._invalid_transition(
                session, command, before, "RunToCompletion requires ResumeSessionCommand while paused"
            )
        error = self._can_execute(session)
        if error:
            return self._invalid_transition(session, command, before, error)
        events = []
        decisions = 0
        pause_reasons = ()
        explanation = None
        start_generation = session.current_generation
        while session.status not in TERMINAL_SESSION_STATUSES:
            result = self._execute_one(session)
            events.extend(result.events)
            decisions += result.decisions_executed
            explanation = result.decision_explanation
            if result.should_pause:
                pause_reasons = result.pause_reasons
                break
            if result.error and result.decisions_executed == 0:
                break
        return self._result(
            session,
            command,
            before,
            events,
            decision_explanation=explanation,
            decisions_executed=decisions,
            generation_advanced=session.current_generation != start_generation,
            should_pause=bool(pause_reasons),
            pause_reasons=pause_reasons,
        )

    def _cancel(self, session, command, before):
        if before in TERMINAL_SESSION_STATUSES:
            return self._invalid_transition(session, command, before, "Session is already terminal")
        session.status = SessionStatus.CANCELLED
        session.episode_termination_reason = f"cancelled:{command.reason}"
        event = self._emit(
            session,
            RuntimeEventType.SESSION_CANCELLED,
            f"Session cancelled: {command.reason}.",
            payload={"reason": command.reason},
        )
        return self._result(session, command, before, [event])

    def _reset(self, session, command, before):
        if session.replay_controller is not None:
            return self._invalid_transition(session, command, before, "Replay sessions cannot be reset")
        seed = session.environment_seed if command.seed is None else int(command.seed)
        session.environment.reset(
            seed,
            initial_farm=copy.deepcopy(session.initial_farm_state),
            rules=session.environment.state.rules,
            objective_profile=session.objective_profile,
        )
        session.environment.state.episode_id = session.session_id
        session.environment_seed = seed
        session.agent_seed = seed + 1_000_003
        session.agent.reset(session.agent_seed)
        session.event_history.clear()
        session.latest_ranked_candidates.clear()
        session.latest_decision_explanation = None
        session.episode_termination_reason = session.environment.state.termination_reason
        session.failure_message = None
        session.status = SessionStatus.CREATED
        event = self._emit(
            session,
            RuntimeEventType.SESSION_RESET,
            f"Session reset with environment seed {seed}.",
            payload={"seed": seed},
        )
        return self._result(session, command, before, [event])

    def _seek_replay(self, session, command, before):
        if session.replay_controller is None:
            return self._invalid_transition(session, command, before, "SeekReplay requires a replay session")
        frame = session.replay_controller.seek(command.position)
        event = self._emit(
            session,
            RuntimeEventType.REPLAY_SEEKED,
            f"Replay moved to frame {frame.position}.",
            payload={"position": frame.position, "decision_index": frame.decision_index},
            post_metrics=frame.farm_summary,
        )
        return self._result(session, command, before, [event])

    def export_session_state(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session.replay_controller is not None:
            raise ValueError("Replay-only sessions are rebuilt from episode records, not runtime saves")
        environment = session.environment
        game = environment.game
        state = environment.state
        return {
            "runtime_save_version": RUNTIME_SAVE_VERSION,
            "session": {
                "session_id": session.session_id,
                "status": session.status.value,
                "runtime_configuration": session.runtime_configuration.to_dict(),
                "objective_profile": session.objective_profile.to_dict(),
                "environment_seed": session.environment_seed,
                "agent_seed": session.agent_seed,
                "episode_termination_reason": session.episode_termination_reason,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "checkpoint_metadata": session.checkpoint_metadata,
                "failure_message": session.failure_message,
                "agent_configuration": session.agent.configuration(),
                "agent_runtime_state": {
                    "rng_state": _jsonable_rng_state(session.agent.rng.getstate()),
                    "oracle_decision_index": getattr(session.agent, "_decision_index", None),
                    "recurrent_state": (
                        session.agent.policy.export_state()
                        if isinstance(session.agent, RecurrentNeatBreedingAgent) else None
                    ),
                },
                "event_history": [event.to_dict() for event in session.event_history],
                "latest_ranked_candidates": [candidate.to_dict() for candidate in session.latest_ranked_candidates],
                "latest_decision_explanation": (
                    session.latest_decision_explanation.to_dict()
                    if session.latest_decision_explanation is not None else None
                ),
                "initial_farm": game_to_dict(session.initial_farm_state),
            },
            "environment": {
                "configuration": environment.config.to_dict(),
                "rng_state": _jsonable_rng_state(environment.rng.getstate()),
                "game": game_to_dict(game),
                "game_rng_states": {
                    "game": _jsonable_rng_state(game.rng.getstate()),
                    "genome_factory": _jsonable_rng_state(game.genome_factory.rng.getstate()),
                    "name_generator": _jsonable_rng_state(game.name_generator.rng.getstate()),
                    "breeding_master": _jsonable_rng_state(game.breeding_master.rng.getstate()),
                },
                "state": {
                    "rules": state.rules.to_dict(),
                    "episode_id": state.episode_id,
                    "initial_farm_summary": state.initial_farm_summary,
                    "decisions": [decision.to_dict() for decision in state.decisions],
                    "pending_decision_by_pair": [
                        {"pair": list(pair), "decision_index": index}
                        for pair, index in state.pending_decision_by_pair.items()
                    ],
                    "decision_count": state.decision_count,
                    "valid_decisions": state.valid_decisions,
                    "invalid_decisions": state.invalid_decisions,
                    "early_stop_count": state.early_stop_count,
                    "mutation_count": state.mutation_count,
                    "cumulative_pair_utility": state.cumulative_pair_utility,
                    "terminated": state.terminated,
                    "termination_reason": state.termination_reason,
                    "errors": state.errors,
                    "warnings": state.warnings,
                },
            },
        }

    def save_session(self, session_id: str, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.export_session_state(session_id), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    def _restore_agent(self, config: dict, metadata: dict, model_registry=None) -> BreedingAgent:
        objective = FarmerObjectiveProfile(**config["objective_profile"])
        if model_registry is not None:
            if callable(model_registry):
                agent = model_registry(config, metadata)
                if agent is not None:
                    return agent
            elif isinstance(model_registry, Mapping):
                candidate = (
                    model_registry.get(config.get("agent_id"))
                    or model_registry.get(config.get("agent_type"))
                    or model_registry.get(metadata.get("ranker_checkpoint_path"))
                )
                if _implements_breeding_agent_interface(candidate):
                    return candidate
                if callable(candidate):
                    return candidate(config, metadata)
        agent_type = config["agent_type"]
        if agent_type == "RandomBreedingAgent":
            return RandomBreedingAgent(
                objective, stop_chance=float(config.get("stop_chance", 0.0)), agent_id=config["agent_id"]
            )
        if agent_type == "HeuristicBreedingAgent":
            return HeuristicBreedingAgent(objective, agent_id=config["agent_id"])
        if agent_type == "OracleBreedingAgent":
            return OracleBreedingAgent(
                objective,
                trial_count=int(config.get("trial_count", 100)),
                agent_id=config["agent_id"],
            )
        if agent_type == "NeuralBreedingAgent":
            checkpoint_path = metadata.get("ranker_checkpoint_path")
            if not checkpoint_path:
                raise ValueError("Neural session save has no ranker checkpoint path")
            policy = NeuralPairRankingPolicy.load(
                checkpoint_path,
                predictor_checkpoint=metadata.get("predictor_checkpoint_path"),
            )
            return NeuralBreedingAgent(
                policy,
                objective,
                utility_threshold=config.get("utility_threshold"),
                confidence_threshold=config.get("confidence_threshold"),
                temperature=float(config.get("temperature", 0.0)),
                agent_id=config["agent_id"],
            )
        if agent_type == "NeatBreedingAgent":
            artifact_path = metadata.get("neat_network_artifact_path")
            if not artifact_path:
                raise ValueError("NEAT session save has no safe network artifact path")
            policy = NeatPairRankingPolicy.load(artifact_path)
            return NeatBreedingAgent(
                policy,
                objective,
                utility_threshold=config.get("utility_threshold"),
                confidence_threshold=config.get("confidence_threshold"),
                temperature=float(config.get("temperature", 0.0)),
                agent_id=config["agent_id"],
            )
        if agent_type == "RecurrentNeatBreedingAgent":
            artifact_path = (
                metadata.get("recurrent_neat_network_artifact_path")
                or config.get("network_artifact")
            )
            if not artifact_path:
                raise ValueError("Recurrent NEAT session save has no safe topology artifact path")
            policy = RecurrentNeatPairRankingPolicy.load(artifact_path)
            return RecurrentNeatBreedingAgent(
                policy, objective,
                utility_threshold=config.get("utility_threshold"),
                agent_id=config["agent_id"],
            )
        raise ValueError(f"Cannot restore unknown agent type {agent_type!r} without model_registry")

    def load_session(self, path: str | Path, *, model_registry=None) -> AgentSession:
        payload = migrate_runtime_save(json.loads(Path(path).read_text(encoding="utf-8")))
        session_data = payload["session"]
        environment_data = payload["environment"]
        objective = FarmerObjectiveProfile(**session_data["objective_profile"])
        agent = self._restore_agent(
            session_data["agent_configuration"],
            session_data.get("checkpoint_metadata", {}),
            model_registry,
        )
        environment = BreedingCampaignEnvironment(
            seed=int(session_data["environment_seed"]),
            config=BreedingCampaignConfig(**environment_data["configuration"]),
            objective_profile=objective,
        )
        game = game_from_dict(environment_data["game"])
        state_data = environment_data["state"]
        decisions = [_decision_from_dict(row) for row in state_data.get("decisions", [])]
        environment.state = BreedingCampaignState(
            game=game,
            rules=EncodedBreedingRules.from_config(state_data["rules"]),
            objective_profile=objective,
            episode_id=state_data["episode_id"],
            initial_farm_summary=state_data["initial_farm_summary"],
            decisions=decisions,
            pending_decision_by_pair={
                tuple(row["pair"]): int(row["decision_index"])
                for row in state_data.get("pending_decision_by_pair", [])
            },
            decision_count=int(state_data["decision_count"]),
            valid_decisions=int(state_data["valid_decisions"]),
            invalid_decisions=int(state_data["invalid_decisions"]),
            early_stop_count=int(state_data["early_stop_count"]),
            mutation_count=int(state_data["mutation_count"]),
            cumulative_pair_utility=float(state_data["cumulative_pair_utility"]),
            terminated=bool(state_data["terminated"]),
            termination_reason=state_data.get("termination_reason"),
            errors=list(state_data.get("errors", [])),
            warnings=list(state_data.get("warnings", [])),
        )
        environment.seed = int(session_data["environment_seed"])
        environment.rng = random.Random()
        environment.rng.setstate(_tuple_rng_state(environment_data["rng_state"]))
        rng_states = environment_data["game_rng_states"]
        game.rng.setstate(_tuple_rng_state(rng_states["game"]))
        game.genome_factory.rng.setstate(_tuple_rng_state(rng_states["genome_factory"]))
        game.name_generator.rng.setstate(_tuple_rng_state(rng_states["name_generator"]))
        game.breeding_master.rng.setstate(_tuple_rng_state(rng_states["breeding_master"]))
        agent.seed = int(session_data["agent_seed"])
        agent.rng = random.Random()
        agent.rng.setstate(
            _tuple_rng_state(session_data["agent_runtime_state"]["rng_state"])
        )
        if hasattr(agent, "_decision_index") and session_data["agent_runtime_state"].get("oracle_decision_index") is not None:
            agent._decision_index = int(session_data["agent_runtime_state"]["oracle_decision_index"])
        recurrent_state = session_data["agent_runtime_state"].get("recurrent_state")
        if recurrent_state is not None:
            if not isinstance(agent, RecurrentNeatBreedingAgent):
                raise ValueError("Runtime save contains recurrent memory for a non-recurrent agent")
            agent.policy.import_state(recurrent_state)
        candidates = []
        for row in session_data.get("latest_ranked_candidates", []):
            values = dict(row)
            values["parent_ids"] = tuple(values["parent_ids"])
            values["parent_names"] = tuple(values["parent_names"])
            candidates.append(CandidateExplanation(**values))
        explanation = (
            decision_explanation_from_dict(session_data["latest_decision_explanation"])
            if session_data.get("latest_decision_explanation") else None
        )
        session = AgentSession(
            session_id=session_data["session_id"],
            agent=agent,
            environment=environment,
            status=SessionStatus(session_data["status"]),
            runtime_configuration=AgentRuntimeConfig.from_dict(session_data["runtime_configuration"]),
            objective_profile=objective,
            environment_seed=int(session_data["environment_seed"]),
            agent_seed=int(session_data["agent_seed"]),
            event_history=[RuntimeEvent.from_dict(row) for row in session_data.get("event_history", [])],
            latest_ranked_candidates=candidates,
            latest_decision_explanation=explanation,
            episode_termination_reason=session_data.get("episode_termination_reason"),
            created_at=session_data["created_at"],
            updated_at=session_data["updated_at"],
            checkpoint_metadata=session_data.get("checkpoint_metadata", {}),
            failure_message=session_data.get("failure_message"),
            initial_farm_state=game_from_dict(session_data["initial_farm"]),
        )
        self.sessions[session.session_id] = session
        return session

    def build_replay_session(
        self,
        source: EpisodeRecord | str | Path,
        *,
        initial_farm: object | None = None,
        session_id: str | None = None,
    ) -> AgentSession:
        record = load_episode_records(source)[0] if isinstance(source, (str, Path)) else source
        replay = EpisodeReplay.from_episode_record(record, initial_farm=initial_farm)
        objective = FarmerObjectiveProfile(**record.agent_configuration["objective_profile"])
        identifier = session_id or f"replay-session-{uuid.uuid4().hex}"
        session = AgentSession(
            session_id=identifier,
            agent=None,
            environment=None,
            status=SessionStatus.READY,
            runtime_configuration=AgentRuntimeConfig(),
            objective_profile=objective,
            environment_seed=record.initial_seed,
            agent_seed=record.agent_seed,
            episode_termination_reason=record.termination_reason,
            replay_controller=replay,
        )
        self.sessions[identifier] = session
        return session
