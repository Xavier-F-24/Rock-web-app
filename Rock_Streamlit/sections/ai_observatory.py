"""Visual, synchronous control room for persistent Rock AI breeding sessions."""

from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from Rock_AI.agents.breeding_agent_helper import get_objective_profile
from Rock_AI.agents.heuristic_breeding_agent import HeuristicBreedingAgent
from Rock_AI.agents.neural_breeding_agent import NeuralBreedingAgent
from Rock_AI.agents.neat_breeding_agent import NeatBreedingAgent
from Rock_AI.agents.oracle_breeding_agent import OracleBreedingAgent
from Rock_AI.agents.random_breeding_agent import RandomBreedingAgent
from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignConfig
from Rock_AI.evaluation.breeding_agent_metrics import calculate_farm_metrics
from Rock_AI.policies.neural_pair_ranking_policy import NeuralPairRankingPolicy
from Rock_AI.policies.neat_pair_ranking_policy import NeatPairRankingPolicy
from Rock_AI.runtime import (
    AgentRuntimeManager,
    PauseSessionCommand,
    ResetSessionCommand,
    ResumeSessionCommand,
    RunGenerationCommand,
    SeekReplayCommand,
    StartSessionCommand,
    StepSessionCommand,
)
from Rock_AI.runtime.runtime_speed_helper import RuntimeSpeedConfig, RuntimeSpeedMode
from Rock_AI.runtime.runtime_state_helper import AgentRuntimeConfig, SessionStatus
from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.components.agent_control_panel import (
    render_session_configuration,
    render_speed_configuration,
)
from Rock_Streamlit.components.agent_timeline import render_timeline
from Rock_Streamlit.components.candidate_pair_panel import (
    render_candidate_pairs,
    render_parent_comparison,
)
from Rock_Streamlit.components.decision_explanation_panel import render_decision_explanation
from Rock_Streamlit.components.farm_visualizer import render_farm
from Rock_Streamlit.components.generation_summary_panel import render_generation_summaries
from Rock_Streamlit.components.model_info_panel import render_model_info
from Rock_Streamlit.components.mutation_event_panel import render_mutation_events
from Rock_Streamlit.components.network_visualizer import render_network_trace
from Rock_Streamlit.components.training_replay_panel import render_training_replay
from Rock_Streamlit.components.observatory_state_helper import (
    auto_run_should_continue,
    control_states,
    latest_event_rock_ids,
)
from Rock_Streamlit.components.replay_control_panel import render_replay_controls
from Rock_Streamlit.ui_components import metric_strip, page_header, section


MANAGER_KEY = "ai_observatory_runtime_manager"
SESSION_ID_KEY = "ai_observatory_session_id"
LIVE_SESSION_ID_KEY = "ai_observatory_live_session_id"
AUTO_RUN_KEY = "ai_observatory_auto_run"
LAST_RESULT_KEY = "ai_observatory_last_result"
REPLAY_MODE_KEY = "ai_observatory_replay_mode"


def get_runtime_manager() -> AgentRuntimeManager:
    manager = st.session_state.get(MANAGER_KEY)
    if getattr(manager, "interface_version", 0) != AgentRuntimeManager.INTERFACE_VERSION:
        replacement = AgentRuntimeManager()
        st.session_state[MANAGER_KEY] = replacement
        st.session_state.pop(SESSION_ID_KEY, None)
        st.session_state.pop(LIVE_SESSION_ID_KEY, None)
        st.session_state.pop(LAST_RESULT_KEY, None)
        st.session_state[AUTO_RUN_KEY] = False
        st.session_state[REPLAY_MODE_KEY] = False
        manager = replacement
    return manager


def _runtime_configuration(speed: dict) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        speed=RuntimeSpeedConfig(
            mode=RuntimeSpeedMode.AUTO,
            delay_after_decision_seconds=float(speed["delay"]),
            delay_after_breeding_seconds=float(speed["delay"]),
            delay_after_generation_seconds=float(speed["delay"]),
            pause_after_every_action=speed["pause_after_every_action"],
            pause_after_every_breeding=speed["pause_after_every_breeding"],
            pause_after_every_generation=speed["pause_after_every_generation"],
            pause_on_mutation=speed["pause_on_mutation"],
            pause_on_rare_trait=speed["pause_on_rare_trait"],
            pause_on_new_farm_value_record=speed["pause_on_new_farm_value_record"],
            pause_on_new_high_value_rock=speed["pause_on_new_high_value_rock"],
            pause_on_close_decision=speed["pause_on_close_decision"],
            pause_on_warning_or_fallback=speed["pause_on_warning_or_fallback"],
            close_decision_threshold=float(speed["close_decision_threshold"]),
        ),
        retain_top_candidates=int(speed["candidate_limit"]),
    )


def _make_agent(settings: dict, objective):
    agent_type = settings["agent_type"]
    if agent_type == "random":
        return RandomBreedingAgent(objective)
    if agent_type == "heuristic":
        return HeuristicBreedingAgent(objective)
    if agent_type == "oracle":
        return OracleBreedingAgent(objective, trial_count=50)
    checkpoint = settings["checkpoint"]
    if not checkpoint:
        raise ValueError("Choose a pair-ranker checkpoint before starting a neural agent.")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    if agent_type == "neat":
        return NeatBreedingAgent(NeatPairRankingPolicy.load(checkpoint_path), objective)
    predictor_checkpoint = settings.get("predictor_checkpoint")
    predictor_path = None
    if predictor_checkpoint:
        predictor_path = Path(predictor_checkpoint)
        if not predictor_path.is_absolute():
            predictor_path = Path(__file__).resolve().parents[2] / predictor_path
        if not predictor_path.exists():
            raise ValueError(f"Breeding-predictor checkpoint does not exist: {predictor_path}")
    try:
        policy = NeuralPairRankingPolicy.load(
            checkpoint_path,
            predictor_checkpoint=predictor_path,
        )
    except ValueError as error:
        if "requires a breeding-predictor checkpoint" in str(error):
            raise ValueError(
                "This pair ranker needs a companion breeding-predictor checkpoint. "
                "Choose a complete model bundle from the dropdown or provide both paths."
            ) from error
        raise
    return NeuralBreedingAgent(policy, objective)


def _create_session(manager, settings: dict, speed: dict):
    objective = get_objective_profile(settings["objective_name"])
    initial_farm = copy.deepcopy(get_game_state()) if settings["farm_source"] == "Current player game" else None
    environment_config = BreedingCampaignConfig(
        max_generations=settings["max_generations"],
        max_pairs_per_generation=settings["max_pairs"],
    )
    from Rock_AI.environments.breeding_campaign_environment import BreedingCampaignEnvironment

    environment = BreedingCampaignEnvironment(
        seed=settings["seed"],
        config=environment_config,
        objective_profile=objective,
    )
    session = manager.create_session(
        agent=_make_agent(settings, objective),
        environment=environment,
        seed=settings["seed"],
        objective_profile=objective,
        runtime_configuration=_runtime_configuration(speed),
        initial_farm=initial_farm,
        rules=settings["rules"],
    )
    st.session_state[SESSION_ID_KEY] = session.session_id
    st.session_state[LIVE_SESSION_ID_KEY] = session.session_id
    st.session_state[REPLAY_MODE_KEY] = False
    st.session_state[AUTO_RUN_KEY] = False
    return session


def _apply(manager, session_id, command):
    result = manager.apply(session_id, command)
    st.session_state[LAST_RESULT_KEY] = result
    if result.should_pause or result.status_after in {
        SessionStatus.PAUSED,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.CANCELLED,
    }:
        st.session_state[AUTO_RUN_KEY] = False
    return result


def _temporary_upload(upload) -> Path:
    suffix = Path(upload.name).suffix or ".json"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(upload.getvalue())
    handle.close()
    return Path(handle.name)


def _render_sidebar_loaders(manager) -> None:
    st.header("Session Files")
    session_upload = st.file_uploader("Load runtime session", type=("json",), key="ai_obs_session_upload")
    if st.button("Load Session", disabled=session_upload is None, key="ai_obs_load_session"):
        path = _temporary_upload(session_upload)
        try:
            session = manager.load_session(path)
            st.session_state[SESSION_ID_KEY] = session.session_id
            st.session_state[LIVE_SESSION_ID_KEY] = session.session_id
            st.session_state[REPLAY_MODE_KEY] = False
            st.success("Runtime session loaded.")
        except Exception as error:
            st.error(f"Could not load session: {error}")
        finally:
            path.unlink(missing_ok=True)
    replay_upload = st.file_uploader("Load episode replay", type=("jsonl",), key="ai_obs_replay_upload")
    if st.button("Load Replay", disabled=replay_upload is None, key="ai_obs_load_replay"):
        path = _temporary_upload(replay_upload)
        try:
            replay_session = manager.build_replay_session(path)
            st.session_state[SESSION_ID_KEY] = replay_session.session_id
            st.session_state[REPLAY_MODE_KEY] = True
            st.session_state[AUTO_RUN_KEY] = False
            st.success("Replay loaded.")
        except Exception as error:
            st.error(f"Could not load replay: {error}")
        finally:
            path.unlink(missing_ok=True)


def _render_agent_session() -> None:
    manager = get_runtime_manager()
    with st.sidebar:
        settings = render_session_configuration()
        speed = render_speed_configuration()
        _render_sidebar_loaders(manager)

    session_id = st.session_state.get(SESSION_ID_KEY)
    session = manager.sessions.get(session_id) if session_id else None
    replay_mode = bool(session and session.replay_controller is not None)
    controls = control_states(session.status if session else None, replay_mode=replay_mode)
    command_columns = st.columns(8)
    if command_columns[0].button("Start", disabled=not controls["start"], key="ai_obs_start"):
        try:
            if session is None or session.status != SessionStatus.CREATED:
                session = _create_session(manager, settings, speed)
            _apply(manager, session.session_id, StartSessionCommand())
            st.rerun()
        except Exception as error:
            st.error(f"Could not start agent: {error}")
    if command_columns[1].button("Step Action", disabled=not controls["step"], key="ai_obs_step"):
        _apply(manager, session.session_id, StepSessionCommand())
        st.rerun()
    if command_columns[2].button("Run Generation", disabled=not controls["run_generation"], key="ai_obs_generation"):
        _apply(manager, session.session_id, RunGenerationCommand())
        st.rerun()
    if command_columns[3].button("Auto Run", disabled=not controls["auto"], key="ai_obs_auto"):
        st.session_state[AUTO_RUN_KEY] = True
        st.rerun()
    if command_columns[4].button("Pause", disabled=not controls["pause"], key="ai_obs_pause"):
        _apply(manager, session.session_id, PauseSessionCommand())
        st.session_state[AUTO_RUN_KEY] = False
        st.rerun()
    if command_columns[5].button("Resume", disabled=not controls["resume"], key="ai_obs_resume"):
        _apply(manager, session.session_id, ResumeSessionCommand())
        st.rerun()
    if command_columns[6].button("Reset", disabled=not controls["reset"], key="ai_obs_reset"):
        _apply(manager, session.session_id, ResetSessionCommand(seed=settings["seed"]))
        st.session_state[AUTO_RUN_KEY] = False
        st.rerun()
    if command_columns[7].button("Replay", disabled=session is None or replay_mode, key="ai_obs_replay_live"):
        try:
            replay = manager.build_replay_from_session(session.session_id)
            st.session_state[SESSION_ID_KEY] = replay.session_id
            st.session_state[REPLAY_MODE_KEY] = True
            st.session_state[AUTO_RUN_KEY] = False
            st.rerun()
        except Exception as error:
            st.error(f"Could not build replay: {error}")

    if session is None:
        st.info("Choose an agent and settings in the sidebar, then press Start.")
        return

    if replay_mode:
        def seek(position):
            _apply(manager, session.session_id, SeekReplayCommand(position))
            st.rerun()

        render_replay_controls(session.replay_controller, seek)
        if st.button("Return to live session", key="ai_obs_return_live"):
            live_id = st.session_state.get(LIVE_SESSION_ID_KEY)
            if live_id in manager.sessions:
                st.session_state[SESSION_ID_KEY] = live_id
                st.session_state[REPLAY_MODE_KEY] = False
                st.rerun()

    game = session.current_farm_state
    metrics = calculate_farm_metrics(game)
    checkpoint = session.checkpoint_metadata.get("ranker_checkpoint_path") or "None"
    metric_strip(
        [
            ("Agent", session.agent.name if session.agent else "Replay"),
            ("Status", "REPLAY" if replay_mode else session.status.value.upper()),
            ("Generation", session.current_generation),
            ("Decision", session.current_decision_index),
            ("Seed", session.environment_seed),
            ("Objective", settings["objective_name"] if not replay_mode else "Recorded"),
        ]
    )
    st.caption(f"Session {session.session_id} | Checkpoint: {checkpoint}")
    if session.episode_termination_reason:
        st.info(f"Termination: {session.episode_termination_reason}")

    if not replay_mode:
        session.runtime_configuration = _runtime_configuration(speed)
    explanation = session.latest_decision_explanation
    if replay_mode:
        frame = session.replay_controller.current_frame
        explanation = frame.decision_explanation
        st.caption(f"Recorded action: {frame.selected_action or 'initial state'}")
    selected_ids = explanation.selected_parent_ids if explanation and explanation.selected_parent_ids else ()
    event_ids = latest_event_rock_ids(session.event_history)

    with section("Current Farm", "Gallery and authoritative lineage views share the same runtime state."):
        show_hidden_truth = bool(settings["show_hidden_truth"])
        summary_columns = st.columns(5)
        summary_columns[0].metric("Active Rocks", metrics["active_rock_count"])
        summary_columns[1].metric("Active Value", f"${metrics['final_active_rock_value']:.2f}")
        summary_columns[2].metric("Maximum Value", f"${metrics['final_maximum_rock_value']:.2f}")
        if show_hidden_truth:
            summary_columns[3].metric("Privileged genotype diversity", f"{metrics['genotype_diversity']:.2f}")
            summary_columns[4].metric("Privileged rare alleles", metrics["rare_trait_count"])
        else:
            summary_columns[3].metric("Average Value", f"${metrics['average_rock_value']:.2f}")
            summary_columns[4].metric("Surviving Offspring", metrics["surviving_offspring"])
        render_farm(
            game,
            selected_ids=selected_ids,
            child_ids=event_ids["children"],
            mutation_ids=event_ids["mutations"],
            show_hidden_truth=show_hidden_truth,
        )

    with section("Current Decision"):
        render_parent_comparison(game, explanation)
        render_decision_explanation(explanation)
        if explanation and explanation.predicted_offspring_summary:
            st.subheader("Predicted Offspring")
            st.json(explanation.predicted_offspring_summary)

    lower_left, lower_right = st.columns((1.2, 1))
    with lower_left, st.container(border=True):
        render_candidate_pairs(explanation, limit=int(speed["candidate_limit"]))
    with lower_right, st.container(border=True):
        render_mutation_events(session.event_history)
        render_generation_summaries(session.event_history)

    with section("Runtime History"):
        render_timeline(session.event_history)
    render_model_info(session)

    if not replay_mode:
        session_json = json.dumps(manager.export_session_state(session.session_id), indent=2, sort_keys=True)
        episode_json = json.dumps(manager.episode_record(session.session_id).to_dict(), sort_keys=True) + "\n"
        save_left, save_right = st.columns(2)
        save_left.download_button(
            "Save Session",
            session_json,
            file_name=f"{session.session_id}.json",
            mime="application/json",
            key="ai_obs_save_session",
        )
        save_right.download_button(
            "Save Episode Replay",
            episode_json,
            file_name=f"{session.session_id}.jsonl",
            mime="application/x-ndjson",
            key="ai_obs_save_episode",
        )

    result = st.session_state.get(LAST_RESULT_KEY)
    if result and result.error:
        st.error(result.error)
    auto_enabled = bool(st.session_state.get(AUTO_RUN_KEY, False))
    should_continue = auto_run_should_continue(
        session.status,
        auto_enabled,
        bool(result and result.should_pause),
    )
    if should_continue and not replay_mode:
        time.sleep(float(speed["delay"]))
        _apply(manager, session.session_id, StepSessionCommand())
        st.rerun()


def render() -> None:
    page_header(
        "AI Breeding Observatory",
        "Watch player-like agents decide, inspect safe network signals, and replay evolution.",
    )
    session_tab, network_tab, training_tab = st.tabs(
        ("Agent Session", "Network", "Training Replay")
    )
    with session_tab:
        _render_agent_session()
    with network_tab:
        manager = get_runtime_manager()
        session_id = st.session_state.get(SESSION_ID_KEY)
        session = manager.sessions.get(session_id) if session_id else None
        if session is None:
            st.info("Start or load an agent session to inspect its latest network trace.")
        else:
            decision = (
                session.environment.state.decisions[-1]
                if session.environment.state and session.environment.state.decisions
                else None
            )
            trace = decision.model_trace if decision is not None else None
            render_network_trace(trace)
            if trace:
                with st.expander("Certified trace metadata"):
                    st.json({
                        "model_type": trace.get("model_type"),
                        "checkpoint_id": trace.get("checkpoint_id"),
                        "topology_id": trace.get("topology_id"),
                        "observation_schema_version": trace.get("observation_schema_version"),
                        "normalizer_version": trace.get("normalizer_version"),
                        "observation_hash": trace.get("observation_hash"),
                        "trace_semantics": trace.get("trace_semantics"),
                    })
    with training_tab:
        render_training_replay()
