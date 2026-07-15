"""Sidebar configuration and command controls for the AI Observatory."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import streamlit as st

from Rock_AI.agents.breeding_agent_helper import DEFAULT_OBJECTIVE_PROFILES
from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_Streamlit.components.checkpoint_selector_helper import (
    discover_pair_ranker_checkpoints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _checkpoint_options(include_latest: bool):
    return discover_pair_ranker_checkpoints(REPOSITORY_ROOT, include_latest=include_latest)


def render_session_configuration() -> dict:
    st.header("Agent Setup")
    agent_type = st.selectbox("Agent", ("heuristic", "random", "neural", "oracle"), key="ai_obs_agent_type")
    checkpoint = ""
    predictor_checkpoint = ""
    if agent_type == "neural":
        custom_checkpoint = st.checkbox(
            "Use custom checkpoint paths",
            key="ai_obs_custom_checkpoint",
            help="Advanced option for models outside training_runs.",
        )
        if custom_checkpoint:
            checkpoint = st.text_input(
                "Pair-ranker checkpoint",
                key="ai_obs_checkpoint_path",
                placeholder="training_runs/pair_ranker/best.pt",
            ).strip()
            predictor_checkpoint = st.text_input(
                "Companion breeding-predictor checkpoint",
                key="ai_obs_predictor_checkpoint_path",
                placeholder="Only required by predictor-backed rankers",
            ).strip()
        else:
            show_latest = st.checkbox(
                "Show latest checkpoints",
                key="ai_obs_show_latest_checkpoints",
                help="Best checkpoints are recommended for normal use.",
            )
            options = _checkpoint_options(show_latest)
            if options:
                selected = st.selectbox(
                    "Neural model",
                    options,
                    format_func=lambda option: option.label,
                    key="ai_obs_checkpoint_option",
                )
                checkpoint = selected.ranker_path
                predictor_checkpoint = selected.predictor_path or ""
                st.caption(
                    "The companion breeding predictor is attached automatically."
                    if selected.predictor_path
                    else "This model is a standalone pair ranker."
                )
            else:
                st.warning("No complete pair-ranker checkpoint bundles were found locally.")
    farm_source = st.selectbox("Initial farm", ("Generated farm", "Current player game"), key="ai_obs_farm_source")
    objective_name = st.selectbox("Objective", tuple(DEFAULT_OBJECTIVE_PROFILES), key="ai_obs_objective")
    seed = st.number_input("Environment seed", min_value=0, value=1234, step=1, key="ai_obs_seed")
    max_generations = st.number_input("Maximum generations", 1, 50, 7, key="ai_obs_max_generations")
    max_pairs = st.number_input("Pairs per generation", 1, 10, 3, key="ai_obs_max_pairs")
    with st.expander("Breeding rules"):
        defaults = EncodedBreedingRules()
        mutation = st.slider("Mutation chance", 0.0, 1.0, float(defaults.mutation_chance), 0.005, key="ai_obs_mutation")
        death = st.slider("Child death chance", 0.0, 1.0, float(defaults.child_death_chance), 0.01, key="ai_obs_death")
        craisen = st.slider("Craisen chance", 0.0, 1.0, float(defaults.craisen_chance), 0.01, key="ai_obs_craisen")
        clutch_mean = st.number_input("Clutch mean", 0.1, 20.0, float(defaults.clutch_mean), 0.1, key="ai_obs_clutch_mean")
        clutch_std = st.number_input("Clutch standard deviation", 0.0, 20.0, float(defaults.clutch_std), 0.1, key="ai_obs_clutch_std")
    return {
        "agent_type": agent_type,
        "checkpoint": checkpoint,
        "predictor_checkpoint": predictor_checkpoint,
        "farm_source": farm_source,
        "objective_name": objective_name,
        "seed": int(seed),
        "max_generations": int(max_generations),
        "max_pairs": int(max_pairs),
        "rules": {
            **asdict(EncodedBreedingRules()),
            "mutation_chance": mutation,
            "child_death_chance": death,
            "craisen_chance": craisen,
            "clutch_mean": clutch_mean,
            "clutch_std": clutch_std,
        },
    }


def render_speed_configuration() -> dict:
    st.header("Presentation")
    delay = st.slider("Auto-run delay", 0.0, 5.0, 0.5, 0.1, key="ai_obs_delay")
    candidate_limit = st.number_input(
        "Ranked pairs to retain", 5, 100, 20, key="ai_obs_candidate_limit"
    )
    st.caption("Auto mode executes one decision per rerun.")
    with st.expander("Automatic pause triggers"):
        values = {
            "pause_after_every_action": st.checkbox("After every action", key="ai_obs_pause_action"),
            "pause_after_every_breeding": st.checkbox("After breeding", key="ai_obs_pause_breeding"),
            "pause_after_every_generation": st.checkbox("After generation", key="ai_obs_pause_generation"),
            "pause_on_mutation": st.checkbox("On mutation", value=True, key="ai_obs_pause_mutation"),
            "pause_on_rare_trait": st.checkbox("On rare trait", key="ai_obs_pause_rare"),
            "pause_on_new_farm_value_record": st.checkbox("On farm-value record", key="ai_obs_pause_farm_record"),
            "pause_on_new_high_value_rock": st.checkbox("On maximum-value record", key="ai_obs_pause_max_record"),
            "pause_on_close_decision": st.checkbox("When candidates are close", key="ai_obs_pause_close"),
            "pause_on_warning_or_fallback": st.checkbox("On warning or fallback", key="ai_obs_pause_warning"),
        }
        values["close_decision_threshold"] = st.number_input(
            "Close-score threshold", 0.0, 1000.0, 0.05, 0.01, key="ai_obs_close_threshold"
        )
    return {"delay": delay, "candidate_limit": int(candidate_limit), **values}
