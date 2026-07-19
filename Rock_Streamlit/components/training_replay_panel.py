"""Safe discovery and replay of exported NEAT training artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import streamlit as st

from Rock_AI.visualization.network_visualization_helper import training_metrics_rows
from Rock_AI.logging.episode_record import episode_record_from_dict
from Rock_AI.replay.episode_replay_helper import EpisodeReplay
from Rock_Serialization.rock_serialization_helper import game_from_dict
from Rock_Streamlit.components.farm_visualizer import render_farm
from Rock_Streamlit.components.network_visualizer import render_network_trace, render_recurrent_topology


ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def discover_training_runs():
    return tuple(sorted(
        path.parent.relative_to(ROOT).as_posix()
        for path in ROOT.glob("training_runs/*neat*/run_manifest.json")
    ))


def render_training_replay() -> None:
    pending_run = st.session_state.pop("ai_obs_pending_training_run", None)
    if pending_run:
        discover_training_runs.clear()
        st.session_state["ai_obs_training_run"] = pending_run
        st.session_state.pop("ai_obs_training_generation", None)
    runs = discover_training_runs()
    if not runs:
        st.info("No exported NEAT training runs were found under training_runs.")
        return
    selected = st.selectbox("Saved training run", runs, key="ai_obs_training_run")
    run = ROOT / selected
    metrics_path = run / "generation_metrics.jsonl"
    metrics = training_metrics_rows(metrics_path.read_text(encoding="utf-8").splitlines())
    if metrics:
        st.line_chart({
            "Best fitness": [row["best_fitness"] for row in metrics],
            "Validation quality": [row["validation_quality"] for row in metrics],
        })
    champions = sorted((run / "champions").glob("generation_*"))
    generation = st.select_slider(
        "Champion generation", options=list(range(len(champions))),
        format_func=lambda index: champions[index].name if champions else "None",
        key="ai_obs_training_generation",
    ) if champions else None
    if generation is None:
        st.warning("This run has no exported champions.")
        return
    directory = champions[generation]
    network = json.loads((directory / "network.json").read_text(encoding="utf-8"))
    with gzip.open(directory / "showcase_episode.jsonl.gz", "rt", encoding="utf-8") as stream:
        showcase = json.loads(stream.readline())
    decisions = showcase.get("episode", {}).get("decisions", [])
    if not decisions:
        st.info("This champion export has no campaign decision replay yet; showing its evolved recurrent topology.")
        render_recurrent_topology(network)
        memory_path = directory / "memory_trace.npz"
        if memory_path.exists():
            memory = np.load(memory_path, allow_pickle=False)
            with st.expander("Showcase memory trace"):
                st.json({name: memory[name].tolist() for name in memory.files})
        return
    cursor = st.slider(
        "Showcase decision", 1, len(decisions), 1,
        key="ai_obs_showcase_decision",
    )
    decision = decisions[cursor - 1]
    trace = decision.get("model_trace")
    selected = trace.get("selected_candidate_ids") if trace else None
    score_key = "|".join(map(str, selected)) if selected else None
    output = trace.get("output_scores", {}).get(score_key) if trace else None
    if output is not None:
        st.metric("Showcase selected score", f"{float(output):.4f}")
    st.caption(showcase["note"])
    try:
        replay = EpisodeReplay.from_episode_record(
            episode_record_from_dict(showcase["episode"]),
            initial_farm=game_from_dict(showcase["initial_game"]),
        )
        frame = replay.frames[min(cursor, len(replay.frames) - 1)]
        selected_ids = tuple(decision.get("selected_parent_ids") or ())
        render_farm(
            frame.snapshot.state.game,
            selected_ids=selected_ids,
            child_ids=tuple(decision.get("resulting_child_ids") or ()),
            mutation_ids=tuple(
                row.get("rock_id") for row in decision.get("mutation_outcomes", ())
                if row.get("rock_id") is not None
            ),
            show_hidden_truth=False,
        )
        if replay.validation.divergences:
            st.warning("Showcase replay diverged from its recorded farm metrics.")
    except Exception as error:
        st.warning(f"Showcase farm replay is unavailable: {error}")
    render_network_trace(trace)
    with st.expander("Champion metadata"):
        st.json({
            "network": network["metadata"],
            "showcase_seed": showcase.get("showcase_seed"),
            "decision": decision,
            "final_summary": showcase.get("episode", {}).get("final_farm_summary"),
        })
