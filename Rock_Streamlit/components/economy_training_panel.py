import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def discover_full_farmer_runs():
    rows = []
    for manifest in ROOT.glob("training_runs/*/run_manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("run_type") == "recurrent_neat_full_farmer":
            rows.append(manifest.parent)
    return sorted(rows)


def render_economy_training_panel():
    st.subheader("Full Farmer Training")
    st.caption("Training runs in an external CLI worker; completed safe JSON champions can be watched here without deployment.")
    population = st.number_input("Population", 4, 1000, 20, key="economy_train_population")
    generations = st.number_input("Generations", 1, 1000, 5, key="economy_train_generations")
    worlds = st.number_input("Worlds per genome", 1, 100, 3, key="economy_train_worlds")
    stage = st.selectbox("Curriculum start", ("breeding", "imports", "potions", "selling_listings", "bids", "trades", "full_economy"), index=1, key="economy_train_stage")
    output = st.text_input("Output run", "training_runs/full_farmer_next", key="economy_train_output")
    command = f"python -m Rock_AI.scripts.train_full_neat_farmers --output {output} --population {population} --generations {generations} --worlds-per-genome {worlds} --curriculum-start {stage} --seed 1234 --single-process"
    st.code(command, language="powershell")
    runs = discover_full_farmer_runs()
    if runs:
        st.success(f"{len(runs)} completed full-farmer run(s) available.")
        st.dataframe([{"run": path.relative_to(ROOT).as_posix()} for path in runs], width="stretch", hide_index=True)
