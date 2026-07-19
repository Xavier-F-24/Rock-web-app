"""Champion branch diversity controls."""

from __future__ import annotations

import streamlit as st

from Rock_AI.training_jobs.training_job_config import BranchInitializationStrategy


def branch_population_counts(population: int, elite: int, descendant_fraction: float, historical_fraction: float) -> dict[str, int]:
    descendants = min(population - elite, round(population * descendant_fraction))
    historical = min(population - elite - descendants, round(population * historical_fraction))
    return {"elite": elite, "descendants": descendants, "historical": historical, "fresh": population - elite - descendants - historical}


def render_branch_settings(population: int) -> dict:
    strategy = st.selectbox("Initialization strategy", [row.value for row in BranchInitializationStrategy], index=1, key="neat_branch_strategy")
    elite = st.number_input("Exact elite copies", 1, min(10, population), 1, key="neat_branch_elite")
    descendants = st.slider("Champion descendants", 0.0, 0.95, 0.60, 0.05, key="neat_branch_descendants")
    historical = st.slider("Historical diversity", 0.0, max(0.0, 1.0 - descendants), min(0.15, 1.0 - descendants), 0.05, key="neat_branch_historical")
    fresh = 1.0 - descendants - historical
    structural = st.slider("Structural mutation scale", 0.1, 5.0, 1.0, 0.1, key="neat_branch_structural")
    weight = st.slider("Weight mutation scale", 0.1, 5.0, 1.0, 0.1, key="neat_branch_weight")
    counts = branch_population_counts(population, int(elite), descendants, historical)
    st.caption(f"Population: {counts['elite']} elite, {counts['descendants']} descendants, {counts['historical']} historical, {counts['fresh']} fresh")
    return {"initialization_strategy": strategy, "exact_elite_count": int(elite), "champion_descendant_fraction": descendants, "historical_diversity_fraction": historical, "fresh_genome_fraction": fresh, "structural_mutation_scale": structural, "weight_mutation_scale": weight}
