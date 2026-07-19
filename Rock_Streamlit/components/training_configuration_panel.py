"""Bounded training controls and pure review helpers."""

from __future__ import annotations

import re

import streamlit as st

from Rock_AI.training_jobs.training_job_config import TrainingSafetyTier


def sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not cleaned: raise ValueError("Run name must contain letters or numbers")
    return cleaned[:80]


def estimated_evaluations(population: int, generations: int, training_scenarios: int, validation_scenarios: int) -> int:
    return int(population) * int(generations) * (int(training_scenarios) + int(validation_scenarios))


def render_training_scope() -> dict:
    tier = st.segmented_control("Training size", [row.value for row in TrainingSafetyTier], default="smoke", key="neat_job_tier")
    smoke = tier == "smoke"
    generations = st.number_input("Additional generations", 1, 5 if smoke else 500, 2, key="neat_job_generations")
    population = st.number_input("Population", 2, 20 if smoke else 2000, 10 if smoke else 100, key="neat_job_population")
    training = st.number_input("Training scenarios", 1, 10 if smoke else 1000, 5, key="neat_job_training_scenarios")
    validation = st.number_input("Validation scenarios", 1, 5 if smoke else 500, 3, key="neat_job_validation_scenarios")
    checkpoint = st.number_input("Checkpoint frequency", 1, 100, 1, key="neat_job_checkpoint_frequency")
    campaign = st.number_input("Campaign depth", 1, 20, 3, key="neat_job_campaign_depth")
    advanced = st.checkbox("I understand Advanced Training can consume substantial local resources", key="neat_job_advanced_ack") if tier == "advanced" else False
    return {"tier": tier, "generations": int(generations), "population": int(population), "training_scenarios": int(training), "validation_scenarios": int(validation), "checkpoint_frequency": int(checkpoint), "campaign_generations": int(campaign), "advanced_acknowledged": advanced}
