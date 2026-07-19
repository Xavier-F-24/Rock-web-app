"""Durable local full-farmer NEAT controls for the World Economy Observatory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from Rock_AI.runtime.local_training_environment_helper import detect_training_environment
from Rock_AI.training_jobs import (
    TrainingBackendKind, TrainingJobConfig, TrainingJobManager, TrainingJobState,
    TrainingOperation, TrainingSafetyTier,
)
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader
from .training_job_status_panel import render_training_job_status
from .training_progress_panel import render_training_progress


ROOT = Path(__file__).resolve().parents[2]
FULL_JOB_KEY = "ai_obs_full_farmer_training_job_id"


def discover_full_farmer_runs():
    rows = []
    for manifest in ROOT.glob("training_runs/*/run_manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("run_type") == "recurrent_neat_full_farmer" and (manifest.parent / "champions" / "best_validation" / "network.json").exists():
            rows.append(manifest.parent)
    return sorted(rows)


def _discover_training_sources():
    rows = []
    for manifest in ROOT.glob("training_runs/*/run_manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("run_type") != "recurrent_neat_full_farmer":
            continue
        run = manifest.parent
        rows.append({
            "run": run.relative_to(ROOT).as_posix(),
            "checkpoints": sorted(run.glob("checkpoints/neat-checkpoint-*")),
            "champions": sorted(run.glob("champions/generation_*/network.json")),
        })
    return rows


def _render_active_job(manager, job_id):
    directory = manager.jobs_root / job_id
    if not directory.exists():
        st.error("The selected training job no longer exists.")
        return
    reader = TrainingProgressReader(directory)
    status = reader.status()
    render_training_job_status(status, reader.orphan_warning())
    if status.current_curriculum_stage:
        st.caption(
            f"Curriculum: {status.current_curriculum_stage.replace('_', ' ').title()} | "
            f"World evaluations: {status.worlds_evaluated:,} | "
            f"Invalid actions: {(status.invalid_action_rate or 0):.1%} | "
            f"Market actions: {(status.market_transaction_rate or 0):.1%}"
        )
    left, middle, right = st.columns(3)
    left.button("Refresh", key="full_farmer_job_refresh")
    terminal = status.status in {TrainingJobState.COMPLETED, TrainingJobState.CANCELLED, TrainingJobState.FAILED, TrainingJobState.ORPHANED}
    if middle.button("Cancel", disabled=terminal, key="full_farmer_job_cancel"):
        manager.request_cancel(job_id)
        st.rerun()
    if right.button("Clear", key="full_farmer_job_clear"):
        st.session_state.pop(FULL_JOB_KEY, None)
        st.rerun()
    render_training_progress(reader.progress())
    if status.status is TrainingJobState.COMPLETED and status.latest_safe_champion_export:
        st.success("Generation completed. The safe champion is ready for the World Economy viewer or another training branch.")
    elif not terminal and st.checkbox("Auto refresh", True, key="full_farmer_job_auto_refresh"):
        time.sleep(2)
        st.rerun()


def render_economy_training_panel():
    st.subheader("Full Farmer Training")
    capabilities = detect_training_environment(ROOT)
    if not capabilities.subprocess_supported:
        st.warning(capabilities.reason or "This deployment supports completed-run observation only.")
        return
    manager = TrainingJobManager(ROOT)
    active_job = st.session_state.get(FULL_JOB_KEY)
    if active_job:
        _render_active_job(manager, active_job)
        st.divider()

    sources = _discover_training_sources()
    operation_label = st.segmented_control(
        "Operation", ("New Run", "Continue Run", "Continue As New", "Branch Champion"),
        default="New Run", key="full_farmer_operation",
    )
    operation = {
        "New Run": TrainingOperation.NEW_RUN,
        "Continue Run": TrainingOperation.CONTINUE,
        "Continue As New": TrainingOperation.CONTINUE_AS_BRANCH,
        "Branch Champion": TrainingOperation.BRANCH_CHAMPION,
    }[operation_label]
    source = None
    if operation is not TrainingOperation.NEW_RUN:
        if not sources:
            st.info("No durable full-farmer source run is available yet.")
            return
        source = st.selectbox("Source run", sources, format_func=lambda row: row["run"], key="full_farmer_source")

    checkpoint = None
    champion = None
    source_generation = None
    if operation in {TrainingOperation.CONTINUE, TrainingOperation.CONTINUE_AS_BRANCH} and source:
        if source["checkpoints"]:
            checkpoint = st.selectbox("Population checkpoint", source["checkpoints"], format_func=lambda path: path.name, key="full_farmer_checkpoint")
        else:
            st.error("This run has no resumable population checkpoint. Use a champion branch instead.")
    elif operation is TrainingOperation.BRANCH_CHAMPION and source:
        if source["champions"]:
            champion = st.selectbox("Safe champion", source["champions"], format_func=lambda path: path.parent.name, key="full_farmer_champion")
            source_generation = int(champion.parent.name.rsplit("_", 1)[-1])
        else:
            st.error("This run has no per-generation safe champion.")

    population = st.number_input("Population", 4, 1000, 20, key="economy_train_population")
    generations = st.number_input("Generations this job", 1, 1000, 5, key="economy_train_generations")
    worlds = st.number_input("Worlds per genome", 1, 100, 3, key="economy_train_worlds")
    rounds = st.number_input("Rounds per world", 1, 100, 6, key="economy_train_rounds")
    start_stage = st.selectbox("Curriculum start", ("breeding", "imports", "potions", "selling_listings", "bids", "trades", "full_economy", "opponent_generalization"), index=1, key="economy_train_stage")
    seed = st.number_input("Training seed", min_value=0, value=1234, key="economy_train_seed")
    default_name = "full_farmer_next" if source is None else Path(source["run"]).name + "_branch"
    output_name = st.text_input("Destination run", default_name, disabled=operation is TrainingOperation.CONTINUE, key="economy_train_output")
    output = source["run"] if operation is TrainingOperation.CONTINUE else f"training_runs/{''.join(character for character in output_name if character.isalnum() or character in '_-')}"
    conflict = operation is not TrainingOperation.CONTINUE and (ROOT / output).exists() and any((ROOT / output).iterdir())
    if conflict:
        st.error("Destination already exists and is not empty.")
    confirmed = st.checkbox("I confirm this bounded local training job", key="economy_train_confirm")
    needs_checkpoint = operation in {TrainingOperation.CONTINUE, TrainingOperation.CONTINUE_AS_BRANCH} and checkpoint is None
    needs_champion = operation is TrainingOperation.BRANCH_CHAMPION and champion is None
    if st.button("Launch Full Farmer Worker", disabled=not confirmed or conflict or needs_checkpoint or needs_champion, key="economy_train_launch"):
        config = TrainingJobConfig(
            operation=operation, source_run="" if source is None else source["run"], output_run=output,
            additional_generations=int(generations), seed=int(seed),
            source_checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
            source_champion=str(champion.relative_to(ROOT)) if champion else None,
            source_generation=source_generation, population_size=int(population),
            training_scenarios=1, validation_scenarios=1,
            checkpoint_frequency=1, showcase_frequency=1,
            safety_tier=TrainingSafetyTier.STANDARD,
            trainer_kind=TrainingBackendKind.FULL_FARMER,
            worlds_per_genome=int(worlds), max_rounds_per_world=int(rounds),
            curriculum_start=start_stage,
        )
        manifest = manager.create_job(config)
        manager.launch(manifest.job_id)
        st.session_state[FULL_JOB_KEY] = manifest.job_id
        st.rerun()
