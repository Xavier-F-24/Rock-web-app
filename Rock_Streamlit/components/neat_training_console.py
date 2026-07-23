"""Durable local NEAT training console for the AI Observatory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from Rock_AI.runtime.local_training_environment_helper import detect_training_environment
from Rock_AI.training_jobs import TrainingJobConfig, TrainingJobManager, TrainingJobState, TrainingOperation
from Rock_AI.training_jobs.training_job_config import BranchInitializationStrategy, TrainingSafetyTier
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader
from .champion_branch_panel import render_branch_settings
from .completed_generation_panel import render_completed_job, topology_diff
from .training_configuration_panel import estimated_evaluations, render_training_scope, sanitize_run_name
from .training_job_status_panel import render_training_job_status
from .training_log_panel import render_training_log
from .training_progress_panel import render_training_progress


ROOT = Path(__file__).resolve().parents[2]
JOB_KEY = "ai_obs_training_job_id"


def discover_resumable_runs():
    rows = []
    for manifest in ROOT.glob("training_runs/*/run_manifest.json"):
        run = manifest.parent
        try: metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        if metadata.get("run_type") != "recurrent_neat_player_farmer": continue
        champions = sorted(run.glob("champions/generation_*/network.json"))
        checkpoints = sorted(run.glob("checkpoints/neat-checkpoint-*"))
        if champions: rows.append({"run": run.relative_to(ROOT).as_posix(), "champions": champions, "checkpoints": checkpoints})
    return rows


def launch_enabled(capabilities, confirmed: bool, active_job: bool) -> bool:
    return bool(capabilities.subprocess_supported and capabilities.persistent_storage_supported and capabilities.writable_training_directory and confirmed and not active_job)


def _render_durable_progress(status):
    generation_current, generation_total, generation_fraction = status.generation_progress
    st.progress(
        generation_fraction,
        text=f"Training generations: {generation_current} / {generation_total}",
    )
    operation = status.operation_progress
    if operation:
        current, total, fraction = operation
        label = status.operation_progress_label or "Current operation"
        st.progress(fraction, text=f"{label}: {current} / {total}")
    context = [
        f"Phase: {status.heartbeat_phase or 'starting'}",
        f"Operation: {status.last_completed_operation or 'waiting for first update'}",
    ]
    if status.current_genome_id is not None:
        context.append(f"Genome: {status.current_genome_id}")
    if status.current_scenario_id is not None:
        context.append(f"Scenario: {status.current_scenario_id}")
    if status.current_world_turn is not None:
        context.append(f"World turn: {status.current_world_turn}")
    st.caption(" | ".join(context))


def _render_active_job(manager: TrainingJobManager, job_id: str):
    directory = manager.jobs_root / job_id
    if not directory.exists(): st.error("Selected training job no longer exists."); return
    reader = TrainingProgressReader(directory); status = reader.status()
    orphan = reader.orphan_warning()
    render_training_job_status(status, orphan)
    _render_durable_progress(status)
    controls = st.columns(3)
    controls[0].button("Refresh", key="neat_job_refresh")
    if controls[1].button("Request cancellation", disabled=status.status in {TrainingJobState.COMPLETED, TrainingJobState.CANCELLED, TrainingJobState.FAILED}, key="neat_job_cancel"):
        manager.request_cancel(job_id); st.rerun()
    if controls[2].button("Clear selection", key="neat_job_clear"): st.session_state.pop(JOB_KEY, None); st.rerun()
    render_training_progress(reader.progress()); render_training_log(reader.console_tail())
    auto_refresh = st.checkbox("Automatically refresh active job", value=True, key="neat_job_auto_refresh")
    refresh_interval = st.slider("Refresh interval", 1, 10, 2, key="neat_job_refresh_interval")
    if orphan and status.latest_checkpoint and st.button("Create recovery job", key="neat_job_recover"):
        recovered = manager.recover(job_id); manager.launch(recovered.job_id)
        st.session_state[JOB_KEY] = recovered.job_id; st.rerun()
    if status.status is TrainingJobState.COMPLETED:
        reference_path = directory / "output_run_reference.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
        render_completed_job(status, reference)
        if reference and st.button("Watch completed champion", key="neat_job_watch_champion"):
            run_path = Path(reference["output_run"])
            st.session_state["ai_obs_pending_training_run"] = run_path.relative_to(ROOT).as_posix()
            st.rerun()
        manifest_path = directory / "job_manifest.json"
        if reference and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest["config"]
            parent = ROOT / config["source_champion"] if config.get("source_champion") else None
            child = Path(reference.get("latest_champion") or "")
            if parent and parent.exists() and child.exists():
                with st.expander("Parent versus completed champion"):
                    st.json(topology_diff(json.loads(parent.read_text(encoding="utf-8")), json.loads(child.read_text(encoding="utf-8"))))
    elif auto_refresh and status.status not in {TrainingJobState.CANCELLED, TrainingJobState.FAILED, TrainingJobState.ORPHANED}:
        time.sleep(float(refresh_interval)); st.rerun()


def render_neat_training_console():
    st.subheader("Local NEAT Training")
    capabilities = detect_training_environment(ROOT)
    if not capabilities.subprocess_supported:
        st.warning(capabilities.reason or "This environment supports replay and inference only.")
    manager = TrainingJobManager(ROOT)
    active_job = st.session_state.get(JOB_KEY)
    if active_job:
        _render_active_job(manager, active_job)
        st.divider()
    runs = discover_resumable_runs()
    if not runs: st.info("No compatible exported NEAT runs were found."); return
    run_row = st.selectbox("Source run", runs, format_func=lambda row: row["run"], key="neat_job_source_run")
    operation_label = st.segmented_control("Operation", ("Continue Full Run", "Continue As New Run", "Branch From Champion"), default="Branch From Champion", key="neat_job_operation")
    operation = {"Continue Full Run": TrainingOperation.CONTINUE, "Continue As New Run": TrainingOperation.CONTINUE_AS_BRANCH, "Branch From Champion": TrainingOperation.BRANCH_CHAMPION}[operation_label]
    checkpoint = None; champion = None; source_generation = None
    if operation is TrainingOperation.BRANCH_CHAMPION:
        champion = st.selectbox("Source champion", run_row["champions"], format_func=lambda path: path.parent.name, key="neat_job_champion")
        source_generation = int(champion.parent.name.rsplit("_", 1)[-1])
    else:
        if not run_row["checkpoints"]: st.error("This run has no full resumable checkpoint.")
        else: checkpoint = st.selectbox("Trusted local checkpoint", run_row["checkpoints"], format_func=lambda path: path.name, key="neat_job_checkpoint")
    scope = render_training_scope()
    with st.expander("Fitness configuration"):
        ranking_weight = st.slider("Supervised ranking weight", 0.0, 1.0, 0.60, 0.05, key="neat_job_ranking_weight")
        campaign_weight = 1.0 - ranking_weight
        complexity_penalty = st.number_input("Topology complexity penalty", 0.0, 0.01, 0.00001, 0.00001, format="%.5f", key="neat_job_complexity_penalty")
        st.caption(f"Campaign weight: {campaign_weight:.2f}. Memory and novelty fitness terms are not supported by the current trainer and remain zero.")
        st.caption("Invalid actions and numerical failures retain the trainer's fixed safety penalties.")
    branch = render_branch_settings(scope["population"]) if operation is TrainingOperation.BRANCH_CHAMPION else {}
    default_name = Path(run_row["run"]).name if operation is TrainingOperation.CONTINUE else f"{Path(run_row['run']).name}_branch"
    run_name = st.text_input("Destination run name", value=default_name, disabled=operation is TrainingOperation.CONTINUE, key="neat_job_output_name")
    seed = st.number_input("Branch/training seed", min_value=0, value=1234, key="neat_job_seed")
    destination = run_row["run"] if operation is TrainingOperation.CONTINUE else f"training_runs/{sanitize_run_name(run_name)}"
    destination_path = ROOT / destination
    destination_conflict = operation is not TrainingOperation.CONTINUE and destination_path.exists() and any(destination_path.iterdir())
    writer_lock = destination_path.parent / f".{destination_path.name}.training_writer_lock"
    if destination_conflict: st.error("Destination run already exists and is not empty. Choose a new branch name.")
    if writer_lock.exists(): st.error("Another worker currently owns the destination run.")
    estimate = estimated_evaluations(scope["population"], scope["generations"], scope["training_scenarios"], scope["validation_scenarios"])
    command = f"python -m Rock_AI.scripts.run_neat_training_job --job training_jobs/job_<uuid>"
    st.info(f"{operation.value.replace('_', ' ').title()} into `{destination}` for {scope['generations']} generations. Estimated genome-scenario evaluations: {estimate:,}.")
    st.code(command, language="powershell")
    confirmed = st.checkbox("I confirm this operation and destination", key="neat_job_confirm")
    active = bool(active_job and (manager.jobs_root / active_job).exists() and TrainingProgressReader(manager.jobs_root / active_job).status().status not in {TrainingJobState.COMPLETED, TrainingJobState.CANCELLED, TrainingJobState.FAILED})
    if st.button("Launch Training Worker", disabled=not launch_enabled(capabilities, confirmed, active) or destination_conflict or writer_lock.exists() or (operation is not TrainingOperation.BRANCH_CHAMPION and checkpoint is None), key="neat_job_launch"):
        config = TrainingJobConfig(
            operation=operation, source_run=run_row["run"], output_run=destination,
            additional_generations=scope["generations"], seed=int(seed),
            source_checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
            source_generation=source_generation,
            source_champion=str(champion.relative_to(ROOT)) if champion else None,
            population_size=scope["population"], training_scenarios=scope["training_scenarios"],
            validation_scenarios=scope["validation_scenarios"], campaign_generations=scope["campaign_generations"],
            checkpoint_frequency=scope["checkpoint_frequency"], safety_tier=TrainingSafetyTier(scope["tier"]),
            advanced_acknowledged=scope["advanced_acknowledged"],
            initialization_strategy=BranchInitializationStrategy(branch.get("initialization_strategy", "champion_and_diverse_seeds")),
            exact_elite_count=branch.get("exact_elite_count", 1), champion_descendant_fraction=branch.get("champion_descendant_fraction", 0.60),
            fresh_genome_fraction=branch.get("fresh_genome_fraction", 0.25), historical_diversity_fraction=branch.get("historical_diversity_fraction", 0.15),
            structural_mutation_scale=branch.get("structural_mutation_scale", 1.0), weight_mutation_scale=branch.get("weight_mutation_scale", 1.0),
            supervised_weight=float(ranking_weight), campaign_weight=float(campaign_weight), complexity_penalty=float(complexity_penalty),
        )
        manifest = manager.create_job(config); manager.launch(manifest.job_id)
        st.session_state[JOB_KEY] = manifest.job_id; st.rerun()
