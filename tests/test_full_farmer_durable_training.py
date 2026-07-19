import json

from Rock_AI.training.action_curriculum import ActionCurriculumStage
from Rock_AI.training.full_farmer_training_config import FullFarmerTrainingConfig
from Rock_AI.training.full_farmer_training_state import FullFarmerTrainingState, should_advance_curriculum
from Rock_AI.training_jobs import (
    TrainingBackendKind, TrainingJobConfig, TrainingJobManager, TrainingOperation,
)
from Rock_AI.training_jobs.training_job_manifest import TrainingJobManifest
from Rock_AI.training_jobs.training_job_worker import run_training_job


def _job_config(root, output, operation=TrainingOperation.NEW_RUN, **changes):
    values = dict(
        operation=operation,
        source_run="",
        output_run=str(output.relative_to(root)),
        additional_generations=1,
        seed=4242,
        population_size=4,
        training_scenarios=1,
        validation_scenarios=1,
        trainer_kind=TrainingBackendKind.FULL_FARMER,
        worlds_per_genome=1,
        max_rounds_per_world=1,
    )
    values.update(changes)
    return TrainingJobConfig(**values)


def test_full_farmer_worker_can_continue_one_generation(tmp_path):
    root = tmp_path
    output = root / "training_runs" / "durable_full_farmer"
    manager = TrainingJobManager(root, root / "jobs")
    first = manager.create_job(_job_config(root, output), job_id="job_full_first")
    first_status = run_training_job(first.job_directory)
    checkpoint = output / "checkpoints" / "neat-checkpoint-1"

    assert first_status.status.value == "completed"
    assert checkpoint.exists()
    assert (output / "champions" / "generation_0000" / "network.json").exists()

    second_config = _job_config(
        root,
        output,
        TrainingOperation.CONTINUE,
        source_run=str(output.relative_to(root)),
        source_checkpoint=str(checkpoint.relative_to(root)),
    )
    second = manager.create_job(second_config, job_id="job_full_second")
    second_status = run_training_job(second.job_directory)

    assert second_status.status.value == "completed"
    assert second_status.starting_generation == 1
    assert (output / "champions" / "generation_0001" / "network.json").exists()
    state = json.loads((output / "full_farmer_training_state.json").read_text(encoding="utf-8"))
    assert state["generation"] == 1
    assert len(state["champion_archive"]) == 2

    branch_output = root / "training_runs" / "continued_as_new"
    branch_config = _job_config(
        root,
        branch_output,
        TrainingOperation.CONTINUE_AS_BRANCH,
        source_run=str(output.relative_to(root)),
        source_checkpoint=str((output / "checkpoints" / "neat-checkpoint-2").relative_to(root)),
    )
    branch = manager.create_job(branch_config, job_id="job_full_continue_as_new")
    branch_status = run_training_job(branch.job_directory)
    assert branch_status.starting_generation == 2
    assert (branch_output / "champions" / "generation_0002" / "network.json").exists()
    branch_state = json.loads((branch_output / "full_farmer_training_state.json").read_text(encoding="utf-8"))
    assert len(branch_state["champion_archive"]) == 3


def test_curriculum_requires_stable_validation_and_low_invalid_rate(tmp_path):
    config = FullFarmerTrainingConfig(
        str(tmp_path / "run"), population=4, generations=1, worlds_per_genome=1,
        max_rounds_per_world=1, minimum_generations_per_stage=2,
        curriculum_stability_window=2, curriculum_validation_threshold=.1,
        curriculum_invalid_rate_threshold=.05,
    )
    state = FullFarmerTrainingState(1, ActionCurriculumStage.IMPORTS, 0, [.2, .21], [.01, .02])
    assert should_advance_curriculum(state, config)
    state.invalid_rate_history[-1] = .2
    assert not should_advance_curriculum(state, config)


def test_manifest_v1_defaults_to_breeding_backend():
    payload = {
        "job_id": "job_old", "created_at": "now", "repository_root": ".",
        "job_directory": "jobs/job_old", "command": ["python"], "manifest_version": 1,
        "config": {
            "operation": "branch_champion", "source_run": "training_runs/old",
            "output_run": "training_runs/new", "additional_generations": 1, "seed": 1,
            "source_champion": "training_runs/old/network.json",
        },
    }
    manifest = TrainingJobManifest.from_dict(payload)
    assert manifest.config.trainer_kind is TrainingBackendKind.BREEDING_PAIR
    assert manifest.manifest_version == 2
