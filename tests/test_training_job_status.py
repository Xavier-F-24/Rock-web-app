import pytest

from Rock_AI.training_jobs.training_job_status import TrainingJobState,TrainingJobStatus


def test_legal_status_transitions_are_enforced():
    created=TrainingJobStatus("job",TrainingJobState.CREATED,"continue")
    assert created.transition(TrainingJobState.VALIDATING).status is TrainingJobState.VALIDATING
    with pytest.raises(ValueError,match="Illegal"):
        created.transition(TrainingJobState.COMPLETED)


def test_generation_and_operation_progress_are_bounded():
    status = TrainingJobStatus(
        "job", TrainingJobState.RUNNING, "continue",
        starting_generation=5, requested_ending_generation=14,
        completed_generations=3, operation_progress_current=7,
        operation_progress_total=10,
    )
    generation = status.generation_progress
    operation = status.operation_progress
    assert generation[:2] == (3, 10) and generation[2] == pytest.approx(0.3)
    assert operation is not None
    assert operation[:2] == (7, 10) and operation[2] == pytest.approx(0.7)
    completed = status.transition(
        TrainingJobState.RUNNING,
        completed_generations=99,
        operation_progress_current=99,
    )
    assert completed.generation_progress == (10, 10, 1.0)


def test_oracle_operation_details_round_trip():
    status = TrainingJobStatus(
        "job",
        TrainingJobState.RUNNING,
        "continue",
        operation_elapsed_seconds=12.5,
        operation_eta_seconds=7.25,
        current_candidate_parent_ids=("10", "11"),
    )

    restored = TrainingJobStatus.from_dict(status.to_dict())

    assert restored.operation_elapsed_seconds == pytest.approx(12.5)
    assert restored.operation_eta_seconds == pytest.approx(7.25)
    assert restored.current_candidate_parent_ids == ("10", "11")
