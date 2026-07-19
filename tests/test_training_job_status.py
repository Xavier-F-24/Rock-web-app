import pytest

from Rock_AI.training_jobs.training_job_status import TrainingJobState,TrainingJobStatus


def test_legal_status_transitions_are_enforced():
    created=TrainingJobStatus("job",TrainingJobState.CREATED,"continue")
    assert created.transition(TrainingJobState.VALIDATING).status is TrainingJobState.VALIDATING
    with pytest.raises(ValueError,match="Illegal"):
        created.transition(TrainingJobState.COMPLETED)
