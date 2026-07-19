from Rock_AI.training_jobs import TrainingJobConfig,TrainingJobManager,TrainingOperation
from Rock_AI.training_jobs.training_job_status import TrainingJobState
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader,atomic_write_json


def test_recovery_creates_new_job_and_preserves_original(tmp_path):
    manager=TrainingJobManager(tmp_path,jobs_root=tmp_path/"jobs")
    config=TrainingJobConfig(operation=TrainingOperation.CONTINUE,source_run="training_runs/source",output_run="training_runs/source",additional_generations=1,seed=2,source_checkpoint="latest")
    original=manager.create_job(config)
    directory=manager.jobs_root/original.job_id
    status=TrainingProgressReader(directory).status().transition(TrainingJobState.VALIDATING).transition(TrainingJobState.QUEUED).transition(TrainingJobState.STARTING).transition(TrainingJobState.RUNNING)
    status=status.transition(TrainingJobState.ORPHANED,latest_checkpoint="training_runs/source/checkpoints/neat-checkpoint-4")
    atomic_write_json(directory/"status.json",status.to_dict())
    recovered=manager.recover(original.job_id)
    assert recovered.job_id!=original.job_id
    assert TrainingProgressReader(directory).status().status is TrainingJobState.ORPHANED
    assert recovered.config.operation is TrainingOperation.CONTINUE_AS_BRANCH
