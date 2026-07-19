from Rock_AI.training_jobs import TrainingJobConfig,TrainingJobManager,TrainingOperation
from Rock_AI.training_jobs.training_job_status import TrainingJobState
from Rock_AI.training_jobs.training_job_worker import run_training_job


def test_prestart_cancellation_finishes_without_loading_checkpoint(tmp_path):
    manager=TrainingJobManager(tmp_path,jobs_root=tmp_path/"jobs")
    config=TrainingJobConfig(operation=TrainingOperation.CONTINUE,source_run="training_runs/missing",output_run="training_runs/output",additional_generations=1,seed=2,source_checkpoint="latest")
    manifest=manager.create_job(config)
    manager.request_cancel(manifest.job_id)
    status=run_training_job(manifest.job_directory)
    assert status.status is TrainingJobState.CANCELLED
    assert status.cancellation_state=="before_start"
    assert not (tmp_path/"training_runs/output").exists()
