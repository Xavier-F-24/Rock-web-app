"""Request cooperative cancellation of one local training job."""

import argparse
from pathlib import Path
from Rock_AI.training_jobs.training_job_manager import TrainingJobManager

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--job", required=True); args=parser.parse_args(argv)
    job=Path(args.job).resolve(); manager=TrainingJobManager(Path.cwd(), jobs_root=job.parent); print(manager.request_cancel(job.name))
if __name__ == "__main__": main()
