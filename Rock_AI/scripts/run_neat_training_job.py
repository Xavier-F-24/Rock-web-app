"""Run one manifest-driven NEAT training job in an isolated local process."""

from __future__ import annotations

import argparse
from pathlib import Path

from Rock_AI.training_jobs.training_job_config import TrainingJobConfig, TrainingOperation
from Rock_AI.training_jobs.training_job_manager import TrainingJobManager
from Rock_AI.training_jobs.training_job_worker import run_training_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job")
    parser.add_argument("--operation", choices=[row.value for row in TrainingOperation])
    parser.add_argument("--source-run"); parser.add_argument("--checkpoint")
    parser.add_argument("--generation", type=int); parser.add_argument("--champion")
    parser.add_argument("--population", type=int, default=20); parser.add_argument("--additional-generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--output-run")
    args = parser.parse_args(argv)
    if args.job:
        status = run_training_job(args.job)
    else:
        if not all((args.operation, args.source_run, args.output_run)): parser.error("Direct mode requires operation, source-run, and output-run")
        root = Path.cwd()
        manager = TrainingJobManager(root)
        manifest = manager.create_job(TrainingJobConfig(
            operation=TrainingOperation(args.operation), source_run=args.source_run,
            output_run=args.output_run, additional_generations=args.additional_generations,
            seed=args.seed, source_checkpoint=args.checkpoint,
            source_generation=args.generation, source_champion=args.champion,
            population_size=args.population,
        ))
        status = run_training_job(manifest.job_directory)
    print(f"{status.job_id}: {status.status.value}")


if __name__ == "__main__": main()
