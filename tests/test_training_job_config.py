from pathlib import Path

import pytest

from Rock_AI.training_jobs import TrainingJobConfig, TrainingOperation
from Rock_AI.training_jobs.training_job_config import TrainingSafetyTier


def _branch(**changes):
    values = dict(operation=TrainingOperation.BRANCH_CHAMPION, source_run="training_runs/source", output_run="training_runs/branch", additional_generations=2, seed=1, source_champion="training_runs/source/network.json")
    values.update(changes); return TrainingJobConfig(**values)


def test_smoke_limits_are_hard_bounds():
    with pytest.raises(ValueError, match="Smoke Training"):
        _branch(population_size=21)


def test_advanced_requires_acknowledgement():
    with pytest.raises(ValueError, match="acknowledgement"):
        _branch(safety_tier=TrainingSafetyTier.ADVANCED)


def test_paths_cannot_escape_repository(tmp_path):
    config = _branch(output_run="../escape")
    with pytest.raises(ValueError, match="inside the repository"):
        config.validate_paths(tmp_path)


def test_branch_cannot_overwrite_parent(tmp_path):
    config = _branch(output_run="training_runs/source")
    with pytest.raises(ValueError, match="cannot overwrite"):
        config.validate_paths(tmp_path)


def test_config_round_trip():
    config = _branch()
    assert TrainingJobConfig.from_dict(config.to_dict()) == config
