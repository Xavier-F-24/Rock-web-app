from pathlib import Path
from types import SimpleNamespace

import Rock_AI.training.recurrent_neat_training_helper as recurrent_training
import pytest
from Rock_AI.training.recurrent_neat_training_helper import (
    RecurrentNeatTrainer,
    RecurrentNeatTrainingConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recurrent_trainer_emits_phase_progress_and_checkpoint(tmp_path):
    events = []
    config = RecurrentNeatTrainingConfig(
        dataset_path=str(ROOT / "training_data" / "player_pair_ranker_impl_smoke"),
        output_directory=str(tmp_path / "run"),
        seed=812,
        population=4,
        generations=1,
        training_scenarios_per_generation=1,
        validation_scenarios=1,
        checkpoint_frequency=1,
        minimum_campaign_generation=999,
        heartbeat_interval_seconds=0.01,
    )
    trainer = RecurrentNeatTrainer(config, progress_callback=events.append)
    trainer.train()

    heartbeats = [row for row in events if row["event_type"] == "worker_heartbeat"]
    assert heartbeats
    assert any(row["phase"] == "genome_evaluation" for row in heartbeats)
    assert any(row.get("progress_total") == 4 for row in heartbeats)
    assert (tmp_path / "run" / "checkpoints" / "neat-checkpoint-1").exists()


def test_campaign_baselines_report_random_and_oracle_phases(tmp_path, monkeypatch):
    events = []

    class FastEvaluator:
        def __init__(self, config):
            self.config = config

        def run_episode(self, agent, **kwargs):
            return SimpleNamespace(final_farm_summary={"objective_utility": 1.0})

    monkeypatch.setattr(recurrent_training, "BreedingAgentEvaluator", FastEvaluator)
    config = RecurrentNeatTrainingConfig(
        dataset_path=str(ROOT / "training_data" / "player_pair_ranker_impl_smoke"),
        output_directory=str(tmp_path / "run"),
        population=4,
        generations=1,
        campaign_scenarios_per_generation=1,
        campaign_generations=1,
    )
    trainer = RecurrentNeatTrainer(config, progress_callback=events.append)

    trainer._campaign_inputs()

    operations = {
        row.get("operation") for row in events
        if row["event_type"] == "worker_heartbeat"
    }
    assert {"random_campaign_baseline", "oracle_campaign_baseline", "campaign_baseline_completed"} <= operations


def test_campaign_baseline_honors_cancellation_before_expensive_work(tmp_path):
    cancel = tmp_path / "cancel.request"
    cancel.write_text("cancel", encoding="ascii")
    config = RecurrentNeatTrainingConfig(
        dataset_path=str(ROOT / "training_data" / "player_pair_ranker_impl_smoke"),
        output_directory=str(tmp_path / "run"),
        population=4,
        generations=1,
    )
    trainer = RecurrentNeatTrainer(config, cancel_path=cancel)

    with pytest.raises(recurrent_training.TrainingCancelled, match="campaign baseline"):
        trainer._campaign_inputs()
