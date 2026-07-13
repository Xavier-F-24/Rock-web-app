from __future__ import annotations

from pathlib import Path

import pytest

from Rock_AI.datasets.breeding_record_helper import EncodedBreedingRules
from Rock_AI.evaluation.predictor_evaluator import BreedingPredictor
from Rock_AI.training.train_breeding_predictor import train_breeding_predictor
from Rock_AI.training.training_config_helper import PredictorTrainingConfig
from Rock_GameState.rock_game_state_helper import GameMaster


def test_small_training_reduces_loss_and_infers_from_real_rocks(tmp_path):
    dataset = Path(__file__).resolve().parents[2] / "training_data" / "breeding_predictor_smoke"
    if not (dataset / "manifest.json").exists():
        pytest.skip("Smoke dataset has not been generated")
    config = PredictorTrainingConfig(
        dataset_path=str(dataset),
        output_directory=str(tmp_path / "run"),
        seed=7100,
        batch_size=8,
        learning_rate=3e-3,
        number_of_epochs=6,
        encoder_hidden_dimensions=(24,),
        trunk_hidden_dimensions=(24,),
        parent_embedding_dimension=12,
        rule_embedding_dimension=6,
        context_embedding_dimension=4,
        dropout=0.0,
        early_stopping_patience=0,
        device="cpu",
    )
    result = train_breeding_predictor(config)
    losses = [row["train"]["total_loss"] for row in result["history"].epochs]

    assert min(losses[1:]) < losses[0]
    assert result["best_checkpoint"].exists()
    assert result["latest_checkpoint"].exists()

    predictor = BreedingPredictor.load(result["best_checkpoint"])
    game = GameMaster(seed=7110)
    prediction = predictor.predict(
        game.get_rock(1),
        game.get_rock(2),
        EncodedBreedingRules(),
    )

    assert prediction["parent_ids"] == [1, 2]
    assert "expected_raw_clutch_size" in prediction["scalar_predictions"]
    assert prediction["genotype_distributions"]
    assert all(
        abs(sum(values.values()) - 1.0) < 1e-5
        for values in prediction["genotype_distributions"].values()
    )
