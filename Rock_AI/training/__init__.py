"""Configuration objects for future Rock AI model training."""

from Rock_AI.training.training_config_helper import (
    PairRankerTrainingConfig,
    PairRankingDataConfig,
    PredictorTrainingConfig,
    TrainingDataConfig,
)

__all__ = [
    "PairRankerTrainingConfig",
    "PairRankingDataConfig",
    "PredictorTrainingConfig",
    "TrainingDataConfig",
    "train_breeding_predictor",
    "train_pair_ranker",
]


def __getattr__(name: str):
    if name == "train_breeding_predictor":
        from Rock_AI.training.train_breeding_predictor import train_breeding_predictor

        return train_breeding_predictor
    if name == "train_pair_ranker":
        from Rock_AI.training.train_pair_ranker import train_pair_ranker

        return train_pair_ranker
    raise AttributeError(name)
