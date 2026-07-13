"""Configuration objects for future Rock AI model training."""

from Rock_AI.training.training_config_helper import PredictorTrainingConfig, TrainingDataConfig

__all__ = ["PredictorTrainingConfig", "TrainingDataConfig", "train_breeding_predictor"]


def __getattr__(name: str):
    if name == "train_breeding_predictor":
        from Rock_AI.training.train_breeding_predictor import train_breeding_predictor

        return train_breeding_predictor
    raise AttributeError(name)
