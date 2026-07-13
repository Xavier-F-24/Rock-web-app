"""Serializable records and reproducible dataset generation."""

from Rock_AI.datasets.breeding_record_helper import BreedingRecord, EncodedBreedingRules
from Rock_AI.datasets.breeding_expectation_record import (
    BreedingExpectationRecord,
    GeneOutcomeDistribution,
    ScalarEstimate,
)

__all__ = [
    "BreedingDatasetGenerator",
    "BreedingExpectationRecord",
    "BreedingRecord",
    "EncodedBreedingRules",
    "GeneOutcomeDistribution",
    "PredictorDatasetGenerator",
    "PredictorDatasetSplits",
    "PredictorExample",
    "PredictorTargetSchema",
    "ScalarEstimate",
    "save_predictor_dataset",
    "split_predictor_examples",
]


def __getattr__(name: str):
    if name == "BreedingDatasetGenerator":
        from Rock_AI.datasets.breeding_dataset_generator import BreedingDatasetGenerator

        return BreedingDatasetGenerator
    if name in {"PredictorExample", "PredictorTargetSchema"}:
        from Rock_AI.datasets import predictor_example_helper

        return getattr(predictor_example_helper, name)
    if name == "PredictorDatasetGenerator":
        from Rock_AI.datasets.predictor_dataset_generator import PredictorDatasetGenerator

        return PredictorDatasetGenerator
    if name in {"PredictorDatasetSplits", "split_predictor_examples"}:
        from Rock_AI.datasets import dataset_split_helper

        return getattr(dataset_split_helper, name)
    if name == "save_predictor_dataset":
        from Rock_AI.datasets.dataset_storage_helper import save_predictor_dataset

        return save_predictor_dataset
    raise AttributeError(name)
