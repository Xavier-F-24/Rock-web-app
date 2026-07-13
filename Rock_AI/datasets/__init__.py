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
    "ScalarEstimate",
]


def __getattr__(name: str):
    if name == "BreedingDatasetGenerator":
        from Rock_AI.datasets.breeding_dataset_generator import BreedingDatasetGenerator

        return BreedingDatasetGenerator
    raise AttributeError(name)
