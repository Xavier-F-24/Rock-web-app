"""Serializable records and reproducible dataset generation."""

from Rock_AI.datasets.breeding_record_helper import BreedingRecord, EncodedBreedingRules

__all__ = ["BreedingDatasetGenerator", "BreedingRecord", "EncodedBreedingRules"]


def __getattr__(name: str):
    if name == "BreedingDatasetGenerator":
        from Rock_AI.datasets.breeding_dataset_generator import BreedingDatasetGenerator

        return BreedingDatasetGenerator
    raise AttributeError(name)
