from __future__ import annotations

import numpy as np

from Rock_AI.datasets.pair_ranking_dataset_generator import PairRankingDatasetGenerator
from Rock_AI.datasets.pair_ranking_storage_helper import load_pair_ranking_split, save_pair_ranking_dataset
from Rock_AI.training.training_config_helper import PairRankingDataConfig


def _config(tmp_path, seed=88):
    return PairRankingDataConfig(
        number_of_farms=3,
        trials_per_pair=1,
        seed=seed,
        output_directory=str(tmp_path),
        minimum_rocks=4,
        maximum_rocks=4,
    )


def test_dataset_is_deterministic_excludes_invalid_pairs_and_preserves_group_boundaries(tmp_path):
    first_generator = PairRankingDatasetGenerator(_config(tmp_path / "first"))
    first = first_generator.generate()
    second = PairRankingDatasetGenerator(_config(tmp_path / "second")).generate()
    assert [[candidate.parent_ids for candidate in group.candidates] for group in first] == [
        [candidate.parent_ids for candidate in group.candidates] for group in second
    ]
    assert np.allclose(
        [candidate.utility_score for group in first for candidate in group.candidates],
        [candidate.utility_score for group in second for candidate in group.candidates],
    )
    for group in first:
        assert len(group.candidates) == 4  # two males x two females; same-sex pairs are excluded
    splits = first_generator.split_groups(first)
    lineage_sets = [{group.lineage_group_id for group in split} for split in splits.values()]
    assert not (lineage_sets[0] & lineage_sets[1] or lineage_sets[0] & lineage_sets[2] or lineage_sets[1] & lineage_sets[2])
    save_pair_ranking_dataset(tmp_path / "saved", splits, first_generator.manifest())
    arrays, metadata, manifest = load_pair_ranking_split(tmp_path / "saved", "train")
    assert arrays["group_offsets"].tolist() == [0, 4]
    assert arrays["candidate_mask"].all()
    assert metadata[0]["parent_ids"]
    assert manifest["dimensions"]["parent"] == 142


def test_different_seed_changes_generated_labels(tmp_path):
    first = PairRankingDatasetGenerator(_config(tmp_path / "a", 1)).generate()
    second = PairRankingDatasetGenerator(_config(tmp_path / "b", 2)).generate()
    first_values = [candidate.utility_score for group in first for candidate in group.candidates]
    second_values = [candidate.utility_score for group in second for candidate in group.candidates]
    assert first_values != second_values
