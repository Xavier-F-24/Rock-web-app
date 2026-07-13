from __future__ import annotations

import numpy as np
import pytest

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.representations.encoding_schema_helper import get_default_encoding_schema
from Rock_AI.representations.farm_encoder_helper import encode_farm
from Rock_AI.representations.rock_encoder_helper import encode_parent_pair, encode_rock
from Rock_GameState.rock_game_state_helper import GameMaster


def _parent_pair(game: GameMaster):
    male = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE)
    female = next(rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE)
    return male, female


def test_schema_and_allele_encoding_are_stable_and_retain_both_alleles():
    game = GameMaster(seed=11)
    rock, _ = _parent_pair(game)
    gene_name = get_default_encoding_schema().gene_names[0]
    rock.genotype.genes[gene_name].allele_a = genetics.Allele(0)
    rock.genotype.genes[gene_name].allele_b = genetics.Allele(1)

    encoded = encode_rock(rock)
    offset = encoded.genotype_feature_names.index(f"{gene_name}.allele_a")

    assert encoded.genotype_features[offset : offset + 4].tolist() == [0.0, 1.0, 0.0, 1.0]
    death_name = get_default_encoding_schema().death_gene_names[0]
    death_offset = encoded.genotype_feature_names.index(f"{death_name}.allele_a")
    death_pair = rock.death_genes.genes[death_name]
    assert encoded.genotype_features[death_offset : death_offset + 2].tolist() == [
        float(death_pair.allele_a.value),
        float(death_pair.allele_b.value),
    ]
    assert get_default_encoding_schema().gene_names == tuple(sorted(genetics.GENE_SPECS))


def test_farm_encoding_zero_pads_and_preserves_rock_ids():
    game = GameMaster(seed=12)
    encoded = encode_farm(game, max_rocks=6)

    assert encoded.rock_feature_matrix.shape[0] == 6
    assert encoded.rock_presence_mask.tolist() == [True, True, True, True, False, False]
    assert encoded.original_rock_id_ordering[:4] == (1, 2, 3, 4)
    assert encoded.original_rock_id_ordering[4:] == (None, None)
    assert np.all(encoded.rock_feature_matrix[4:] == 0)


def test_parent_pair_encoding_preserves_order_and_trace_ids():
    parents = _parent_pair(GameMaster(seed=15))
    encoded = encode_parent_pair(*parents)

    assert encoded.parent_ids == (parents[0].id, parents[1].id)
    assert encoded.parent_feature_matrix.shape == (2, len(encoded.feature_names))


def test_farm_legal_pair_mask_matches_authoritative_validation():
    game = GameMaster(seed=13)
    encoded = encode_farm(game, max_rocks=4, game=game)
    ids = encoded.original_rock_id_ordering

    for left in range(4):
        for right in range(4):
            if left == right:
                assert not encoded.legal_breeding_pair_mask[left, right]
                continue
            expected = game.breeding_master.validate_breeding_pair(
                game.get_rock(ids[left]),
                game.get_rock(ids[right]),
                game=game,
                warn_relatedness=False,
            )["valid"]
            assert bool(encoded.legal_breeding_pair_mask[left, right]) is expected


def test_farm_encoding_rejects_implicit_overflow():
    with pytest.raises(ValueError, match="max_rocks"):
        encode_farm(GameMaster(seed=14), max_rocks=2)
