"""Fixed-size farm encoding built on the authoritative breeding validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_AI.representations.encoding_schema_helper import EncodingSchema, get_default_encoding_schema
from Rock_AI.representations.rock_encoder_helper import encode_rock


class _RockLookup:
    def __init__(self, rocks: Iterable[genetics.Rock]):
        self.rocks = {int(rock.id): rock for rock in rocks}

    def get_rock(self, rock_id: int) -> genetics.Rock | None:
        return self.rocks.get(int(rock_id))


def _extract_rocks(farm: object) -> list[genetics.Rock]:
    source = getattr(farm, "rocks", None)
    if source is None:
        source = getattr(farm, "rock_list", None)
    if source is None and isinstance(farm, (Mapping, list, tuple)):
        source = farm
    if source is None:
        raise TypeError("farm must expose rocks or rock_list, or be a rock collection")
    values = source.values() if isinstance(source, Mapping) else source
    return list(values)


@dataclass(frozen=True)
class EncodedFarm:
    rock_feature_matrix: np.ndarray
    rock_presence_mask: np.ndarray
    legal_breeding_pair_mask: np.ndarray
    global_farm_features: np.ndarray
    original_rock_id_ordering: tuple[int | str | None, ...]
    rock_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    schema_version: int


def encode_farm(
    farm: object,
    max_rocks: int,
    schema: EncodingSchema | None = None,
    *,
    breeding_master: breeding.BreedingMaster | None = None,
    game: object | None = None,
    overflow: str = "error",
) -> EncodedFarm:
    """Encode a farm, deterministically sorted by rock ID and zero padded."""

    if max_rocks <= 0:
        raise ValueError("max_rocks must be positive")
    if overflow not in {"error", "truncate"}:
        raise ValueError("overflow must be 'error' or 'truncate'")

    schema = schema or get_default_encoding_schema()
    rocks = sorted(
        _extract_rocks(farm),
        key=lambda rock: (0, int(rock.id)) if isinstance(rock.id, int) else (1, str(rock.id)),
    )
    if len(rocks) > max_rocks:
        if overflow == "error":
            raise ValueError(f"Farm has {len(rocks)} rocks but max_rocks is {max_rocks}")
        rocks = rocks[:max_rocks]

    width = len(schema.rock_matrix_feature_names)
    matrix = np.zeros((max_rocks, width), dtype=np.float64)
    presence = np.zeros(max_rocks, dtype=np.bool_)
    ids: list[int | str | None] = [None] * max_rocks
    for row, rock in enumerate(rocks):
        encoded = encode_rock(rock, schema)
        matrix[row] = encoded.as_feature_vector()
        presence[row] = True
        ids[row] = encoded.rock_id

    pair_mask = np.zeros((max_rocks, max_rocks), dtype=np.bool_)
    validator = breeding_master or breeding.BreedingMaster()
    lookup = game or (_RockLookup(rocks) if rocks else None)
    for left in range(len(rocks)):
        for right in range(left + 1, len(rocks)):
            result = validator.validate_breeding_pair(
                rocks[left],
                rocks[right],
                game=lookup,
                warn_relatedness=False,
            )
            pair_mask[left, right] = pair_mask[right, left] = bool(result["valid"])

    money = float(getattr(farm, "money", getattr(getattr(farm, "inventory", None), "money", 0)) or 0)
    generation = float(getattr(farm, "generation", 0) or 0)
    active_count = sum(rock.status == genetics.RockStatus.ACTIVE for rock in rocks)
    global_features = np.asarray(
        (
            money / schema.money_scale,
            generation / schema.generation_scale,
            len(rocks) / max_rocks,
            active_count / len(rocks) if rocks else 0.0,
        ),
        dtype=np.float64,
    )
    return EncodedFarm(
        rock_feature_matrix=matrix,
        rock_presence_mask=presence,
        legal_breeding_pair_mask=pair_mask,
        global_farm_features=global_features,
        original_rock_id_ordering=tuple(ids),
        rock_feature_names=schema.rock_matrix_feature_names,
        global_feature_names=schema.farm_global_feature_names,
        schema_version=schema.version,
    )
