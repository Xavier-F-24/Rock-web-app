"""Deterministic starter-world construction and global rock IDs."""

from __future__ import annotations

import random

from Rock_GameState.rock_game_state_helper import GameMaster

from .rock_farm_profile_helper import create_default_farm_profiles
from .rock_world_state_helper import FarmState, WorldState


def _rebase_rocks(game: GameMaster, next_id: int) -> int:
    mapping = {old: next_id + index for index, old in enumerate(sorted(game.rock_list))}
    rebased = {}
    for old_id, rock in game.rock_list.items():
        rock.id = mapping[old_id]
        rock.parent_ids = [mapping.get(parent_id, parent_id) for parent_id in rock.parent_ids]
        rebased[rock.id] = rock
    game.rock_list = rebased
    game.next_rock_id = max(rebased, default=next_id - 1) + 1
    return game.next_rock_id


def create_starter_world(
    *, seed: int = 0, player_generation: int = 0, starting_money: int = 40,
    farm_count: int = 3,
) -> WorldState:
    rng = random.Random(seed)
    profiles = create_default_farm_profiles()[:farm_count]
    farms: dict[str, FarmState] = {}
    owners: dict[int, str] = {}
    next_rock_id = 1
    for index, profile in enumerate(profiles):
        ahead = rng.choice((1, 2))
        game = GameMaster(starting_money=starting_money, seed=seed + 1000 + index)
        game.generation = player_generation + ahead
        for rock in game.rock_list.values():
            rock.generation = game.generation
        next_rock_id = _rebase_rocks(game, next_rock_id)
        farm_id = f"farm:{index + 1}"
        farm = FarmState(farm_id, profile, game, visible_rock_ids=set(game.rock_list))
        farms[farm_id] = farm
        owners.update({rock_id: farm_id for rock_id in game.rock_list})
    return WorldState(farms, owners, seed, generation=player_generation)
