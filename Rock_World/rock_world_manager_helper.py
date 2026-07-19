"""Deterministic starter-world construction and global rock IDs."""

from __future__ import annotations

import itertools
import random

from Rock_GameState.rock_game_state_helper import GameMaster

from .rock_farm_profile_helper import FarmObjective, FarmProfile, create_default_farm_profiles, create_farm_profiles
from .rock_world_state_helper import FarmerControllerSpec, FarmState, WorldState


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


def _simulate_lineage(game: GameMaster, generations: int) -> None:
    for _ in range(generations):
        active = sorted((rock for rock in game.rocks.values() if rock.is_active), key=lambda rock: rock.id)
        pairs = []
        used = set()
        for left, right in itertools.combinations(active, 2):
            if left.id in used or right.id in used:
                continue
            validation = game.breeding_master.validate_breeding_pair(left, right, game=game, warn_relatedness=False)
            if validation["valid"]:
                pairs.append((left.id, right.id))
                used.update((left.id, right.id))
            if len(pairs) >= game.max_pairs_per_generation:
                break
        for left_id, right_id in pairs:
            game.add_pair_to_queue(left_id, right_id)
        if pairs:
            game.advance_generation()
        else:
            game.generation += 1


def create_playable_world(
    player_game: GameMaster,
    *,
    seed: int,
    npc_count: int,
    starting_money: int = 40,
    allow_neural_farmers: bool = True,
    clear_legacy_market: bool = True,
) -> WorldState:
    if not 2 <= int(npc_count) <= 12:
        raise ValueError("Playable worlds require between 2 and 12 NPC farmers")
    rng = random.Random(int(seed))
    if clear_legacy_market:
        player_game.market_pods = []
        player_game.pending_market_pod = None
    player_profile = FarmProfile("player", "Your Rock Farm", FarmObjective.PLAYER)
    farms = {
        "player": FarmState(
            "player", player_profile, player_game, visible_rock_ids=set(player_game.rock_list),
            controller=FarmerControllerSpec("player", int(seed)),
        )
    }
    owners = {rock_id: "player" for rock_id in player_game.rock_list}
    next_rock_id = max(owners, default=0) + 1
    profiles = create_farm_profiles(int(npc_count), rng)
    for index, profile in enumerate(profiles):
        ahead = rng.randint(1, 3)
        farm_seed = int(seed) + 1000 + index * 97
        game = GameMaster(
            starting_money=starting_money,
            max_generation=player_game.max_generation + ahead + 3,
            max_pairs_per_generation=player_game.max_pairs_per_generation,
            seed=farm_seed,
        )
        _simulate_lineage(game, player_game.generation + ahead)
        game.market_pods = []
        game.pending_market_pod = None
        next_rock_id = _rebase_rocks(game, next_rock_id)
        farm_id = f"farm:{index + 1}"
        policy_id = "auto" if allow_neural_farmers else "heuristic"
        farms[farm_id] = FarmState(
            farm_id, profile, game, visible_rock_ids=set(game.rock_list),
            controller=FarmerControllerSpec(policy_id, farm_seed),
        )
        owners.update({rock_id: farm_id for rock_id in game.rock_list})
    world = WorldState(farms, owners, int(seed), generation=player_game.generation, resolved_npc_count=int(npc_count))
    player_game.world = world
    return world
