"""World manager helpers for NPC farm lifecycle operations.

Future steps will create starter farms, advance NPC generations, refresh market
listings, and create player-facing messages here.
"""

from __future__ import annotations

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster, Inventory
from Rock_World.rock_farm_profile_helper import FarmProfile, create_default_farm_profiles
from Rock_World.rock_world_state_helper import FarmState, NPC_ROCK_ID_START, WorldState, get_rock_owner


def create_empty_default_world() -> WorldState:
    """
    Create a world with default farm profiles and no rocks yet.

    This keeps the first integration step data-only while making the next step
    straightforward: farm rock generation can fill these FarmState containers.
    """

    world = WorldState()
    for profile in create_default_farm_profiles():
        world.add_farm(FarmState(profile=profile))
    return world


def create_starter_world(
    game: GameMaster,
    profiles: list[FarmProfile] | None = None,
    founder_count: int = 6,
) -> WorldState:
    """
    Create a starter NPC world using the active player's game services.

    The returned world is not attached to ``game`` yet. It is a separate core
    state object that uses globally unique rock IDs above the player's current
    ID range, so the next integration step can add explicit transfer helpers.
    """

    profiles = profiles or create_default_farm_profiles()
    world = WorldState(
        world_generation=int(getattr(game, "generation", 0)),
        next_world_rock_id=first_world_rock_id(game),
    )

    for profile in profiles:
        farm = FarmState(
            profile=profile,
            inventory=Inventory(money=profile.starting_money),
        )
        create_founder_rocks(game, world, farm, founder_count=founder_count)
        advance_farm_starter_generations(game, world, farm, profile.starting_generation_offset)
        farm.events.append(
            f"{profile.farm_name} opened with {len(farm.rocks)} farm-owned rock(s)."
        )
        world.add_farm(farm)

    return world


def game_has_world(game: GameMaster) -> bool:
    return getattr(game, "world", None) is not None


def attach_starter_world(
    game: GameMaster,
    profiles: list[FarmProfile] | None = None,
    founder_count: int = 6,
) -> WorldState:
    if game_has_world(game):
        return game.world

    game.world = create_starter_world(
        game=game,
        profiles=profiles,
        founder_count=founder_count,
    )
    return game.world


def get_or_create_world(game: GameMaster) -> WorldState:
    return attach_starter_world(game)


def first_world_rock_id(game: GameMaster) -> int:
    player_ids = [int(rock_id) for rock_id in getattr(game, "rocks", {})]
    next_player_id = int(getattr(game, "next_rock_id", 1))
    return max(NPC_ROCK_ID_START, max([next_player_id, *player_ids], default=0) + 1)


def create_founder_rocks(
    game: GameMaster,
    world: WorldState,
    farm: FarmState,
    founder_count: int = 6,
) -> list[genetics.Rock]:
    founders = []
    sexes = [genetics.Sex.MALE, genetics.Sex.FEMALE]
    for index in range(max(2, int(founder_count))):
        rock = game.market_manager.make_random_rock(
            rock_id=world.reserve_world_rock_id(),
            sex=sexes[index % len(sexes)],
            generation=0,
            market_guest=False,
        )
        finalize_farm_rock(game, farm, rock)
        founders.append(rock)

    farm.events.append(f"Created {len(founders)} founder rock(s).")
    return founders


def advance_farm_starter_generations(
    game: GameMaster,
    world: WorldState,
    farm: FarmState,
    generation_count: int,
) -> list[genetics.Rock]:
    all_children: list[genetics.Rock] = []
    for _ in range(max(0, int(generation_count))):
        pairs = choose_starter_pairs(farm)
        if not pairs:
            break

        next_generation = farm.generation + 1
        generation_children: list[genetics.Rock] = []
        for parent_a, parent_b in pairs:
            clutch = game.breeding_master.breed_parent_set(
                parent_a=parent_a,
                parent_b=parent_b,
                next_id=world.next_world_rock_id,
                child_generation=next_generation,
                death_chance=0,
                craisen_chance=0,
                mutation_chance=0,
                spore_death_chance=0,
            )
            parent_a.change_status(genetics.RockStatus.ACTIVE)
            parent_b.change_status(genetics.RockStatus.ACTIVE)
            for child in clutch:
                child.id = world.reserve_world_rock_id()
                child.generation = next_generation
                finalize_farm_rock(game, farm, child)
                generation_children.append(child)

        if not generation_children:
            break

        farm.generation = next_generation
        farm.events.append(
            f"Advanced starter farm to generation {farm.generation} with "
            f"{len(generation_children)} child rock(s)."
        )
        all_children.extend(generation_children)

    return all_children


def choose_starter_pairs(farm: FarmState, max_pairs: int = 3) -> list[tuple[genetics.Rock, genetics.Rock]]:
    active = [
        rock
        for rock in farm.rocks.values()
        if rock.status == genetics.RockStatus.ACTIVE
    ]
    males = [rock for rock in active if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in active if rock.sex == genetics.Sex.FEMALE]
    return list(zip(males, females))[: max(0, int(max_pairs))]


def finalize_farm_rock(game: GameMaster, farm: FarmState, rock: genetics.Rock) -> genetics.Rock:
    rock.is_market = False
    rock.change_status(genetics.RockStatus.ACTIVE)
    game.finalize_rock(rock)
    farm.add_rock(rock)
    if get_rock_owner(rock) != farm.owner_id:
        raise ValueError(f"Farm rock #{rock.id} was not assigned to {farm.owner_id}.")
    return rock
