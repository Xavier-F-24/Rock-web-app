"""World manager placeholders for NPC farm lifecycle operations.

Future steps will create starter farms, advance NPC generations, refresh market
listings, and create player-facing messages here.
"""

from __future__ import annotations

from Rock_World.rock_farm_profile_helper import create_default_farm_profiles
from Rock_World.rock_world_state_helper import FarmState, WorldState


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
