"""Fixed-seed mixed, scarcity, and complementary-trade templates."""

from dataclasses import dataclass
from enum import Enum

from Rock_World.rock_world_manager_helper import create_starter_world


class WorldTemplate(str, Enum):
    MIXED = "mixed"
    SCARCITY = "scarcity"
    TRADE = "trade"


@dataclass(frozen=True)
class ScenarioDefinition:
    template: WorldTemplate
    seed: int
    starting_money: int


class MultiAgentScenarioManager:
    def scenarios(self, seed: int, count: int):
        templates = tuple(WorldTemplate)
        return tuple(ScenarioDefinition(templates[index % len(templates)], seed + index * 1009, 18 if templates[index % len(templates)] == WorldTemplate.SCARCITY else 40) for index in range(count))

    @staticmethod
    def build(definition: ScenarioDefinition):
        world = create_starter_world(seed=definition.seed, starting_money=definition.starting_money)
        if definition.template == WorldTemplate.TRADE:
            # Visibility is public, while hidden genomes remain unavailable to agents.
            for farm in world.farms.values():
                farm.visible_rock_ids = set(farm.rocks)
        return world
