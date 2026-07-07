"""
Prototype game-state manager for the split rock game modules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Market.rock_market_helper import MarketManager


DEFAULT_STARTING_MONEY = 10
DEFAULT_MAX_GENERATION = 7
DEFAULT_MAX_PAIRS_PER_GENERATION = 3
BASE_MUTATION_CHANCE = 0.01
MUTATION_POTION_CHANCE = 0.12
BASE_CHILD_DEATH_CHANCE = 0.05
BASE_CRAISEN_CHANCE = 0.50
ANTI_CRAISEN_CHANCE = 0.0
BASE_SPORE_DEATH_CHANCE = 0.25
BASE_SPORE_CLONE_COUNT = 3
STARTER_SEXES = (
    genetics.Sex.MALE,
    genetics.Sex.FEMALE,
    genetics.Sex.MALE,
    genetics.Sex.FEMALE,
)


@dataclass
class Inventory:
    money: int = DEFAULT_STARTING_MONEY
    potions: dict[str, int] = field(default_factory=dict)
    specials: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueuedPair:
    parent_a_id: int
    parent_b_id: int
    potion_key: str | None = None


@dataclass
class GameMaster:
    """
    Main prototype coordinator.

    The game can be instantiated, given starters, bred through a queue, and
    connected to the MarketManager for pods, potions, sales, and imports.
    """

    starting_money: int = DEFAULT_STARTING_MONEY
    max_generation: int = DEFAULT_MAX_GENERATION
    max_pairs_per_generation: int = DEFAULT_MAX_PAIRS_PER_GENERATION
    seed: int | None = None
    auto_start: bool = True

    rock_list: dict[int, genetics.Rock] = field(default_factory=dict)
    inventory: Inventory = field(default_factory=Inventory)
    generation: int = 0
    next_rock_id: int = 1
    breeding_queue: list[QueuedPair] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    game_over: bool = False
    market_pods: list[Any] = field(default_factory=list)
    pending_market_pod: Any = None

    rng: random.Random = field(default_factory=random.Random)
    genome_factory: genetics.GenomeFactory = field(default_factory=genetics.GenomeFactory)
    expression_engine: genetics.ExpressionEngine = field(default_factory=genetics.ExpressionEngine)
    value_calculator: genetics.ValueCalculator = field(default_factory=genetics.ValueCalculator)
    name_generator: genetics.NameGenerator = field(default_factory=genetics.NameGenerator)
    breeding_master: breeding.BreedingMaster = field(default_factory=breeding.BreedingMaster)
    market_manager: MarketManager | None = None

    def __post_init__(self):
        self.inventory.money = self.starting_money

        if self.seed is not None:
            self.rng.seed(self.seed)
            self.genome_factory.rng.seed(self.seed + 1)
            self.name_generator.rng.seed(self.seed + 2)
            self.breeding_master.rng.seed(self.seed + 3)

        self.breeding_master.GenomeFactory = self.genome_factory
        self.breeding_master.ExpressionEngine = self.expression_engine
        self.breeding_master.ValueCalculator = self.value_calculator
        self.breeding_master.NameGenerator = self.name_generator

        if self.market_manager is None:
            self.market_manager = MarketManager(
                rng=self.rng,
                genome_factory=self.genome_factory,
                expression_engine=self.expression_engine,
                value_calculator=self.value_calculator,
                name_generator=self.name_generator,
                breeding_master=self.breeding_master,
            )

        if self.auto_start and not self.rock_list:
            self.create_starting_set()

    @property
    def rocks(self) -> dict[int, genetics.Rock]:
        return self.rock_list

    @rocks.setter
    def rocks(self, value: dict[int, genetics.Rock]) -> None:
        self.rock_list = value

    @property
    def money(self) -> int:
        return self.inventory.money

    @money.setter
    def money(self, value: int) -> None:
        self.inventory.money = int(value)

    @property
    def potions(self) -> dict[str, int]:
        return self.inventory.potions

    @potions.setter
    def potions(self, value: dict[str, int]) -> None:
        self.inventory.potions = value

    def reserve_rock_id(self) -> int:
        rock_id = self.next_rock_id
        self.next_rock_id += 1
        return rock_id

    def get_rock(self, rock_id: int | genetics.Rock | None) -> genetics.Rock | None:
        if rock_id is None:
            return None
        if hasattr(rock_id, "id"):
            rock_id = rock_id.id
        return self.rock_list.get(int(rock_id))

    def finalize_rock(self, rock: genetics.Rock) -> genetics.Rock:
        self.expression_engine.instantiate_phenotype(rock)
        self.value_calculator.set_rock_value(rock)
        return rock

    def create_starting_set(self, count: int = 4, force: bool = False) -> list[genetics.Rock]:
        if self.rock_list and not force:
            return list(self.rock_list.values())

        self.rock_list.clear()
        self.breeding_queue.clear()
        self.next_rock_id = 1
        starters = []

        for index in range(count):
            sex = STARTER_SEXES[index % len(STARTER_SEXES)]
            rock = genetics.Rock(
                id=self.reserve_rock_id(),
                name=self.name_generator.generate_name(sex=sex),
                sex=sex,
                genotype=self.genome_factory.make_random_rock_genome(),
                death_genes=self.genome_factory.make_death_genes(),
                generation=self.generation,
                status=genetics.RockStatus.ACTIVE,
            )
            self.finalize_rock(rock)
            self.rock_list[rock.id] = rock
            starters.append(rock)

        self.events.append(f"Created {len(starters)} starter rocks.")
        self.update_market(force=True)
        return starters

    def add_pair_to_queue(
        self,
        parent_a_id: int,
        parent_b_id: int,
        potion_key: str | None = None,
    ) -> QueuedPair:
        if len(self.breeding_queue) >= self.max_pairs_per_generation:
            raise ValueError("Breeding queue is full for this generation.")

        parent_a = self.get_rock(parent_a_id)
        parent_b = self.get_rock(parent_b_id)
        validation = self.breeding_master.validate_breeding_pair(parent_a, parent_b, game=self)
        if not validation["valid"]:
            raise ValueError("Invalid breeding pair: " + "; ".join(validation["errors"]))

        queued_ids = {pair.parent_a_id for pair in self.breeding_queue}
        queued_ids.update(pair.parent_b_id for pair in self.breeding_queue)
        if parent_a.id in queued_ids or parent_b.id in queued_ids:
            raise ValueError("One or both parents are already queued this generation.")

        if potion_key is not None:
            if self.potions.get(potion_key, 0) <= 0:
                raise ValueError(f"Potion not owned: {potion_key}")
            self.potions[potion_key] -= 1
            if self.potions[potion_key] <= 0:
                del self.potions[potion_key]

        pair = QueuedPair(parent_a.id, parent_b.id, potion_key)
        self.breeding_queue.append(pair)
        self.events.append(f"Queued #{parent_a.id} x #{parent_b.id}.")
        for warning in validation.get("warnings", []):
            self.events.append(warning)
        return pair

    def breed_queue(self) -> list[genetics.Rock]:
        if self.game_over:
            raise ValueError("Game is already over.")
        if not self.breeding_queue:
            return []

        children: list[genetics.Rock] = []
        next_generation = self.generation + 1

        for pair in list(self.breeding_queue):
            parent_a = self.get_rock(pair.parent_a_id)
            parent_b = self.get_rock(pair.parent_b_id)

            potion_settings = self.potion_settings(pair.potion_key)
            clutch = self.breeding_master.breed_parent_set(
                parent_a=parent_a,
                parent_b=parent_b,
                next_id=self.next_rock_id,
                child_generation=next_generation,
                mutation_chance=potion_settings["mutation_chance"],
                death_chance=potion_settings["death_chance"],
                craisen_chance=potion_settings["craisen_chance"],
                spore_death_chance=potion_settings["spore_death_chance"],
                spore_clone_count=potion_settings["spore_clone_count"],
                clutch_reroll=potion_settings["clutch_reroll"],
                clutch_plus_one=potion_settings["clutch_plus_one"],
            )

            for child in clutch:
                child.id = self.reserve_rock_id()
                child.generation = next_generation
                self.finalize_rock(child)
                self.rock_list[child.id] = child
                children.append(child)

        self.breeding_queue.clear()
        self.events.append(f"Bred {len(children)} child rock(s).")
        return children

    @staticmethod
    def potion_settings(potion_key: str | None) -> dict[str, Any]:
        mutation_chance = BASE_MUTATION_CHANCE
        death_chance = BASE_CHILD_DEATH_CHANCE
        craisen_chance = BASE_CRAISEN_CHANCE
        spore_death_chance = BASE_SPORE_DEATH_CHANCE
        spore_clone_count = BASE_SPORE_CLONE_COUNT
        clutch_reroll = None
        clutch_plus_one = None

        if potion_key == "mutation":
            mutation_chance = MUTATION_POTION_CHANCE
        elif potion_key == "anti_craisen":
            craisen_chance = ANTI_CRAISEN_CHANCE
        elif potion_key == "reroll":
            clutch_reroll = True
        elif potion_key == "fertility":
            clutch_plus_one = True

        return {
            "mutation_chance": mutation_chance,
            "death_chance": death_chance,
            "craisen_chance": craisen_chance,
            "spore_death_chance": spore_death_chance,
            "spore_clone_count": spore_clone_count,
            "clutch_reroll": clutch_reroll,
            "clutch_plus_one": clutch_plus_one,
        }

    def advance_generation(self) -> list[genetics.Rock]:
        children = self.breed_queue()
        if children:
            self.generation = max(self.generation + 1, max(child.generation for child in children))

        if self.generation >= self.max_generation:
            self.game_over = True

        self.update_market(force=True)
        self.events.append(f"Advanced to generation {self.generation}.")
        return children

    def update_market(self, force: bool = False):
        return self.market_manager.create_market_pods(self, force=force)

    def update_display(self) -> dict[str, Any]:
        self.evaluate_all_rocks()
        active = [rock for rock in self.rock_list.values() if rock.status == genetics.RockStatus.ACTIVE]
        sold = [rock for rock in self.rock_list.values() if rock.status == genetics.RockStatus.SOLD]

        return {
            "generation": self.generation,
            "money": self.money,
            "rock_count": len(self.rock_list),
            "active_count": len(active),
            "sold_count": len(sold),
            "queued_pairs": len(self.breeding_queue),
            "market_pods": len(self.market_pods),
            "potions": dict(self.potions),
            "events": self.events[-10:],
        }

    def show_rocks(self) -> list[str]:
        self.evaluate_all_rocks()
        lines = []
        for rock_id, rock in sorted(self.rock_list.items()):
            lines.append(
                f"#{rock_id} {rock.name.full_name if hasattr(rock.name, 'full_name') else rock.name} "
                f"| {rock.sex.value} | gen {rock.generation} | {rock.status.value} | ${rock.sell_value}"
            )
        return lines

    def evaluate_all_rocks(self) -> None:
        for rock in self.rock_list.values():
            self.finalize_rock(rock)

    def buy_potion(self, potion_key: str) -> bool:
        return self.market_manager.buy_potion(self, potion_key)

    def sell_rock(self, rock_id: int, allow_zero_sale: bool = False) -> int:
        return self.market_manager.sell_rock(self, rock_id, allow_zero_sale=allow_zero_sale)

    def buy_random_rock(self, **kwargs) -> genetics.Rock:
        return self.market_manager.buy_random_rock(self, **kwargs)

    def buy_defined_trait_rock(self, selected_traits: dict[str, object], **kwargs) -> genetics.Rock:
        return self.market_manager.buy_defined_trait_rock(self, selected_traits, **kwargs)


def create_new_game(**kwargs) -> GameMaster:
    game = GameMaster(**kwargs)
    return game
