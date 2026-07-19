"""
Prototype market manager for the split rock game modules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import Rock_Breeding.rock_breeding_helper as breeding
import Rock_Genetics.rock_genetic_helper as genetics


POTION_SHOP = {
    "anti_craisen": {
        "name": "Anti-Craisen Potion",
        "cost": 5,
        "description": "Reduce or reroll craisen offspring risk.",
    },
    "mutation": {
        "name": "Mutation Potion",
        "cost": 5,
        "description": "Increase mutation chance for one breeding pair.",
    },
    "fertility": {
        "name": "Fertility Potion",
        "cost": 3,
        "description": "Produce extra child from one pair.",
    },
    "reroll": {
        "name": "Reroll Potion",
        "cost": 3,
        "description": "Reroll clutch size from one pair.",
    },
}

MARKET_POD_TIERS = {
    "low": {
        "name": "Craigslist Gravel",
        "tagline": "Cheap, chaotic, and questionably damp.",
        "price": 3,
        "parent_value_min": 1,
        "parent_value_max": 4,
    },
    "medium": {
        "name": "Respectable Gravel",
        "tagline": "Decent family lines. Probably has a LinkedIn.",
        "price": 6,
        "parent_value_min": 3,
        "parent_value_max": 8,
    },
    "high": {
        "name": "Boulder Elite",
        "tagline": "Pedigreed, polished, and financially insufferable.",
        "price": 10,
        "parent_value_min": 7,
        "parent_value_max": 999,
    },
}

RANDOM_ROCK_COST = 8
REQUESTED_ROCK_BASE_COST = 8
REQUESTED_ROCK_VALUE_MULTIPLIER = 2
REQUESTED_TRAIT_SURCHARGE = 2


@dataclass
class MarketPodOffer:
    offer_id: str
    tier: str
    name: str
    tagline: str
    price: int
    parent_a: genetics.Rock
    parent_b: genetics.Rock
    used: bool = False


@dataclass
class PendingMarketPod:
    offer: MarketPodOffer
    parent_a_id: int
    parent_b_id: int
    children: list[genetics.Rock]


@dataclass
class MarketManager:
    """
    Handles shop actions for a GameMaster-like object.
    """

    rng: random.Random = field(default_factory=random.Random)
    genome_factory: genetics.GenomeFactory = field(default_factory=genetics.GenomeFactory)
    expression_engine: genetics.ExpressionEngine = field(default_factory=genetics.ExpressionEngine)
    value_calculator: genetics.ValueCalculator = field(default_factory=genetics.ValueCalculator)
    name_generator: genetics.NameGenerator = field(default_factory=genetics.NameGenerator)
    breeding_master: breeding.BreedingMaster = field(default_factory=breeding.BreedingMaster)
    pod_tiers: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(MARKET_POD_TIERS))
    potion_shop: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(POTION_SHOP))

    def __post_init__(self):
        self.breeding_master.GenomeFactory = self.genome_factory
        self.breeding_master.ExpressionEngine = self.expression_engine
        self.breeding_master.ValueCalculator = self.value_calculator
        self.breeding_master.NameGenerator = self.name_generator

    def make_random_rock(
        self,
        rock_id: int,
        sex: genetics.Sex | None = None,
        generation: int = 0,
        market_guest: bool = False,
    ) -> genetics.Rock:
        sex = sex or self.rng.choice([genetics.Sex.MALE, genetics.Sex.FEMALE])
        rock = genetics.Rock(
            id=rock_id,
            name=self.name_generator.generate_name(sex=sex),
            sex=sex,
            genotype=self.genome_factory.make_random_rock_genome(),
            death_genes=self.genome_factory.make_death_genes(),
            generation=generation,
            status=genetics.RockStatus.ACTIVE,
            is_market=market_guest,
        )
        self.finalize_rock(rock)
        return rock

    def make_selected_rock(
        self,
        rock_id: int,
        selected_traits: dict[str, object] | None = None,
        sex: genetics.Sex | None = None,
        generation: int = 0,
        random_fill: bool = True,
    ) -> genetics.Rock:
        selected_traits = dict(selected_traits or {})
        requested_sex = selected_traits.pop("sex", None) or selected_traits.pop("gender", None)
        sex = sex or self.parse_sex(requested_sex) or self.rng.choice([genetics.Sex.MALE, genetics.Sex.FEMALE])

        rock = genetics.Rock(
            id=rock_id,
            name=self.name_generator.generate_name(sex=sex),
            sex=sex,
            genotype=self.genome_factory.make_selected_rock_genome(
                selected_traits=selected_traits,
                random_fill=random_fill,
            ),
            death_genes=self.genome_factory.make_death_genes(),
            generation=generation,
            status=genetics.RockStatus.ACTIVE,
        )
        self.finalize_rock(rock)
        return rock

    @staticmethod
    def parse_sex(value: object) -> genetics.Sex | None:
        if value is None:
            return None

        text = str(value).strip().lower()
        if text in {"male", "m", "1", "01", "10"}:
            return genetics.Sex.MALE
        if text in {"female", "f", "0", "00"}:
            return genetics.Sex.FEMALE

        raise ValueError(f"Unknown sex value: {value!r}")

    def finalize_rock(self, rock: genetics.Rock) -> genetics.Rock:
        self.expression_engine.instantiate_phenotype(rock)
        self.value_calculator.set_rock_value(rock)
        return rock

    def create_market_pods(self, game, force: bool = False) -> list[MarketPodOffer]:
        if getattr(game, "market_pods", None) and not force:
            return game.market_pods

        offers: list[MarketPodOffer] = []
        for tier_key, tier in self.pod_tiers.items():
            parent_a = self.make_market_parent_for_tier(game, tier_key, genetics.Sex.MALE)
            parent_b = self.make_market_parent_for_tier(game, tier_key, genetics.Sex.FEMALE)

            offers.append(
                MarketPodOffer(
                    offer_id=f"pod_g{game.generation}_{tier_key}_0",
                    tier=tier_key,
                    name=tier["name"],
                    tagline=tier["tagline"],
                    price=int(tier["price"]),
                    parent_a=parent_a,
                    parent_b=parent_b,
                )
            )

        game.market_pods = offers
        return offers

    def make_market_parent_for_tier(
        self,
        game,
        tier_key: str,
        sex: genetics.Sex,
        max_attempts: int = 75,
    ) -> genetics.Rock:
        tier = self.pod_tiers[tier_key]
        best = None
        best_distance = float("inf")

        for _ in range(max_attempts):
            rock = self.make_random_rock(
                rock_id=-1,
                sex=sex,
                generation=max(0, game.generation - 1),
                market_guest=True,
            )
            value = rock.value
            if tier["parent_value_min"] <= value <= tier["parent_value_max"]:
                return rock

            distance = min(
                abs(value - tier["parent_value_min"]),
                abs(value - tier["parent_value_max"]),
            )
            if distance < best_distance:
                best = rock
                best_distance = distance

        return best

    def buy_market_pod(self, game, offer_id: str) -> PendingMarketPod:
        offer = self.get_market_pod(game, offer_id)
        if offer is None:
            raise ValueError(f"Market pod not found: {offer_id}")
        if offer.used:
            raise ValueError(f"Market pod already used: {offer_id}")
        if game.money < offer.price:
            raise ValueError(f"Not enough money. Need ${offer.price}, have ${game.money}.")
        if game.pending_market_pod is not None:
            raise ValueError("Choose a child from the pending market pod before buying another.")

        game.money -= offer.price
        offer.used = True

        self.add_rock_to_game(game, offer.parent_a, owned=False)
        self.add_rock_to_game(game, offer.parent_b, owned=False)

        children = self.breeding_master.breed_parent_set(
            parent_a=offer.parent_a,
            parent_b=offer.parent_b,
            next_id=-1000,
            child_generation=game.generation,
            death_chance=0,
            craisen_chance=0,
            mutation_chance=0,
            spore_death_chance=0,
        )

        for index, child in enumerate(children):
            child.id = -(1000 + index)
            child.parent_ids = [offer.parent_a.id, offer.parent_b.id]
            child.is_market = True
            self.finalize_rock(child)

        pending = PendingMarketPod(
            offer=offer,
            parent_a_id=offer.parent_a.id,
            parent_b_id=offer.parent_b.id,
            children=children,
        )
        game.pending_market_pod = pending
        game.events.append(f"Bought {offer.name} pod for ${offer.price}.")
        return pending

    def choose_market_pod_child(self, game, child_index: int = 0) -> genetics.Rock:
        pending = game.pending_market_pod
        if pending is None:
            raise ValueError("No pending market pod.")
        if child_index < 0 or child_index >= len(pending.children):
            raise IndexError(f"Invalid market child index: {child_index}")

        child = pending.children[child_index]
        child.parent_ids = [pending.parent_a_id, pending.parent_b_id]
        child.generation = game.generation
        self.add_rock_to_game(game, child, owned=True)
        child.is_market = True

        game.pending_market_pod = None
        game.events.append(f"Kept market child #{child.id} from {pending.offer.name}.")
        return child

    @staticmethod
    def get_market_pod(game, offer_id: str) -> MarketPodOffer | None:
        for offer in getattr(game, "market_pods", []):
            if offer.offer_id == offer_id:
                return offer
        return None

    def add_rock_to_game(self, game, rock: genetics.Rock, owned: bool = True) -> genetics.Rock:
        rock.id = game.reserve_rock_id()
        rock.is_market = not owned
        game.rock_list[rock.id] = rock
        self.finalize_rock(rock)
        return rock

    def sell_rock(self, game, rock_id: int, allow_zero_sale: bool = False) -> int:
        rock = game.get_rock(rock_id)
        if rock is None:
            raise ValueError(f"Unknown rock id: {rock_id}")

        self.finalize_rock(rock)
        value = rock.sell_value
        if value <= 0 and not allow_zero_sale:
            raise ValueError(f"Rock #{rock_id} has no sell value.")

        rock.change_status(genetics.RockStatus.SOLD)
        game.money += value
        game.events.append(f"Sold rock #{rock.id} for ${value}.")
        return value

    def buy_potion(self, game, potion_key: str) -> bool:
        if potion_key not in self.potion_shop:
            raise ValueError(f"Unknown potion: {potion_key}")

        potion = self.potion_shop[potion_key]
        cost = int(potion["cost"])
        if game.money < cost:
            raise ValueError(f"Not enough money. Need ${cost}, have ${game.money}.")

        game.money -= cost
        game.potions[potion_key] = game.potions.get(potion_key, 0) + 1
        game.events.append(f"Bought {potion['name']} for ${cost}.")
        return True

    def buy_random_rock(self, game, cost: int = RANDOM_ROCK_COST, sex: genetics.Sex | None = None) -> genetics.Rock:
        if game.money < cost:
            raise ValueError(f"Not enough money. Need ${cost}, have ${game.money}.")

        game.money -= cost
        rock = self.make_random_rock(game.reserve_rock_id(), sex=sex, generation=game.generation)
        rock.is_market = True
        game.rock_list[rock.id] = rock
        game.events.append(f"Bought random rock #{rock.id} for ${cost}.")
        return rock

    def buy_defined_trait_rock(
        self,
        game,
        selected_traits: dict[str, object],
        cost: int | None = None,
        random_fill: bool = True,
    ) -> genetics.Rock:
        preview = self.make_selected_rock(
            rock_id=game.next_rock_id,
            selected_traits=selected_traits,
            generation=game.generation,
            random_fill=random_fill,
        )
        cost = cost if cost is not None else self.price_defined_trait_rock(preview)
        if game.money < cost:
            raise ValueError(f"Not enough money. Need ${cost}, have ${game.money}.")

        game.money -= cost
        preview.id = game.reserve_rock_id()
        preview.is_market = True
        game.rock_list[preview.id] = preview
        game.events.append(f"Bought defined-trait rock #{preview.id} for ${cost}.")
        return preview

    @staticmethod
    def quote_defined_trait_request(selected_traits: dict[str, object]) -> int:
        """Return a public quote without generating the rock's hidden genome."""
        public_traits = {
            str(name): value for name, value in selected_traits.items()
            if value is not None
        }
        return REQUESTED_ROCK_BASE_COST + REQUESTED_TRAIT_SURCHARGE * len(public_traits)

    @staticmethod
    def price_defined_trait_rock(rock: genetics.Rock) -> int:
        return max(REQUESTED_ROCK_BASE_COST, rock.value * REQUESTED_ROCK_VALUE_MULTIPLIER)


def show_potion_shop() -> None:
    for key, potion in POTION_SHOP.items():
        print(f"{key}: {potion['name']} (${potion['cost']}) - {potion['description']}")
