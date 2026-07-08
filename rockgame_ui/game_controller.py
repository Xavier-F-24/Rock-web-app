"""Streamlit-safe controller functions for the rock genetics game.

This module is intentionally UI-framework-free. Streamlit pages can call these
small functions without knowing the internals of GameMaster, MarketManager, or
the renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Drawing.rock_drawing_helper import rock_to_image_uri
from Rock_Drawing.rock_lineage_drawing_helper import TreeDrawer
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_Market.rock_market_helper import POTION_SHOP
from Rock_Serialization.rock_serialization_helper import game_from_json_string, game_to_json_string


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str
    payload: Any = None


def start_new_game(seed: int | None = None) -> GameMaster:
    return GameMaster(seed=seed)


def rock_name(rock: genetics.Rock) -> str:
    if hasattr(rock.name, "full_name"):
        return rock.name.full_name
    return str(rock.name)


def rock_label(rock: genetics.Rock) -> str:
    return f"#{rock.id} {rock_name(rock)} ({rock.sex.value}, gen {rock.generation})"


def get_active_rocks(game: GameMaster) -> list[genetics.Rock]:
    return [
        rock
        for rock in game.rocks.values()
        if rock.status == genetics.RockStatus.ACTIVE
    ]


def get_breedable_rocks(game: GameMaster) -> list[genetics.Rock]:
    return get_active_rocks(game)


def get_sellable_rocks(game: GameMaster) -> list[genetics.Rock]:
    game.evaluate_all_rocks()
    return [
        rock
        for rock in game.rocks.values()
        if rock.status != genetics.RockStatus.SOLD and rock.sell_value > 0
    ]


def get_rock_rows(game: GameMaster) -> list[dict[str, Any]]:
    game.evaluate_all_rocks()
    return [
        {
            "id": rock.id,
            "name": rock_name(rock),
            "sex": rock.sex.value,
            "generation": rock.generation,
            "status": rock.status.value,
            "value": rock.value,
            "sell_value": rock.sell_value,
            "parents": ", ".join(str(parent_id) for parent_id in rock.parent_ids),
        }
        for _, rock in sorted(game.rocks.items())
    ]


def get_game_summary(game: GameMaster) -> dict[str, Any]:
    summary = game.update_display()
    summary["max_generation"] = game.max_generation
    return summary


def get_recent_events(game: GameMaster, limit: int = 10) -> list[str]:
    return game.events[-limit:]


def get_queue_summary(game: GameMaster) -> dict[str, int]:
    return {
        "queued_pairs": len(game.breeding_queue),
        "max_pairs": game.max_pairs_per_generation,
    }


def get_potion_options(game: GameMaster) -> list[str | None]:
    return [None] + sorted(game.potions)


def has_rocks(game: GameMaster) -> bool:
    return bool(game.rocks)


def get_raw_state_summary(game: GameMaster) -> dict[str, Any]:
    return {
        "generation": game.generation,
        "max_generation": game.max_generation,
        "money": game.money,
        "rock_count": len(game.rocks),
        "active_rock_ids": [rock.id for rock in get_active_rocks(game)],
        "queued_pairs": [
            {
                "parent_a_id": pair.parent_a_id,
                "parent_b_id": pair.parent_b_id,
                "potion_key": pair.potion_key,
            }
            for pair in game.breeding_queue
        ],
        "market_pod_ids": [offer.offer_id for offer in game.market_pods],
        "pending_market_pod": game.pending_market_pod is not None,
        "potions": dict(game.potions),
        "game_over": game.game_over,
        "recent_events": game.events[-10:],
    }


def get_potion_rows(game: GameMaster) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "name": potion["name"],
            "cost": potion["cost"],
            "description": potion["description"],
            "owned": game.potions.get(key, 0),
        }
        for key, potion in POTION_SHOP.items()
    ]


def get_market_pod_rows(game: GameMaster) -> list[dict[str, Any]]:
    return [
        {
            "offer_id": offer.offer_id,
            "name": offer.name,
            "tier": offer.tier,
            "price": offer.price,
            "used": offer.used,
            "tagline": offer.tagline,
        }
        for offer in game.market_pods
    ]


def validate_breeding_pair(game: GameMaster, parent_a_id: int, parent_b_id: int) -> dict[str, Any]:
    parent_a = game.get_rock(parent_a_id)
    parent_b = game.get_rock(parent_b_id)
    return game.breeding_master.validate_breeding_pair(parent_a, parent_b, game=game)


def buy_random_rock(game: GameMaster) -> ActionResult:
    try:
        rock = game.buy_random_rock()
    except Exception as exc:
        return ActionResult(False, str(exc))
    return ActionResult(True, f"Bought random rock #{rock.id}.", rock)


def sell_rock(game: GameMaster, rock_id: int) -> ActionResult:
    try:
        value = game.sell_rock(rock_id)
    except Exception as exc:
        return ActionResult(False, str(exc))
    return ActionResult(True, f"Sold rock #{rock_id} for ${value}.", value)


def breed_pair(
    game: GameMaster,
    parent_a_id: int,
    parent_b_id: int,
    options: dict[str, Any] | None = None,
) -> ActionResult:
    options = dict(options or {})
    potion_key = options.get("potion_key")

    try:
        pair = game.add_pair_to_queue(parent_a_id, parent_b_id, potion_key=potion_key)
    except Exception as exc:
        return ActionResult(False, str(exc))

    return ActionResult(True, f"Queued #{pair.parent_a_id} x #{pair.parent_b_id}.", pair)


def render_tree_for_streamlit(game: GameMaster, **kwargs):
    return TreeDrawer(game=game, **kwargs).draw()


def render_rock(rock: genetics.Rock, **kwargs) -> str:
    return rock_to_image_uri(rock, **kwargs)


def serialize_game(game: GameMaster) -> str:
    return game_to_json_string(game)


def load_game_from_json(json_string: str) -> GameMaster:
    try:
        return game_from_json_string(json_string)
    except Exception as exc:
        raise ValueError(f"Could not load save: {exc}") from exc
