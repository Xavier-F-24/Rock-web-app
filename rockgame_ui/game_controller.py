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


@dataclass(frozen=True)
class GameStartSettings:
    seed: int | None = None
    starting_money: int = 10
    max_generation: int = 7
    max_pairs_per_generation: int = 3
    rock_farm_cost: int = 75


def start_new_game(
    seed: int | None = None,
    settings: GameStartSettings | dict[str, Any] | None = None,
    **overrides: Any,
) -> GameMaster:
    if settings is None:
        settings_data: dict[str, Any] = {}
    elif isinstance(settings, GameStartSettings):
        settings_data = {
            "seed": settings.seed,
            "starting_money": settings.starting_money,
            "max_generation": settings.max_generation,
            "max_pairs_per_generation": settings.max_pairs_per_generation,
            "rock_farm_cost": settings.rock_farm_cost,
        }
    else:
        settings_data = dict(settings)

    if seed is not None:
        settings_data["seed"] = seed
    settings_data.update(overrides)
    return GameMaster(**settings_data)


def rock_name(rock: genetics.Rock) -> str:
    if hasattr(rock.name, "full_name"):
        return rock.name.full_name
    return str(rock.name)


def rock_label(rock: genetics.Rock) -> str:
    return f"#{rock.id} {rock_name(rock)} ({rock.sex.value}, gen {rock.generation})"


def rock_summary_row(rock: genetics.Rock) -> dict[str, Any]:
    return {
        "id": rock.id,
        "name": rock_name(rock),
        "sex": rock.sex.value,
        "generation": rock.generation,
        "status": rock.status.value,
        "value": rock.value,
        "sell_value": rock.sell_value,
        "parents": ", ".join(str(parent_id) for parent_id in rock.parent_ids),
    }


def get_active_rocks(game: GameMaster) -> list[genetics.Rock]:
    return [
        rock
        for rock in game.rocks.values()
        if rock.status == genetics.RockStatus.ACTIVE
    ]


def get_rock(game: GameMaster, rock_id: int) -> genetics.Rock:
    rock = game.get_rock(rock_id)
    if rock is None:
        raise ValueError(f"Unknown rock id: {rock_id}")
    return rock


def get_breedable_rocks(game: GameMaster) -> list[genetics.Rock]:
    return get_active_rocks(game)


def get_available_breeding_candidates(game: GameMaster) -> list[genetics.Rock]:
    queued_ids = {pair.parent_a_id for pair in game.breeding_queue}
    queued_ids.update(pair.parent_b_id for pair in game.breeding_queue)
    return [
        rock
        for rock in get_breedable_rocks(game)
        if rock.id not in queued_ids
    ]


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
        rock_summary_row(rock)
        for _, rock in sorted(game.rocks.items())
    ]


def get_active_rock_rows(game: GameMaster) -> list[dict[str, Any]]:
    game.evaluate_all_rocks()
    return [
        rock_summary_row(rock)
        for rock in sorted(get_active_rocks(game), key=lambda owned_rock: owned_rock.id)
    ]


def get_sellable_rock_rows(game: GameMaster) -> list[dict[str, Any]]:
    return [
        {
            "id": rock.id,
            "name": rock_name(rock),
            "sex": rock.sex.value,
            "generation": rock.generation,
            "status": rock.status.value,
            "sell_value": rock.sell_value,
        }
        for rock in sorted(get_sellable_rocks(game), key=lambda sellable_rock: sellable_rock.id)
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


def get_breeding_queue_rows(game: GameMaster) -> list[dict[str, Any]]:
    rows = []
    for index, pair in enumerate(game.breeding_queue, start=1):
        parent_a = game.get_rock(pair.parent_a_id)
        parent_b = game.get_rock(pair.parent_b_id)
        rows.append(
            {
                "slot": index,
                "parent_a": rock_label(parent_a) if parent_a is not None else f"Unknown #{pair.parent_a_id}",
                "parent_b": rock_label(parent_b) if parent_b is not None else f"Unknown #{pair.parent_b_id}",
                "potion": ", ".join(pair.potion_keys) if pair.potion_keys else "None",
            }
        )
    return rows


def get_queued_parent_badges(game: GameMaster) -> dict[int, dict[str, str]]:
    colors = [
        "#E15759",
        "#4E79A7",
        "#59A14F",
        "#B07AA1",
        "#F28E2B",
        "#76B7B2",
    ]
    badges: dict[int, dict[str, str]] = {}

    for index, pair in enumerate(game.breeding_queue):
        color = colors[index % len(colors)]
        for rock_id in (pair.parent_a_id, pair.parent_b_id):
            badges[int(rock_id)] = {
                "text": "\u2665",
                "color": color,
            }

    return badges


def get_active_score_total(game: GameMaster) -> int:
    game.evaluate_all_rocks()
    return sum(rock.score_value for rock in get_active_rocks(game))


def get_final_score_summary(game: GameMaster) -> dict[str, int]:
    active_score = get_active_score_total(game)
    return {
        "active_rock_score": active_score,
        "money": int(game.money),
        "rock_farm_cost": int(game.rock_farm_cost),
        "final_score": active_score + int(game.money) - int(game.rock_farm_cost),
    }


def is_game_finished(game: GameMaster | None) -> bool:
    return bool(game is not None and game.game_over)


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
                "potion_keys": list(pair.potion_keys),
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


def buy_potions(game: GameMaster, quantities: dict[str, int]) -> ActionResult:
    requested = {
        potion_key: int(quantity)
        for potion_key, quantity in quantities.items()
        if int(quantity) > 0
    }
    if not requested:
        return ActionResult(False, "Choose at least one potion to buy.")

    unknown = sorted(set(requested) - set(POTION_SHOP))
    if unknown:
        return ActionResult(False, f"Unknown potion(s): {', '.join(unknown)}.")

    total_cost = sum(POTION_SHOP[potion_key]["cost"] * quantity for potion_key, quantity in requested.items())
    if game.money < total_cost:
        return ActionResult(False, f"Not enough money. Need ${total_cost}, have ${game.money}.")

    bought_count = 0
    try:
        for potion_key, quantity in requested.items():
            for _ in range(quantity):
                game.buy_potion(potion_key)
                bought_count += 1
    except Exception as exc:
        return ActionResult(False, str(exc))

    return ActionResult(True, f"Bought {bought_count} potion(s) for ${total_cost}.", dict(requested))


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


def get_pending_market_pod_rows(game: GameMaster) -> list[dict[str, Any]]:
    pending = game.pending_market_pod
    if pending is None:
        return []

    return [
        {
            "index": index,
            "name": rock_name(child),
            "sex": child.sex.value,
            "generation": child.generation,
            "status": child.status.value,
            "value": child.value,
            "sell_value": child.sell_value,
            "parents": ", ".join(str(parent_id) for parent_id in child.parent_ids),
        }
        for index, child in enumerate(pending.children)
    ]


def get_pending_market_pod_rocks(game: GameMaster) -> dict[str, list[genetics.Rock]]:
    pending = game.pending_market_pod
    if pending is None:
        return {"parents": [], "children": []}

    parents = [
        rock
        for rock in (
            game.get_rock(pending.parent_a_id),
            game.get_rock(pending.parent_b_id),
        )
        if rock is not None
    ]
    return {"parents": parents, "children": list(pending.children)}


def buy_market_pod(game: GameMaster, offer_id: str) -> ActionResult:
    try:
        pending = game.market_manager.buy_market_pod(game, offer_id)
    except Exception as exc:
        return ActionResult(False, str(exc))

    return ActionResult(
        True,
        f"Bought {pending.offer.name} pod. Choose one child to keep.",
        get_pending_market_pod_rows(game),
    )


def choose_market_pod_child(game: GameMaster, child_index: int) -> ActionResult:
    try:
        child = game.market_manager.choose_market_pod_child(game, child_index)
    except Exception as exc:
        return ActionResult(False, str(exc))

    return ActionResult(True, f"Kept market child #{child.id} {rock_name(child)}.", rock_summary_row(child))


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
    potion_keys = options.get("potion_keys")

    try:
        pair = game.add_pair_to_queue(
            parent_a_id,
            parent_b_id,
            potion_key=potion_key,
            potion_keys=potion_keys,
        )
    except Exception as exc:
        return ActionResult(False, str(exc))

    potion_text = f" with {', '.join(pair.potion_keys)}" if pair.potion_keys else ""
    return ActionResult(True, f"Queued #{pair.parent_a_id} x #{pair.parent_b_id}{potion_text}.", pair)


def remove_queued_pair(game: GameMaster, slot: int) -> ActionResult:
    try:
        removed = game.remove_pair_from_queue(slot - 1)
    except Exception as exc:
        return ActionResult(False, str(exc))

    refund = f" Refunded {', '.join(removed.potion_keys)}." if removed.potion_keys else ""
    return ActionResult(True, f"Removed queued pair #{removed.parent_a_id} x #{removed.parent_b_id}.{refund}", removed)


def advance_breeding_generation(game: GameMaster) -> ActionResult:
    if not game.breeding_queue:
        return ActionResult(False, "No breeding pairs are queued.")

    try:
        children = game.advance_generation()
    except Exception as exc:
        return ActionResult(False, str(exc))

    child_rows = [rock_summary_row(child) for child in children]
    return ActionResult(
        True,
        f"Bred {len(children)} child rock(s) and advanced to generation {game.generation}.",
        child_rows,
    )


def render_tree_for_streamlit(game: GameMaster, **kwargs):
    return TreeDrawer(game=game, **kwargs).draw()


def render_rock(rock: genetics.Rock, **kwargs) -> str:
    kwargs.setdefault("sprite_size", 1.4)
    kwargs.setdefault("dpi", 120)
    return rock_to_image_uri(rock, **kwargs)


def serialize_game(game: GameMaster) -> str:
    return game_to_json_string(game)


def load_game_from_json(json_string: str) -> GameMaster:
    try:
        return game_from_json_string(json_string)
    except Exception as exc:
        raise ValueError(f"Could not load save: {exc}") from exc
