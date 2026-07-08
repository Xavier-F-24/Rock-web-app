"""Small presentation adapters for Streamlit views."""

from __future__ import annotations

from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics


def rock_name(rock: genetics.Rock) -> str:
    if hasattr(rock.name, "full_name"):
        return rock.name.full_name
    return str(rock.name)


def rock_label(rock: genetics.Rock) -> str:
    return f"#{rock.id} {rock_name(rock)} ({rock.sex.value}, gen {rock.generation})"


def rock_table_rows(game) -> list[dict[str, Any]]:
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


def active_rocks(game) -> list[genetics.Rock]:
    return [
        rock
        for rock in game.rocks.values()
        if rock.status == genetics.RockStatus.ACTIVE
    ]


def game_summary(game) -> dict[str, Any]:
    return game.update_display()


def raw_state_summary(game) -> dict[str, Any]:
    return {
        "generation": game.generation,
        "max_generation": game.max_generation,
        "money": game.money,
        "rock_count": len(game.rocks),
        "active_rock_ids": [rock.id for rock in active_rocks(game)],
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
