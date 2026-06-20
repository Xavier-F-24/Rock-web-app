#-----------------------------------------------------
"""
Rock GAME Helper 

This file answers:

- How does the game run?
- What are the rules of the game?

- What data needs to be pushed to run a game?

"""
#-----------------------------------------------------

# ============================================================
# SAVE / LOAD SYSTEM
# ============================================================

import json
from datetime import datetime

SAVE_VERSION = "0.1.0"

def rock_to_dict(rock):
    """
    Convert a Rock object into a JSON-safe dictionary.

    We save core rock identity, genes, lineage, and gameplay flags.
    Runtime-only calculated values are also saved for readability,
    but are recalculated after loading.
    """
    ensure_rock_game_attributes(rock)

    parents = getattr(rock, "parents", None)
    if parents is not None:
        parents = list(parents)

    return {
        "id": int(rock.id),
        "name": str(rock.name),
        "genes": dict(rock.genes),
        "parents": parents,
        "generation": int(getattr(rock, "generation", 0)),

        # Gameplay state
        "sold": bool(getattr(rock, "sold", False)),
        "imported": bool(getattr(rock, "imported", False)),
        "used_as_parent": bool(getattr(rock, "used_as_parent", False)),
        "dead": bool(getattr(rock, "dead", False)),
        "puffed": bool(getattr(rock, "puffed", False)),
        "death_reason": getattr(rock, "death_reason", None),

        # Cached values, recalculated on load
        "base_value": int(getattr(rock, "base_value", 0)),
        "sell_value": int(getattr(rock, "sell_value", 0)),
        "score_value": int(getattr(rock, "score_value", 0)),
        "is_craisen": int(getattr(rock, "is_craisen", 0)),
        "rock_cost": int(getattr(rock, "rock_cost", 0)),

        # Gender cache
        "gender": getattr(rock, "gender", None),
    }

def rock_from_dict(data):
    """
    Rebuild a Rock object from saved dictionary data.
    """
    parents = data.get("parents", None)

    if parents is not None:
        parents = tuple(int(p) for p in parents)

    rock = Rock(
        id=int(data["id"]),
        name=str(data.get("name", f"Rock_{data['id']}")),
        genes=dict(data.get("genes", {})),
        parents=parents,
        generation=int(data.get("generation", 0))
    )

    ensure_rock_game_attributes(rock)

    # Restore gameplay state
    rock.sold = bool(data.get("sold", False))
    rock.imported = bool(data.get("imported", False))
    rock.used_as_parent = bool(data.get("used_as_parent", False))
    rock.dead = bool(data.get("dead", False))
    rock.puffed = bool(data.get("puffed", False))
    rock.death_reason = data.get("death_reason", None)

    # Restore cached values, then recalculate later
    rock.base_value = int(data.get("base_value", 0))
    rock.sell_value = int(data.get("sell_value", 0))
    rock.score_value = int(data.get("score_value", 0))
    rock.is_craisen = int(data.get("is_craisen", 0))
    rock.rock_cost = int(data.get("rock_cost", 0))

    # Restore/sync gender
    if data.get("gender", None) is not None:
        rock.gender = data.get("gender", None)

    if "gender" in rock.genes:
        try:
            rock.gender = express_gender_from_gene(rock.genes["gender"])
        except Exception:
            pass

    return rock

def serialize_breeding_queue_entry(entry):
    """
    Convert breeding queue entry to JSON-safe format.

    Supports:
    - old tuple format: (a, b)
    - new dict format: {"parents": (a, b), "potion": potion_key}
    """
    if isinstance(entry, dict):
        a, b = get_queue_entry_pair(entry)
        potion = get_queue_entry_potion(entry)

        return {
            "parents": [int(a), int(b)],
            "potion": potion
        }

    a, b = entry

    return {
        "parents": [int(a), int(b)],
        "potion": None
    }

def deserialize_breeding_queue_entry(data):
    """
    Rebuild breeding queue entry from JSON-safe format.
    """
    parents = data.get("parents", None)

    if parents is None or len(parents) != 2:
        return None

    potion = data.get("potion", None)

    return {
        "parents": (int(parents[0]), int(parents[1])),
        "potion": potion
    }

def game_to_dict(game):
    """
    Convert full GameState into a JSON-safe dictionary.
    """
    evaluate_all_rocks(game)

    save_data = {
        "save_version": SAVE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),

        "game": {
            "player_name": getattr(game, "player_name", ""),
            "cabal_curse_enabled": bool(getattr(game, "cabal_curse_enabled", False)),
            "market_pod_used_generations": list(getattr(game, "market_pod_used_generations", [])),
            "generation": int(game.generation),
            "max_generation": int(game.max_generation),
            "money": int(game.money),
            "next_id": int(game.next_id),
            "max_pairs_per_generation": int(game.max_pairs_per_generation),
            "game_over": bool(game.game_over),

            "potions": dict(game.potions),
            "events": list(game.events),

            "breeding_queue": [
                serialize_breeding_queue_entry(entry)
                for entry in game.breeding_queue
            ],

            "rocks": [
                rock_to_dict(rock)
                for _, rock in sorted(game.rocks.items())
            ],
        }
    }

    return save_data

def game_from_dict(save_data):
    """
    Rebuild GameState from a saved dictionary.
    """
    if "game" not in save_data:
        raise ValueError("Invalid save file: missing 'game' field.")

    g = save_data["game"]

    game = GameState(
        rocks={},
        next_id=int(g.get("next_id", 1)),
        generation=int(g.get("generation", 0)),
        max_generation=int(g.get("max_generation", DEFAULT_MAX_GENERATION)),
        money=int(g.get("money", DEFAULT_STARTING_MONEY)),
        max_pairs_per_generation=int(
            g.get("max_pairs_per_generation", DEFAULT_MAX_PAIRS_PER_GENERATION)
        ),
        breeding_queue=[],
        potions=dict(g.get("potions", {})),
        events=list(g.get("events", [])),
        game_over=bool(g.get("game_over", False))
    )

    #game.player_name = data.get("player_name", "")
    #game.cabal_curse_enabled = bool(data.get("cabal_curse_enabled", False))
    #game.market_pod_used_generations = data.get("market_pod_used_generations", [])

    ensure_market_state(game)
    ensure_player_profile_state(game)

    # Rebuild rocks
    for rock_data in g.get("rocks", []):
        rock = rock_from_dict(rock_data)
        game.rocks[int(rock.id)] = rock

    # Rebuild queue
    queue = []

    for entry_data in g.get("breeding_queue", []):
        entry = deserialize_breeding_queue_entry(entry_data)
        if entry is not None:
            queue.append(entry)

    game.breeding_queue = queue

    # Safety: ensure next_id is above all rock IDs
    if len(game.rocks) > 0:
        game.next_id = max(game.next_id, max(game.rocks.keys()) + 1)

    # Recalculate values and parent flags
    evaluate_all_rocks(game)

    game.events.append("Game loaded from save file.")

    return game

def game_to_json_string(game, indent=2):
    """
    Convert game to pretty JSON string.
    """
    return json.dumps(
        game_to_dict(game),
        indent=indent,
        sort_keys=False
    )

def game_from_json_string(json_string):
    """
    Load game from JSON string.
    """
    save_data = json.loads(json_string)
    return game_from_dict(save_data)

def save_game_json(game, filepath):
    """
    Save game to a JSON file path.
    Useful outside Streamlit.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(game_to_json_string(game))

    return filepath

def load_game_json(filepath):
    """
    Load game from a JSON file path.
    Useful outside Streamlit.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        json_string = f.read()

    return game_from_json_string(json_string)

def make_save_filename(game):
    """
    Create a friendly save filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rock_game_gen{game.generation}_money{game.money}_{timestamp}.json"
