import json

from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_Drawing.rock_drawing_helper import render_game_rock_images
from Rock_Serialization.rock_serialization_helper import (
    game_from_json_string,
    game_to_dict,
    game_to_json_string,
)


def test_game_master_serialization_round_trips_core_state():
    game = GameMaster(seed=31, starting_money=30, rock_farm_cost=90)
    game.buy_potion("fertility")
    game.buy_potion("mutation")

    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1], potion_keys=["fertility", "mutation"])
    game.advance_generation()

    json_string = game_to_json_string(game)
    loaded = game_from_json_string(json_string)

    assert loaded.generation == game.generation
    assert loaded.money == game.money
    assert loaded.next_rock_id == game.next_rock_id
    assert loaded.rock_farm_cost == 90
    assert set(loaded.rocks) == set(game.rocks)
    assert loaded.update_display()["rock_count"] == game.update_display()["rock_count"]


def test_serialization_round_trips_queued_pair_multiple_potions():
    game = GameMaster(seed=33, starting_money=80)
    game.buy_potion("fertility")
    game.buy_potion("mutation")
    ids = list(game.rocks)

    game.add_pair_to_queue(ids[0], ids[1], potion_keys=["fertility", "mutation"])
    save_data = game_to_dict(game)
    loaded = game_from_json_string(json.dumps(save_data))

    assert save_data["game"]["breeding_queue"][0]["potion_keys"] == ["fertility", "mutation"]
    assert loaded.breeding_queue[0].potion_keys == ["fertility", "mutation"]


def test_serialization_loads_old_single_potion_queue_entries():
    game = GameMaster(seed=34, starting_money=80)
    game.buy_potion("fertility")
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1], potion_key="fertility")
    save_data = game_to_dict(game)
    save_data["game"]["breeding_queue"][0].pop("potion_keys")

    loaded = game_from_json_string(json.dumps(save_data))

    assert loaded.breeding_queue[0].potion_keys == ["fertility"]


def test_serialization_defaults_rock_farm_cost_for_old_saves():
    game = GameMaster(seed=32, starting_money=30)
    save_data = game_to_dict(game)
    del save_data["game"]["rock_farm_cost"]

    loaded = game_from_json_string(game_to_json_string(game))
    old_loaded = game_from_json_string(json.dumps(save_data))

    assert loaded.rock_farm_cost == 75
    assert old_loaded.rock_farm_cost == 75


def test_serialization_includes_queue_market_pods_and_pending_pod():
    game = GameMaster(seed=41, starting_money=30)
    offer = game.market_pods[0]
    game.market_manager.buy_market_pod(game, offer.offer_id)

    save_data = game_to_dict(game)
    game_data = save_data["game"]

    assert game_data["market_pods"]
    assert game_data["pending_market_pod"] is not None
    assert game_data["pending_market_pod"]["children"]

    loaded = game_from_json_string(game_to_json_string(game))

    assert loaded.market_pods
    assert loaded.pending_market_pod is not None
    assert loaded.pending_market_pod.children


def test_serialization_does_not_store_rendered_images_or_runtime_cache():
    game = GameMaster(seed=42, starting_money=30)
    render_game_rock_images(game, dpi=40)

    save_data = game_to_dict(game)
    json_string = game_to_json_string(game)

    assert hasattr(game, "rock_image_cache")
    assert "rock_image_cache" not in save_data["game"]
    assert "image_uri" not in json_string
    assert "data:image/png;base64" not in json_string
    assert "image_path" not in json_string
    assert all("image_path" not in rock_data for rock_data in save_data["game"]["rocks"])


def test_loaded_game_regenerates_rock_images_on_demand():
    game = GameMaster(seed=43, starting_money=30)
    loaded = game_from_json_string(game_to_json_string(game))

    assert not hasattr(loaded, "rock_image_cache")

    image_by_id = render_game_rock_images(loaded, dpi=40)

    assert set(image_by_id) == set(loaded.rocks)
    assert hasattr(loaded, "rock_image_cache")
    assert all(uri.startswith("data:image/png;base64,") for uri in image_by_id.values())
