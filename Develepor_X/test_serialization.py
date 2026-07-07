from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_Serialization.rock_serialization_helper import (
    game_from_json_string,
    game_to_dict,
    game_to_json_string,
)


def test_game_master_serialization_round_trips_core_state():
    game = GameMaster(seed=31, starting_money=30)
    game.buy_potion("fertility")

    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1], potion_key="fertility")
    game.advance_generation()

    json_string = game_to_json_string(game)
    loaded = game_from_json_string(json_string)

    assert loaded.generation == game.generation
    assert loaded.money == game.money
    assert loaded.next_rock_id == game.next_rock_id
    assert set(loaded.rocks) == set(game.rocks)
    assert loaded.update_display()["rock_count"] == game.update_display()["rock_count"]


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
