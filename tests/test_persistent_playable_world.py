import copy

from Rock_AI.actions.farmer_action_type import FarmerActionType
from Rock_AI.actions.farmer_action import PlaceBidAction
from Rock_AI.economy.transaction_validator import EconomyTransactionManager
from Rock_Serialization.rock_serialization_helper import game_from_dict, game_from_json_string, game_to_dict, game_to_json_string
from Rock_GameState.rock_game_state_helper import GameMaster
from Rock_World.rock_playable_world_manager import PlayableWorldManager
from rockgame_ui import game_controller


def _game(seed=101, count=3):
    return game_controller.start_new_game(
        seed=seed, world_size_mode="fixed", world_farmer_count=count,
        allow_neural_farmers=False,
    )


def test_fixed_world_has_real_lineages_and_one_owner_per_rock():
    game = _game()
    world = game.world
    assert world.farm("player").game is game
    assert world.resolved_npc_count == 3
    assert len({farm.profile.display_name for key, farm in world.farms.items() if key != "player"}) == 3
    assert all(1 <= farm.generation - game.generation <= 3 for key, farm in world.farms.items() if key != "player")
    assert all(
        all(parent_id in farm.rocks for parent_id in rock.parent_ids)
        for key, farm in world.farms.items() if key != "player"
        for rock in farm.rocks.values() if rock.parent_ids
    )
    world.validate_ownership()
    assert len(world.owner_by_rock_id) == len(set(world.owner_by_rock_id))


def test_random_world_size_is_seeded_and_bounded():
    left = game_controller.start_new_game(seed=707, world_size_mode="random", allow_neural_farmers=False)
    right = game_controller.start_new_game(seed=707, world_size_mode="random", allow_neural_farmers=False)
    assert 3 <= left.world.resolved_npc_count <= 8
    assert left.world.resolved_npc_count == right.world.resolved_npc_count
    assert game_controller.get_public_farms(left) == game_controller.get_public_farms(right)


def test_production_candidates_exclude_synthetic_rock_sources_and_sink():
    game = _game()
    manager = PlayableWorldManager()
    candidates = manager.generator.generate(game.world, "player")
    types = {candidate.action.action_type for candidate in candidates}
    assert FarmerActionType.IMPORT_RANDOM_ROCK not in types
    assert FarmerActionType.IMPORT_REQUESTED_ROCK not in types
    assert FarmerActionType.SELL_ROCK not in types


def test_family_pod_purchase_transfers_existing_child_and_money():
    game = _game(seed=44)
    pod = game_controller.get_family_pod_rows(game)[0]
    child = pod["children"][0]
    world = game.world
    seller_id = pod["seller_farm_id"]
    player_money = game.money
    seller_money = world.farm(seller_id).money
    result = game_controller.purchase_family_pod_child(game, pod["pod_id"], child.id)
    assert result.ok
    assert world.owner_of(child.id) == "player"
    assert game.get_rock(child.id) is child
    assert game.money == player_money - pod["price"]
    assert world.farm(seller_id).money == seller_money + pod["price"]
    assert all(world.reserved_rock_ids.get(rock.id) != pod["pod_id"] for rock in pod["children"])


def test_player_listing_receives_actionable_farmer_bid_message():
    game = _game(seed=51)
    rock = game_controller.get_active_rocks(game)[0]
    asking = game_controller.get_listing_price_options(game, rock.id)[0]
    listing_result = game_controller.create_player_listing(game, rock.id, asking)
    assert listing_result.ok
    listing = next(row for row in game.world.listings.values() if row.seller_farm_id == "player")
    bidder_id = sorted(farm_id for farm_id in game.world.farms if farm_id != "player")[0]
    bid = PlaceBidAction(bidder_id, game.world.turn, listing.listing_id, asking)
    result = EconomyTransactionManager().execute(game.world, bid, "farmer_bid_test")
    assert result.success
    message = game_controller.get_player_messages(game, unread_only=True)[0]
    assert message["kind"] == "bid_received" and message["requires_response"]
    accepted = game_controller.respond_to_message(game, message["message_id"], True)
    assert accepted.ok
    assert game.world.owner_of(rock.id) == bidder_id


def test_one_world_turn_per_call_and_generation_lead_is_preserved():
    game = _game(seed=15)
    before_generations = {farm_id: farm.generation for farm_id, farm in game.world.farms.items() if farm_id != "player"}
    first = game_controller.end_world_turn(game)
    assert first.ok and game.world.turn == 1
    assert len(first.payload["results"]) == len(game.world.farms)
    second = game_controller.end_world_turn(game)
    assert second.ok and game.world.turn == 2

    active = game_controller.get_available_breeding_candidates(game)
    pair = next(
        (left, right) for left in active for right in active
        if left.id < right.id and game_controller.validate_breeding_pair(game, left.id, right.id)["valid"]
    )
    assert game_controller.breed_pair(game, pair[0].id, pair[1].id).ok
    assert game_controller.advance_breeding_generation(game).ok
    assert all(game.world.farm(farm_id).generation == generation + 1 for farm_id, generation in before_generations.items())
    game.world.validate_ownership()


def test_world_save_roundtrip_and_legacy_migration_are_deterministic():
    game = _game(seed=22, count=2)
    game_controller.end_world_turn(game)
    loaded = game_from_json_string(game_to_json_string(game))
    assert loaded.world.farm("player").game is loaded
    assert loaded.world.turn == 1
    assert loaded.world.resolved_npc_count == 2
    assert loaded.world.owner_by_rock_id == game.world.owner_by_rock_id

    legacy = game_to_dict(GameMaster(seed=303))
    legacy.pop("world", None)
    first = game_from_dict(copy.deepcopy(legacy))
    second = game_from_dict(copy.deepcopy(legacy))
    assert 3 <= first.world.resolved_npc_count <= 8
    assert game_controller.get_public_farms(first) == game_controller.get_public_farms(second)
    assert set(first.rocks) == set(second.rocks) == {1, 2, 3, 4}


def test_maximum_world_completes_a_turn_without_collisions():
    game = _game(seed=909, count=12)
    result = game_controller.end_world_turn(game)
    assert result.ok
    assert len(result.payload["results"]) == 13
    game.world.validate_ownership()
