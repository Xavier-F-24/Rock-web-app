import plotly.graph_objects as go

from rockgame_ui import game_controller


def test_controller_starts_game_and_returns_ui_rows():
    game = game_controller.start_new_game(seed=701)

    summary = game_controller.get_game_summary(game)
    rows = game_controller.get_rock_rows(game)
    active_rows = game_controller.get_active_rock_rows(game)
    sellable_rows = game_controller.get_sellable_rock_rows(game)

    assert summary["rock_count"] == 4
    assert summary["max_generation"] == game.max_generation
    assert len(rows) == 4
    assert len(active_rows) == 4
    assert {"id", "name", "sex", "generation", "status", "sell_value"}.issubset(rows[0])
    assert {"id", "name", "sex", "generation", "value", "sell_value"}.issubset(active_rows[0])
    assert all(row["sell_value"] > 0 for row in sellable_rows)


def test_controller_starts_game_with_custom_settings_and_scores():
    settings = game_controller.GameStartSettings(
        seed=709,
        starting_money=55,
        max_generation=2,
        max_pairs_per_generation=1,
        rock_farm_cost=80,
    )

    game = game_controller.start_new_game(settings=settings)
    score = game_controller.get_final_score_summary(game)

    assert game.seed == 709
    assert game.money == 55
    assert game.max_generation == 2
    assert game.max_pairs_per_generation == 1
    assert game.rock_farm_cost == 80
    assert score["final_score"] == score["active_rock_score"] + score["money"] - 80
    assert game_controller.is_game_finished(game) is False


def test_controller_detects_finished_game_after_max_generation():
    game = game_controller.start_new_game(
        settings={
            "seed": 710,
            "starting_money": 80,
            "max_generation": 1,
            "max_pairs_per_generation": 1,
            "rock_farm_cost": 75,
        }
    )
    breeders = game_controller.get_available_breeding_candidates(game)
    male = next(rock for rock in breeders if rock.sex.value == "male")
    female = next(rock for rock in breeders if rock.sex.value == "female")

    game_controller.breed_pair(game, male.id, female.id)
    result = game_controller.advance_breeding_generation(game)

    assert result.ok is True
    assert game_controller.is_game_finished(game) is True


def test_controller_wraps_basic_actions():
    game = game_controller.start_new_game(seed=702)

    buy_result = game_controller.buy_random_rock(game)
    sell_result = game_controller.sell_rock(game, next(iter(game.rocks)))

    assert buy_result.ok is False
    assert "farmer-owned" in buy_result.message
    assert sell_result.ok is False
    assert "listing" in sell_result.message
    assert game_controller.get_market_listings(game)


def test_controller_buys_potions_in_bulk():
    game = game_controller.start_new_game(seed=707)

    result = game_controller.buy_potions(game, {"fertility": 2, "mutation": 0})

    assert result.ok is True
    assert game.potions["fertility"] == 2
    assert game.money == game.starting_money - 6


def test_controller_buys_market_pod_and_keeps_child():
    game = game_controller.start_new_game(seed=708)
    pod = game_controller.get_family_pod_rows(game)[0]
    child = pod["children"][0]

    buy_result = game_controller.purchase_family_pod_child(game, pod["pod_id"], child.id)

    assert buy_result.ok is True
    assert game.world.owner_of(child.id) == "player"
    assert child.id in game.rocks


def test_controller_queues_breeding_pair_with_clean_result():
    game = game_controller.start_new_game(seed=703)
    breeders = game_controller.get_breedable_rocks(game)
    male = next(rock for rock in breeders if rock.sex.value == "male")
    female = next(rock for rock in breeders if rock.sex.value == "female")

    result = game_controller.breed_pair(game, male.id, female.id)

    assert result.ok is True
    assert result.payload.parent_a_id == male.id
    assert game_controller.get_queue_summary(game)["queued_pairs"] == 1


def test_controller_removes_queued_pair_and_refunds_potion():
    game = game_controller.start_new_game(seed=711, starting_money=80)
    game.buy_potion("fertility")
    game.buy_potion("mutation")
    breeders = game_controller.get_available_breeding_candidates(game)
    male = next(rock for rock in breeders if rock.sex.value == "male")
    female = next(rock for rock in breeders if rock.sex.value == "female")

    queue_result = game_controller.breed_pair(
        game,
        male.id,
        female.id,
        options={"potion_keys": ["fertility", "mutation"]},
    )
    badge_map = game_controller.get_queued_parent_badges(game)
    remove_result = game_controller.remove_queued_pair(game, 1)

    assert queue_result.ok is True
    assert queue_result.payload.potion_keys == ["fertility", "mutation"]
    assert male.id in badge_map
    assert female.id in badge_map
    assert badge_map[male.id]["text"] == "\u2665"
    assert remove_result.ok is True
    assert game_controller.get_queue_summary(game)["queued_pairs"] == 0
    assert game.potions["fertility"] == 1
    assert game.potions["mutation"] == 1


def test_controller_advances_queued_breeding_and_returns_children():
    game = game_controller.start_new_game(seed=706)
    breeders = game_controller.get_available_breeding_candidates(game)
    male = next(rock for rock in breeders if rock.sex.value == "male")
    female = next(rock for rock in breeders if rock.sex.value == "female")

    queue_result = game_controller.breed_pair(game, male.id, female.id)
    advance_result = game_controller.advance_breeding_generation(game)

    assert queue_result.ok is True
    assert advance_result.ok is True
    assert game_controller.get_queue_summary(game)["queued_pairs"] == 0
    assert game.generation == 1
    assert len(advance_result.payload) > 0
    assert all(row["generation"] == 1 for row in advance_result.payload)


def test_controller_renders_tree_and_rock():
    game = game_controller.start_new_game(seed=704)
    rock = next(iter(game.rocks.values()))

    fig = game_controller.render_tree_for_streamlit(game, canvas_width=500, canvas_height=400)
    uri = game_controller.render_rock(rock, sprite_size=0.8, dpi=60)

    assert isinstance(fig, go.Figure)
    assert uri.startswith("data:image/png;base64,")


def test_controller_serializes_and_loads_game():
    game = game_controller.start_new_game(seed=705)

    save_json = game_controller.serialize_game(game)
    loaded = game_controller.load_game_from_json(save_json)

    assert len(loaded.rocks) == len(game.rocks)
    assert loaded.generation == game.generation
