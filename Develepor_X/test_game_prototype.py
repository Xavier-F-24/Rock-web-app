import Rock_Genetics.rock_genetic_helper as genetics
from Rock_GameState.rock_game_state_helper import GameMaster, create_new_game


def test_game_master_can_start_breed_advance_and_show_rocks():
    game = GameMaster(seed=7)

    assert isinstance(game, GameMaster)
    assert len(game.rock_list) == 4
    assert game.money == 10
    assert game.market_pods

    males = [rock for rock in game.rock_list.values() if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in game.rock_list.values() if rock.sex == genetics.Sex.FEMALE]

    game.add_pair_to_queue(males[0].id, females[0].id)
    children = game.advance_generation()

    assert children
    assert game.generation == 1
    assert len(game.rock_list) > 4
    assert game.breeding_queue == []
    assert game.show_rocks()


def test_market_manager_can_sell_buy_potion_and_defined_trait_rock():
    game = create_new_game(seed=11, starting_money=30)
    sellable_id = next(iter(game.rock_list))

    sold_value = game.sell_rock(sellable_id)
    assert sold_value >= 1
    assert game.money >= 31

    game.buy_potion("fertility")
    assert game.potions["fertility"] == 1

    rock = game.buy_defined_trait_rock(
        {
            "gender": "female",
            "color": "34",
            "eyes": "11",
            "mouths": "22",
        },
        random_fill=False,
    )

    assert rock.id in game.rock_list
    assert rock.sex == genetics.Sex.FEMALE
    assert rock.genotype.genes["color"].phenotype == "orange"
    assert rock.genotype.genes["eyes"].phenotype == "double eye"


def test_market_pod_purchase_can_keep_one_child():
    game = create_new_game(seed=13, starting_money=30)
    offer = game.market_pods[0]

    pending = game.market_manager.buy_market_pod(game, offer.offer_id)
    assert pending.children
    assert game.pending_market_pod is pending

    child = game.market_manager.choose_market_pod_child(game, 0)

    assert child.id in game.rock_list
    assert child.parent_ids == [pending.parent_a_id, pending.parent_b_id]
    assert game.pending_market_pod is None
