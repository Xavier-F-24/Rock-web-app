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
    assert rock.is_market is True
    assert rock.sex == genetics.Sex.FEMALE
    assert rock.genotype.genes["color"].phenotype == "orange"
    assert rock.genotype.genes["eyes"].phenotype == "double eye"


def test_game_master_can_queue_pair_with_multiple_potion_types_and_refund_them():
    game = GameMaster(seed=37, starting_money=80)
    game.buy_potion("fertility")
    game.buy_potion("mutation")
    ids = list(game.rocks)

    pair = game.add_pair_to_queue(ids[0], ids[1], potion_keys=["fertility", "mutation"])

    assert pair.potion_keys == ["fertility", "mutation"]
    assert "fertility" not in game.potions
    assert "mutation" not in game.potions

    game.remove_pair_from_queue(0)

    assert game.potions["fertility"] == 1
    assert game.potions["mutation"] == 1


def test_market_pod_purchase_can_keep_one_child():
    game = create_new_game(seed=13, starting_money=30)
    offer = game.market_pods[0]

    pending = game.market_manager.buy_market_pod(game, offer.offer_id)
    assert pending.children
    assert game.pending_market_pod is pending

    child = game.market_manager.choose_market_pod_child(game, 0)

    assert child.id in game.rock_list
    assert child.is_market is True
    assert child.parent_ids == [pending.parent_a_id, pending.parent_b_id]
    assert game.pending_market_pod is None


def test_game_master_queues_related_pair_with_warning_event():
    game = GameMaster(seed=17, auto_start=False)
    parent_a = genetics.Rock(
        id=1,
        name=game.name_generator.generate_name(genetics.Sex.MALE),
        sex=genetics.Sex.MALE,
        genotype=game.genome_factory.make_random_rock_genome(),
        death_genes=game.genome_factory.make_death_genes(),
    )
    parent_b = genetics.Rock(
        id=2,
        name=game.name_generator.generate_name(genetics.Sex.FEMALE),
        sex=genetics.Sex.FEMALE,
        genotype=game.genome_factory.make_random_rock_genome(),
        death_genes=game.genome_factory.make_death_genes(),
    )
    sibling_a = genetics.Rock(
        id=3,
        name=game.name_generator.generate_name(genetics.Sex.MALE),
        sex=genetics.Sex.MALE,
        genotype=game.genome_factory.make_random_rock_genome(),
        death_genes=game.genome_factory.make_death_genes(),
        parent_ids=[parent_a.id, parent_b.id],
    )
    sibling_b = genetics.Rock(
        id=4,
        name=game.name_generator.generate_name(genetics.Sex.FEMALE),
        sex=genetics.Sex.FEMALE,
        genotype=game.genome_factory.make_random_rock_genome(),
        death_genes=game.genome_factory.make_death_genes(),
        parent_ids=[parent_a.id, parent_b.id],
    )
    game.rock_list = {rock.id: rock for rock in [parent_a, parent_b, sibling_a, sibling_b]}

    game.add_pair_to_queue(sibling_a.id, sibling_b.id)

    assert game.breeding_queue
    assert any("Parents are related" in event and "R=0.5000" in event for event in game.events)


def test_potion_settings_are_finalized_for_first_balance_pass():
    settings = GameMaster.potion_settings("fertility")
    assert settings["clutch_plus_one"] is True
    assert settings["clutch_reroll"] is None

    settings = GameMaster.potion_settings("reroll")
    assert settings["clutch_reroll"] is True
    assert settings["clutch_plus_one"] is None

    settings = GameMaster.potion_settings("mutation")
    assert settings["mutation_chance"] > GameMaster.potion_settings(None)["mutation_chance"]

    settings = GameMaster.potion_settings("anti_craisen")
    assert settings["craisen_chance"] == 0.0
