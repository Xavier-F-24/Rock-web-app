from pathlib import Path

from Rock_World import (
    PLAYER_OWNER_ID,
    FarmMessage,
    FarmProfile,
    FarmState,
    MarketListing,
    TradeOffer,
    WorldState,
    create_default_farm_profiles,
    create_empty_default_world,
    farm_owner_id,
    is_farm_owner,
)


def test_default_farm_profiles_create_three_distinct_npc_farms():
    profiles = create_default_farm_profiles()

    assert len(profiles) == 3
    assert len({profile.farm_id for profile in profiles}) == 3
    assert [profile.difficulty for profile in profiles] == ["easy", "medium", "hard"]
    assert all(profile.owner_id == farm_owner_id(profile.farm_id) for profile in profiles)
    assert all(profile.starting_generation_offset in {1, 2} for profile in profiles)


def test_world_dataclasses_can_be_instantiated_and_reserve_ids():
    profile = FarmProfile(
        farm_id="test_farm",
        farm_name="Test Farm",
        owner_name="Test Farmer",
        region="Test Ridge",
    )
    farm = FarmState(profile=profile)
    world = WorldState()

    world.add_farm(farm)
    listing = MarketListing(
        listing_id=world.reserve_listing_id(),
        seller_owner_id=farm.owner_id,
        rock_id=world.reserve_world_rock_id(),
        price=12,
        created_generation=0,
    )
    offer = TradeOffer(
        offer_id=world.reserve_offer_id(),
        from_owner_id=farm.owner_id,
        to_owner_id=PLAYER_OWNER_ID,
        offered_money=15,
        requested_rock_ids=[1],
    )
    message = FarmMessage(
        message_id=world.reserve_message_id(),
        from_owner_id=farm.owner_id,
        text="We admire rock #1.",
        related_offer_id=offer.offer_id,
    )

    world.market_listings.append(listing)
    world.trade_offers.append(offer)
    world.messages.append(message)

    assert world.get_farm("test_farm") is farm
    assert listing.is_open is True
    assert offer.is_pending is True
    assert message.to_owner_id == PLAYER_OWNER_ID
    assert is_farm_owner(farm.owner_id) is True
    assert is_farm_owner(PLAYER_OWNER_ID) is False


def test_empty_default_world_has_profiles_but_no_generated_rocks_yet():
    world = create_empty_default_world()

    assert len(world.farms) == 3
    assert all(not farm.rocks for farm in world.farms.values())
    assert world.market_listings == []
    assert world.trade_offers == []
    assert world.messages == []


def test_world_core_modules_do_not_import_streamlit():
    project_root = Path(__file__).resolve().parents[1]
    world_files = list((project_root / "Rock_World").glob("*.py"))
    assert world_files

    for path in world_files:
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text
