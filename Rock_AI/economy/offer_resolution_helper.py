def open_offers_for(world, farm_id: str):
    return tuple(offer for offer in world.trade_offers.values() if offer.recipient_farm_id == farm_id and offer.status.value == "open")
