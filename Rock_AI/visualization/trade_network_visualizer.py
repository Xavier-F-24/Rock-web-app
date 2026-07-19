def trade_network_rows(world):
    return tuple({"source": offer.sender_farm_id, "target": offer.recipient_farm_id, "status": offer.status.value, "offered_money": offer.offered_money, "requested_money": offer.requested_money, "offered_rocks": list(offer.offered_rock_ids), "requested_rocks": list(offer.requested_rock_ids)} for offer in world.trade_offers.values())
