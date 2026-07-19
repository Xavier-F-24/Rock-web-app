def trade_is_self_dealing(offer) -> bool:
    return offer.sender_farm_id == offer.recipient_farm_id
