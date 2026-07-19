def trade_metrics(world):
    offers = list(world.trade_offers.values())
    return {"proposed": len(offers), "accepted": sum(offer.status.value == "accepted" for offer in offers)}
