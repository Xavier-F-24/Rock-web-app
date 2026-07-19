def market_efficiency_metrics(world):
    listings = list(world.listings.values())
    sold = [listing for listing in listings if listing.status.value == "sold"]
    return {"listing_count": len(listings), "completed_count": len(sold), "completion_rate": len(sold) / max(1, len(listings))}
