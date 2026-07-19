def price_to_value_ratio(price: int, appraised_value: int) -> float:
    return float(price) / max(1.0, float(appraised_value))
