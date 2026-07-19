from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorDescriptor:
    breed_fraction: float
    import_fraction: float
    potion_fraction: float
    market_fraction: float
    trade_fraction: float
    pass_fraction: float
