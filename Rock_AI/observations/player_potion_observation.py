from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerPotionObservation:
    inventory: tuple[tuple[str, int], ...]
    public_definitions: tuple[tuple[str, int, str], ...]
