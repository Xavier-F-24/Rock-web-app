from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerInventoryObservation:
    farm_id: str
    money: int
    committed_money: int
    rock_ids: tuple[int, ...]
    visible_rock_summaries: tuple[tuple[int, float, float, int, str, str], ...]
    potions: tuple[tuple[str, int], ...]
