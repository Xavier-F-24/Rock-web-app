from dataclasses import dataclass


@dataclass(frozen=True)
class PublicFarmObservation:
    farm_id: str
    display_name: str
    generation: int
    visible_rock_ids: tuple[int, ...]
    visible_rock_values: tuple[int, ...]
    recent_public_events: tuple[str, ...] = ()
