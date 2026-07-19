from dataclasses import dataclass


@dataclass(frozen=True)
class PrivateObservationRecord:
    farm_id: str
    world_turn: int
    observation_hash: str
    candidate_hashes: tuple[str, ...]
    schema_version: int
