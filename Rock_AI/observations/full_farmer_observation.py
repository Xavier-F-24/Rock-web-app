from dataclasses import dataclass

from Rock_AI.actions.action_candidate import ActionCandidate
from Rock_AI.neat.neat_state_helper import RecurrentAgentState

from .player_economy_observation import PlayerEconomyObservation


@dataclass(frozen=True)
class FullFarmerObservation:
    economy: PlayerEconomyObservation
    legal_candidates: tuple[ActionCandidate, ...]
    recurrent_state_snapshot: RecurrentAgentState | None = None
