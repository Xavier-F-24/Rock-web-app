"""Stateful player-safe recurrent NEAT candidate policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Rock_AI.models.model_trace_helper import ModelTrace
from Rock_AI.neat.neat_export_helper import load_recurrent_artifact
from Rock_AI.neat.neat_recurrent_network import RecurrentNeatNetwork, RecurrentNumericalError
from Rock_AI.neat.neat_state_helper import RecurrentAgentState, RecurrentDecisionObservation
from Rock_AI.policies.neural_pair_ranking_policy import PairRankingDecision, RankedPairDecision
from Rock_AI.representations.player_candidate_helper import neat_symmetric_candidate_vector
from Rock_AI.representations.player_observation_helper import PLAYER_OBSERVATION_SCHEMA_VERSION


class RecurrentNeatPairRankingPolicy:
    """Score all candidates from one memory snapshot, then commit only the winner."""

    def __init__(self, artifact, *, checkpoint_id: str, tie_warning_threshold: float = 0.05):
        if artifact.information_access != "player":
            raise ValueError("Privileged recurrent artifacts cannot run as player agents")
        if artifact.observation_schema_version != PLAYER_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Recurrent artifact uses an incompatible player-observation schema")
        self.artifact = artifact
        self.network = RecurrentNeatNetwork(artifact)
        self.checkpoint_id = checkpoint_id
        self.tie_warning_threshold = float(tie_warning_threshold)
        self.state = self.network.initial_state()
        self.latest_model_trace: ModelTrace | None = None

    @classmethod
    def load(cls, artifact_path: str | Path) -> "RecurrentNeatPairRankingPolicy":
        path = Path(artifact_path)
        return cls(load_recurrent_artifact(path), checkpoint_id=str(path))

    def reset(self, episode_id: str = "episode") -> None:
        self.state = self.network.initial_state(episode_id)
        self.latest_model_trace = None

    def export_state(self) -> dict:
        return self.state.to_dict()

    def import_state(self, payload: dict) -> None:
        state = RecurrentAgentState.from_dict(payload)
        state.validate_for(self.artifact.topology_id, self.artifact.genome_id)
        self.state = state

    def _vector(self, observation: RecurrentDecisionObservation, candidate) -> np.ndarray:
        return np.concatenate((
            neat_symmetric_candidate_vector(candidate),
            np.asarray(observation.temporal_context.model_values(), dtype=np.float64),
        ))

    def rank_observation(self, observation: RecurrentDecisionObservation) -> PairRankingDecision:
        if not isinstance(observation, RecurrentDecisionObservation):
            raise TypeError("Recurrent policy requires RecurrentDecisionObservation")
        player = observation.player_observation
        if player.schema_version != self.artifact.observation_schema_version:
            raise ValueError("Observation schema does not match recurrent artifact")
        if player.normalizer_version != self.artifact.normalizer_version:
            raise ValueError("Observation normalizer does not match recurrent artifact")
        if not player.candidates:
            return PairRankingDecision((), None, 0.0, no_action_reason="No legal breeding pairs")
        memory_before = self.state
        scored: list[float] = []
        vectors: list[np.ndarray] = []
        trial_traces = []
        for candidate in player.candidates:
            vector = self._vector(observation, candidate)
            if len(vector) != len(self.artifact.input_feature_names):
                raise ValueError("Candidate feature dimension is incompatible with recurrent artifact")
            result = self.network.activate(
                vector, memory_before, commit=False,
                temporal_context=observation.temporal_context.values,
            )
            scored.append(float(result.outputs[0]))
            vectors.append(vector)
            trial_traces.append(result.trace)
        order = sorted(
            range(len(scored)),
            key=lambda index: (-scored[index], tuple(map(str, player.candidates[index].canonical_parent_ids))),
        )
        ranked = tuple(RankedPairDecision(
            player.candidates[index].canonical_parent_ids,
            scored[index],
            score_components={"candidate_hash": player.candidates[index].candidate_hash},
        ) for index in order)
        selected_index = order[0]
        committed = self.network.activate(
            vectors[selected_index], memory_before, commit=True,
            temporal_context=observation.temporal_context.values,
        )
        self.state = committed.state
        self.latest_model_trace = ModelTrace(
            model_type="recurrent_neat_pair_ranker", checkpoint_id=self.checkpoint_id,
            topology_id=self.artifact.topology_id,
            observation_schema_version=player.schema_version,
            normalizer_version=player.normalizer_version,
            observation_hash=player.observation_hash,
            feature_names=self.artifact.input_feature_names,
            input_values=tuple(map(float, vectors[selected_index])),
            node_activations=committed.trace["node_activations"],
            connection_signals=tuple(committed.trace["connection_signals"]),
            output_scores={"|".join(map(str, player.candidates[i].canonical_parent_ids)): scored[i] for i in range(len(scored))},
            selected_candidate_ids=ranked[0].parent_ids,
            candidate_input_hashes=tuple(candidate.candidate_hash for candidate in player.candidates),
            metadata={
                "network_generation": self.artifact.metadata.get("generation"),
                "memory_before": memory_before.to_dict(),
                "memory_after": self.state.to_dict(),
                "settling_steps": committed.trace["settling_steps"],
                "candidate_evaluations_committed": 1,
            },
        )
        gap = ranked[0].neural_score - ranked[1].neural_score if len(ranked) > 1 else float("inf")
        output_confidence = committed.outputs[2] if len(committed.outputs) > 2 else gap
        confidence = 1.0 if len(ranked) == 1 else float(1.0 / (1.0 + np.exp(-output_confidence)))
        warning = "Top candidate pairs are nearly tied" if len(ranked) > 1 and gap <= self.tie_warning_threshold else None
        return PairRankingDecision(ranked, ranked[0].parent_ids, confidence, warning)

    def rank_legal_pairs(self, *args, **kwargs):
        raise TypeError("Raw farms are not accepted; use rank_observation(RecurrentDecisionObservation)")
