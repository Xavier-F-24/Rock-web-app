"""Player-safe NEAT candidate scorer backed only by JSON network artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Rock_AI.models.model_trace_helper import ModelTrace
from Rock_AI.models.neat_network_helper import InstrumentedNeatNetwork, load_neat_artifact
from Rock_AI.policies.neural_pair_ranking_policy import PairRankingDecision, RankedPairDecision
from Rock_AI.representations.player_candidate_helper import neat_symmetric_candidate_vector
from Rock_AI.representations.player_observation_helper import PLAYER_OBSERVATION_SCHEMA_VERSION, PlayerObservation


class NeatPairRankingPolicy:
    def __init__(self, artifact, *, checkpoint_id: str, tie_warning_threshold: float = 0.05):
        if artifact.information_access != "player":
            raise ValueError("Privileged NEAT artifacts cannot run as player agents")
        if artifact.observation_schema_version != PLAYER_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("NEAT player-observation schema is incompatible")
        self.artifact = artifact
        self.network = InstrumentedNeatNetwork(artifact)
        self.checkpoint_id = checkpoint_id
        self.tie_warning_threshold = float(tie_warning_threshold)
        self.latest_model_trace: ModelTrace | None = None

    @classmethod
    def load(cls, artifact_path: str | Path) -> "NeatPairRankingPolicy":
        path = Path(artifact_path)
        return cls(load_neat_artifact(path), checkpoint_id=str(path))

    def rank_observation(self, observation: PlayerObservation) -> PairRankingDecision:
        if not isinstance(observation, PlayerObservation):
            raise TypeError("NeatPairRankingPolicy requires PlayerObservation")
        if observation.schema_version != self.artifact.observation_schema_version:
            raise ValueError("Observation schema does not match NEAT artifact")
        if observation.normalizer_version != self.artifact.normalizer_version:
            raise ValueError("Observation normalizer does not match NEAT artifact")
        if not observation.candidates:
            return PairRankingDecision((), None, 0.0, no_action_reason="No legal breeding pairs")
        scored = []
        traces = []
        vectors = []
        for candidate in observation.candidates:
            vector = neat_symmetric_candidate_vector(candidate)
            if len(vector) != len(self.artifact.input_feature_names):
                raise ValueError("Candidate feature dimension is incompatible with NEAT artifact")
            output, trace = self.network.activate(vector)
            scored.append(float(output[0]))
            traces.append(trace)
            vectors.append(vector)
        order = sorted(
            range(len(scored)),
            key=lambda index: (-scored[index], tuple(map(str, observation.candidates[index].canonical_parent_ids))),
        )
        ranked = tuple(
            RankedPairDecision(
                observation.candidates[index].canonical_parent_ids,
                scored[index],
                score_components={"candidate_hash": observation.candidates[index].candidate_hash},
            )
            for index in order
        )
        selected_index = order[0]
        selected_trace = traces[selected_index]
        self.latest_model_trace = ModelTrace(
            model_type="neat_pair_ranker",
            checkpoint_id=self.checkpoint_id,
            topology_id=self.artifact.topology_id,
            observation_schema_version=observation.schema_version,
            normalizer_version=observation.normalizer_version,
            observation_hash=observation.observation_hash,
            feature_names=self.artifact.input_feature_names,
            input_values=tuple(map(float, vectors[selected_index])),
            node_activations=selected_trace["node_activations"],
            connection_signals=tuple(selected_trace["connection_signals"]),
            output_scores={
                "|".join(map(str, observation.candidates[index].canonical_parent_ids)): scored[index]
                for index in range(len(scored))
            },
            selected_candidate_ids=ranked[0].parent_ids,
            candidate_input_hashes=tuple(candidate.candidate_hash for candidate in observation.candidates),
            metadata={"network_generation": self.artifact.metadata.get("generation")},
        )
        gap = ranked[0].neural_score - ranked[1].neural_score if len(ranked) > 1 else float("inf")
        confidence = 1.0 if len(ranked) == 1 else float(1.0 / (1.0 + np.exp(-gap)))
        warning = "Top candidate pairs are nearly tied" if len(ranked) > 1 and gap <= self.tie_warning_threshold else None
        return PairRankingDecision(ranked, ranked[0].parent_ids, confidence, warning)

    def rank_legal_pairs(self, *args, **kwargs):
        raise TypeError("Raw farms are not accepted; use rank_observation(PlayerObservation)")
