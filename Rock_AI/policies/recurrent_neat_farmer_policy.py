"""Player-safe recurrent NEAT scoring across all legal action types."""

from __future__ import annotations

import math
from pathlib import Path

from Rock_AI.actions.action_explanation import ActionExplanation
from Rock_AI.models.model_trace_helper import ModelTrace
from Rock_AI.neat.neat_export_helper import load_recurrent_artifact
from Rock_AI.neat.neat_recurrent_network import RecurrentNeatNetwork
from Rock_AI.neat.neat_state_helper import RecurrentAgentState
from Rock_AI.observations.full_farmer_observation import FullFarmerObservation

from .legal_action_scoring_policy import ActionRankingDecision, RankedAction


FULL_FARMER_OBSERVATION_SCHEMA_VERSION = 1


class RecurrentNeatFarmerPolicy:
    def __init__(self, artifact, *, checkpoint_id: str, tie_threshold: float = 0.05):
        if artifact.information_access != "player":
            raise ValueError("Privileged topology cannot run as a farmer")
        if artifact.observation_schema_version != FULL_FARMER_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Topology is not compatible with the full-farmer observation schema")
        if artifact.metadata.get("policy_kind") != "full_farmer":
            raise ValueError("Breeding-only topology cannot run as a full farmer")
        self.artifact = artifact
        self.network = RecurrentNeatNetwork(artifact)
        self.checkpoint_id = checkpoint_id
        self.tie_threshold = float(tie_threshold)
        self.state = self.network.initial_state()
        self.pending_candidate = None
        self.pending_score_map = {}
        self.latest_model_trace = None

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        return cls(load_recurrent_artifact(path), checkpoint_id=str(path))

    def reset(self, episode_id: str = "economy") -> None:
        self.state = self.network.initial_state(episode_id)
        self.pending_candidate = None
        self.pending_score_map = {}
        self.latest_model_trace = None

    def export_state(self) -> dict:
        return self.state.to_dict()

    def import_state(self, payload: dict) -> None:
        state = RecurrentAgentState.from_dict(payload)
        state.validate_for(self.artifact.topology_id, self.artifact.genome_id)
        self.state = state

    def _vector(self, candidate):
        vector = tuple(map(float, candidate.model_values()))
        if len(vector) != len(self.artifact.input_feature_names):
            raise ValueError("Action candidate feature width is incompatible with topology")
        return vector

    def _result_vector(self, candidate, visible_result, *, delayed=False):
        values = list(candidate.values)
        names = candidate.feature_names
        payload = getattr(visible_result, "public_payload", None) or getattr(visible_result, "payload", None) or {}
        money_delta = float(payload.get("proceeds", payload.get("price", 0))) - float(payload.get("cost", 0))
        updates = {
            "result.success": float(bool(getattr(visible_result, "success", True))),
            "result.money_delta": money_delta,
            "result.asset_delta": float(len(getattr(visible_result, "affected_rock_ids", getattr(visible_result, "rock_ids", ())))),
            "result.delayed_resolution": float(delayed),
        }
        for name, value in updates.items():
            values[names.index(name)] = value
        return tuple(values) + tuple(float(value) for value in candidate.visibility_mask)

    def rank_actions(self, observation: FullFarmerObservation) -> ActionRankingDecision:
        if not isinstance(observation, FullFarmerObservation):
            raise TypeError("Full farmer policy requires FullFarmerObservation")
        snapshot = self.state
        scored = []
        traces = []
        for candidate in observation.legal_candidates:
            result = self.network.activate(self._vector(candidate), snapshot, commit=False)
            score = float(result.outputs[0])
            if not math.isfinite(score):
                raise ValueError("Network produced a non-finite action score")
            scored.append((candidate, score))
            traces.append(result.trace)
        ordered = sorted(scored, key=lambda row: (-row[1], row[0].candidate_hash))
        ranked = tuple(RankedAction(candidate, score, index + 1) for index, (candidate, score) in enumerate(ordered))
        if not ranked:
            return ActionRankingDecision((), None, None)
        gap = ranked[0].score - ranked[1].score if len(ranked) > 1 else float("inf")
        confidence = 1.0 if len(ranked) == 1 else 1.0 / (1.0 + math.exp(-gap))
        selected = ranked[0].candidate
        summary = f"Selected {selected.action.action_type.value} as the highest-scoring legal action."
        explanation = ActionExplanation(
            selected.candidate_hash, summary, ranked[0].score, 1, len(ranked), confidence,
            observations=("Candidate was generated from player-visible state only.",),
            warnings=(("Top actions are nearly tied",) if gap <= self.tie_threshold else ()),
        )
        self.pending_candidate = selected
        self.pending_score_map = {row.candidate.candidate_hash: row.score for row in ranked}
        return ActionRankingDecision(ranked, selected, explanation, {
            "memory_before": snapshot.to_dict(), "candidate_evaluations_committed": 0,
            "output_scores": self.pending_score_map,
        })

    def commit_selected(self, selected, visible_result) -> None:
        if self.pending_candidate is None or selected.candidate_hash != self.pending_candidate.candidate_hash:
            raise ValueError("Only the selected candidate may commit recurrent memory")
        result_vector = self._result_vector(selected, visible_result)
        result = self.network.activate(result_vector, self.state, commit=True)
        before = self.state
        self.state = result.state
        self.latest_model_trace = ModelTrace(
            model_type="recurrent_neat_full_farmer", checkpoint_id=self.checkpoint_id,
            topology_id=self.artifact.topology_id, observation_schema_version=FULL_FARMER_OBSERVATION_SCHEMA_VERSION,
            normalizer_version=self.artifact.normalizer_version, observation_hash=selected.candidate_hash,
            feature_names=self.artifact.input_feature_names, input_values=result_vector,
            node_activations=result.trace["node_activations"], connection_signals=tuple(result.trace["connection_signals"]),
            output_scores=self.pending_score_map, selected_candidate_ids=(selected.candidate_hash,),
            candidate_input_hashes=tuple(sorted(self.pending_score_map)),
            metadata={"memory_before": before.to_dict(), "memory_after": self.state.to_dict(), "visible_result": getattr(visible_result, "public_payload", {})},
        )
        self.pending_candidate = None

    def commit_visible_resolution(self, selected, visible_resolution) -> None:
        """Commit a later bid/offer resolution only once it becomes observable."""
        result_vector = self._result_vector(selected, visible_resolution, delayed=True)
        result = self.network.activate(result_vector, self.state, commit=True)
        self.state = result.state
