"""Structured explanation assembled from explicit scores and game mechanics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from Rock_AI.agents.breeding_agent_helper import AgentAction, BreedPairAction
from Rock_AI.datasets.pair_ranking_record_helper import FarmerObjectiveProfile

from .candidate_explanation_helper import CandidateExplanation, build_candidate_explanations


@dataclass(frozen=True)
class DecisionExplanation:
    selected_action: dict[str, Any]
    selected_parent_ids: tuple[int | str, int | str] | None
    selected_parent_names: tuple[str, str] | None
    selected_candidate_score: float | None
    objective_profile: dict[str, Any]
    predicted_offspring_summary: dict[str, float | None]
    score_component_contributions: dict[str, float]
    selected_pair_rank: int | None
    total_legal_candidates: int
    first_second_score_difference: float | None
    confidence_proxy: float | None
    uncertainty_penalty: float | None
    notable_genetics_observations: tuple[str, ...] = ()
    rejected_alternatives: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_reason: str | None = None
    top_candidates: tuple[CandidateExplanation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decision_explanation_from_dict(data: dict[str, Any]) -> DecisionExplanation:
    values = dict(data)
    if values.get("selected_parent_ids") is not None:
        values["selected_parent_ids"] = tuple(values["selected_parent_ids"])
    if values.get("selected_parent_names") is not None:
        values["selected_parent_names"] = tuple(values["selected_parent_names"])
    values["notable_genetics_observations"] = tuple(
        values.get("notable_genetics_observations", ())
    )
    values["rejected_alternatives"] = tuple(values.get("rejected_alternatives", ()))
    values["warnings"] = tuple(values.get("warnings", ()))
    candidates = []
    for candidate in values.get("top_candidates", ()):
        candidate_values = dict(candidate)
        candidate_values["parent_ids"] = tuple(candidate_values["parent_ids"])
        candidate_values["parent_names"] = tuple(candidate_values["parent_names"])
        candidates.append(CandidateExplanation(**candidate_values))
    values["top_candidates"] = tuple(candidates)
    return DecisionExplanation(**values)


def _heterozygous_overlap(farm, parent_ids) -> int:
    if farm is None or parent_ids is None:
        return 0
    parent_a, parent_b = (farm.get_rock(rock_id) for rock_id in parent_ids)
    if parent_a is None or parent_b is None:
        return 0
    count = 0
    for gene_name, pair_a in parent_a.genotype.genes.items():
        pair_b = parent_b.genotype.genes[gene_name]
        values_a = {pair_a.allele_a.value, pair_a.allele_b.value}
        values_b = {pair_b.allele_a.value, pair_b.allele_b.value}
        if len(values_a) == 2 and len(values_b) == 2 and values_a != values_b:
            count += 1
    return count


def build_decision_explanation(
    action: AgentAction,
    *,
    farm,
    legal_pair_ids: tuple[tuple[int | str, int | str], ...],
    objective_profile: FarmerObjectiveProfile,
    decision_context: dict[str, Any] | None = None,
    retain_top_candidates: int = 5,
    mutation_chance: float = 0.0,
    close_decision_threshold: float = 0.05,
) -> DecisionExplanation:
    context = decision_context or {}
    candidates = build_candidate_explanations(
        list(context.get("ranked_candidate_pairs", [])),
        farm=farm,
        legal_pair_ids=legal_pair_ids,
        limit=max(5, retain_top_candidates),
    )
    selected_ids = (
        (action.parent_a_id, action.parent_b_id)
        if isinstance(action, BreedPairAction)
        else None
    )
    selected = next(
        (
            candidate
            for candidate in candidates
            if selected_ids is not None
            and tuple(sorted(map(str, candidate.parent_ids)))
            == tuple(sorted(map(str, selected_ids)))
        ),
        None,
    )
    score_gap = (
        candidates[0].score - candidates[1].score if len(candidates) > 1 else None
    )
    observations = []
    warnings = list(context.get("warnings", []))
    overlap = _heterozygous_overlap(farm, selected_ids)
    if overlap:
        observations.append(f"Parents carry complementary heterozygous allele sets in {overlap} genes.")
    if selected and selected.phenotype_diversity is not None and selected.phenotype_diversity >= 0.70:
        observations.append("The predictor estimates high phenotype diversity for this pair.")
    if selected and selected.rare_trait_score is not None and selected.rare_trait_score >= 0.25:
        observations.append("The explicit rarity component materially supports this pair.")
    if selected and selected.mutation_probability is not None and selected.mutation_probability >= 0.25:
        observations.append("The current rules give this pair an elevated chance of at least one mutation.")
    if mutation_chance >= 0.05:
        observations.append("The configured mutation rate materially affects expected outcomes.")
    if score_gap is not None and score_gap <= close_decision_threshold:
        observations.append("The leading candidate scores are nearly tied.")
        warnings.append("Small score differences may not be practically meaningful.")
    if selected and candidates:
        available_values = [
            candidate.predicted_average_child_value
            for candidate in candidates
            if candidate.predicted_average_child_value is not None
        ]
        if (
            selected.predicted_average_child_value is not None
            and available_values
            and selected.predicted_average_child_value < max(available_values) - 1e-9
            and (
                objective_profile.genotype_diversity_weight > objective_profile.immediate_expected_value_weight
                or objective_profile.phenotype_diversity_weight > objective_profile.immediate_expected_value_weight
            )
        ):
            observations.append("The objective trades some immediate expected value for diversity.")
    rejected = tuple(
        {
            "parent_ids": list(candidate.parent_ids),
            "rank": candidate.rank,
            "score": candidate.score,
            "score_difference": (
                selected.score - candidate.score if selected is not None else None
            ),
            "reason": "lower objective-conditioned score",
        }
        for candidate in candidates
        if selected is None or candidate.parent_ids != selected.parent_ids
    )[:4]
    predicted_summary = {
        "expected_survivors": selected.predicted_expected_survivors if selected else None,
        "expected_average_child_value": selected.predicted_average_child_value if selected else None,
        "expected_maximum_child_value": selected.predicted_maximum_child_value if selected else None,
        "mutation_probability": selected.mutation_probability if selected else None,
        "genotype_diversity": selected.genotype_diversity if selected else None,
        "phenotype_diversity": selected.phenotype_diversity if selected else None,
    }
    fallback_reason = None if isinstance(action, BreedPairAction) else getattr(action, "reason", None)
    return DecisionExplanation(
        selected_action=action.to_dict(),
        selected_parent_ids=selected_ids,
        selected_parent_names=selected.parent_names if selected else None,
        selected_candidate_score=(
            selected.score if selected else context.get("selected_score")
        ),
        objective_profile=objective_profile.to_dict(),
        predicted_offspring_summary=predicted_summary,
        score_component_contributions=(selected.objective_contributions if selected else {}),
        selected_pair_rank=selected.rank if selected else None,
        total_legal_candidates=len(legal_pair_ids),
        first_second_score_difference=score_gap,
        confidence_proxy=(
            float(context.get("scores", {}).get("confidence_proxy"))
            if context.get("scores", {}).get("confidence_proxy") is not None
            else None
        ),
        uncertainty_penalty=selected.uncertainty if selected else None,
        notable_genetics_observations=tuple(observations),
        rejected_alternatives=rejected,
        warnings=tuple(dict.fromkeys(warnings)),
        fallback_reason=fallback_reason,
        top_candidates=tuple(candidates[:retain_top_candidates]),
    )
