"""Named candidate outcomes derived from policy outputs, never hidden reasoning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateExplanation:
    parent_ids: tuple[int | str, int | str]
    parent_names: tuple[str, str]
    score: float
    rank: int
    predicted_expected_survivors: float | None = None
    predicted_average_child_value: float | None = None
    predicted_maximum_child_value: float | None = None
    mutation_probability: float | None = None
    genotype_diversity: float | None = None
    phenotype_diversity: float | None = None
    rare_trait_score: float | None = None
    uncertainty: float | None = None
    objective_contributions: dict[str, float] = field(default_factory=dict)
    legality_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rock_name(farm, rock_id) -> str:
    rock = farm.get_rock(rock_id) if farm is not None else None
    if rock is None:
        return f"Rock #{rock_id}"
    name = getattr(rock, "name", None)
    return getattr(name, "full_name", None) or str(name or f"Rock #{rock_id}")


def _scalar(result: dict[str, Any] | None, *names: str) -> float | None:
    if not result:
        return None
    values = result.get("scalar_predictions", {})
    for name in names:
        if name in values:
            return float(values[name])
    return None


def build_candidate_explanations(
    ranked_candidates: list[dict[str, Any]],
    *,
    farm,
    legal_pair_ids: tuple[tuple[int | str, int | str], ...],
    limit: int = 5,
) -> list[CandidateExplanation]:
    legal = {tuple(sorted(map(str, pair))) for pair in legal_pair_ids}
    explanations = []
    for rank, row in enumerate(ranked_candidates[:limit], start=1):
        parent_ids = tuple(row.get("parent_ids", ()))
        if len(parent_ids) != 2:
            continue
        predicted = row.get("predicted_breeding_outcomes") or {}
        binary = predicted.get("binary_probability_predictions", {})
        components = {
            str(name): float(value)
            for name, value in (row.get("score_components") or {}).items()
            if isinstance(value, (int, float))
        }
        explanations.append(
            CandidateExplanation(
                parent_ids=(parent_ids[0], parent_ids[1]),
                parent_names=(_rock_name(farm, parent_ids[0]), _rock_name(farm, parent_ids[1])),
                score=float(row.get("score", 0.0)),
                rank=rank,
                predicted_expected_survivors=_scalar(predicted, "expected_survivor_count"),
                predicted_average_child_value=_scalar(
                    predicted,
                    "expected_average_surviving_child_value",
                    "expected_average_child_value",
                ),
                predicted_maximum_child_value=_scalar(
                    predicted,
                    "expected_maximum_surviving_child_value",
                    "expected_maximum_child_value",
                ),
                mutation_probability=(
                    float(binary["probability_at_least_one_mutation"])
                    if "probability_at_least_one_mutation" in binary
                    else None
                ),
                genotype_diversity=_scalar(predicted, "genotype_diversity_estimate"),
                phenotype_diversity=_scalar(predicted, "phenotype_diversity_estimate"),
                rare_trait_score=components.get("rare_trait"),
                uncertainty=components.get("uncertainty_penalty"),
                objective_contributions=components,
                legality_confirmed=tuple(sorted(map(str, parent_ids))) in legal,
            )
        )
    return explanations
