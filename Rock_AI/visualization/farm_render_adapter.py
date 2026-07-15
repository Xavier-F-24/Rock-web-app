"""Convert authoritative rocks into immutable visual display records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from Rock_AI.evaluation.breeding_agent_metrics import RARE_ALLELES


@dataclass(frozen=True)
class FarmRockView:
    rock_id: int | str
    name: str
    sex: str
    generation: int
    value: float
    status: str
    phenotype_traits: tuple[tuple[str, str], ...]
    genotype_summary: tuple[str, ...]
    image_uri: str | None
    selected_parent: bool = False
    newly_created: bool = False
    mutated: bool = False
    rare_trait: bool = False
    high_value: bool = False


def _rock_name(rock: object) -> str:
    name = getattr(rock, "name", None)
    return getattr(name, "full_name", None) or str(name or f"Rock #{rock.id}")


def _genes(rock: object):
    genotype = getattr(rock, "genotype", None)
    return getattr(genotype, "genes", {}) or {}


def rock_has_rare_trait(rock: object) -> bool:
    return any(
        (gene_name, int(allele.value)) in RARE_ALLELES
        for gene_name, pair in _genes(rock).items()
        for allele in pair.alleles
    )


def safe_render_rock_image(
    rock: object,
    renderer: Callable[..., str] | None = None,
    **kwargs: Any,
) -> str | None:
    """Render one rock without allowing optional image failures to break a viewer."""
    if renderer is None:
        try:
            from Rock_Drawing.rock_drawing_helper import rock_to_image_uri
        except (ImportError, ModuleNotFoundError):
            return None
        renderer = rock_to_image_uri
    try:
        return renderer(rock, **kwargs)
    except Exception:
        return None


def build_farm_rock_views(
    game: object,
    *,
    selected_parent_ids: tuple[int | str, ...] = (),
    new_child_ids: tuple[int | str, ...] = (),
    mutation_rock_ids: tuple[int | str, ...] = (),
    include_images: bool = True,
    image_renderer: Callable[..., str] | None = None,
    high_value_quantile: float = 0.85,
) -> list[FarmRockView]:
    source = getattr(game, "rocks", {})
    rocks = list(source.values() if isinstance(source, dict) else source)
    rocks.sort(key=lambda rock: (int(getattr(rock, "generation", 0)), str(rock.id)))
    values = sorted(float(getattr(rock, "value", 0.0)) for rock in rocks)
    threshold_index = max(0, min(len(values) - 1, int((len(values) - 1) * high_value_quantile)))
    high_value_threshold = values[threshold_index] if values else float("inf")
    selected = {str(rock_id) for rock_id in selected_parent_ids}
    children = {str(rock_id) for rock_id in new_child_ids}
    mutations = {str(rock_id) for rock_id in mutation_rock_ids}
    result = []
    for rock in rocks:
        traits = tuple(
            (str(name), str(getattr(pair, "phenotype", "")))
            for name, pair in sorted(_genes(rock).items())
        )
        genotype = tuple(
            f"{name}: {pair.allele_a.value}/{pair.allele_b.value}"
            for name, pair in sorted(_genes(rock).items())
        )
        status = getattr(getattr(rock, "status", None), "value", str(getattr(rock, "status", "")))
        sex = getattr(getattr(rock, "sex", None), "value", str(getattr(rock, "sex", "")))
        result.append(
            FarmRockView(
                rock_id=rock.id,
                name=_rock_name(rock),
                sex=str(sex),
                generation=int(getattr(rock, "generation", 0)),
                value=float(getattr(rock, "value", 0.0)),
                status=str(status),
                phenotype_traits=traits,
                genotype_summary=genotype,
                image_uri=(
                    safe_render_rock_image(rock, image_renderer, sprite_size=1.4, dpi=180)
                    if include_images else None
                ),
                selected_parent=str(rock.id) in selected,
                newly_created=str(rock.id) in children,
                mutated=str(rock.id) in mutations,
                rare_trait=rock_has_rare_trait(rock),
                high_value=float(getattr(rock, "value", 0.0)) >= high_value_threshold,
            )
        )
    return result
