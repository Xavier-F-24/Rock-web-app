#-----------------------------------------------------
"""
Split-out module from rock_drawing_helper.py.
"""
#-----------------------------------------------------

from __future__ import annotations

from typing import Any

import Rock_Genetics.rock_genetic_helper as genetics

#-----------------------------------------------------
# EYE COLOR MAP
#-----------------------------------------------------

EYE_COLOR_MAP = {
    "white":  "white",
    "black":  "black",
    "red":    "red",
    "green":  "green",
    "blue":   "royalblue",
    "yellow": "gold",
    "evil":   "crimson",
    "purple": "purple",
    "callus": "tan",
    "n/a":    "white",
}




BODY_COLOR_MAP = {
    "white":     (0.88, 0.86, 0.80),
    "black":     (0.16, 0.15, 0.15),
    "silver":    (0.62, 0.62, 0.58),

    "brown":     (0.48, 0.30, 0.18),

    "red":       (0.74, 0.25, 0.22),
    "yellow":    (0.90, 0.72, 0.25),
    "blue":      (0.24, 0.40, 0.72),

    "orange":    (0.92, 0.46, 0.16),
    "green":     (0.28, 0.62, 0.30),
    "purple":    (0.52, 0.25, 0.65),

    "patchwork": (0.52, 0.48, 0.42),
    "n/a":       (0.45, 0.42, 0.38),
}

BODY_PRIMARY_CLASS = {"white", "black"}
BODY_SECONDARY_CLASS = {"brown"}
BODY_TERTIARY_CLASS = {"red", "yellow", "blue"}
BODY_RECESSIVE_CLASS = {"patchwork"}

HAIR_COLOR_MAP = {
    "white":  (0.92, 0.90, 0.84),
    "black":  (0.05, 0.04, 0.04),
    "silver": (0.62, 0.62, 0.58),

    "brown":  (0.35, 0.19, 0.08),
    "blonde": (0.95, 0.76, 0.28),
    "red":    (0.72, 0.18, 0.10),
    "pink":   (0.95, 0.32, 0.62),
    "blue":   (0.18, 0.34, 0.80),

    "n/a":    (0.05, 0.04, 0.04),
}

HAIR_DOMINANCE_RANK = {
    "white": 0,
    "black": 0,
    "brown": 1,
    "blonde": 2,
    "red": 3,
    "pink": 4,
    "blue": 5,
}

def clean_color_alleles(
    color_alleles
):
    cleaned = []

    for color_name in color_alleles:
        if color_name is None:
            continue

        color_name = str(color_name).lower()

        if color_name != "n/a":
            cleaned.append(color_name)

    return cleaned

def express_body_color_name(
    color_alleles
):
    """
    Body color rule used by the renderer.
    """

    alleles = clean_color_alleles(color_alleles)

    if len(alleles) == 0:
        return "n/a"

    a = alleles[0]
    b = alleles[1] if len(alleles) > 1 else alleles[0]
    pair = {a, b}

    primary_present = [color_name for color_name in [a, b] if color_name in BODY_PRIMARY_CLASS]

    if len(primary_present) == 2:
        if pair == {"white", "black"}:
            return "silver"
        return primary_present[0]

    if len(primary_present) == 1:
        return primary_present[0]

    if "brown" in pair:
        return "brown"

    tertiary_present = [color_name for color_name in [a, b] if color_name in BODY_TERTIARY_CLASS]

    if len(tertiary_present) == 2:
        tertiary_pair = set(tertiary_present)

        if tertiary_pair == {"red", "yellow"}:
            return "orange"
        if tertiary_pair == {"red", "blue"}:
            return "purple"
        if tertiary_pair == {"yellow", "blue"}:
            return "green"

        return tertiary_present[0]

    if len(tertiary_present) == 1:
        return tertiary_present[0]

    if a == "patchwork" and b == "patchwork":
        return "patchwork"

    return "n/a"

def express_hair_color_name(
    hair_color_alleles
):
    """
    Hair color rule used by the renderer.
    """

    alleles = clean_color_alleles(hair_color_alleles)

    if len(alleles) == 0:
        return "black"

    a = alleles[0]
    b = alleles[1] if len(alleles) > 1 else alleles[0]
    pair = {a, b}

    if pair == {"white", "black"}:
        return "silver"

    ranked = sorted(
        [a, b],
        key = lambda color_name: HAIR_DOMINANCE_RANK.get(color_name, 999),
    )

    return ranked[0]

def get_gene_pair(
    rock,
    gene_name: str
):
    genotype = getattr(rock, "genotype", None)

    if genotype is None:
        return None

    return genotype.genes.get(gene_name)

def get_gene_values(
    rock,
    gene_name: str
) -> list[int]:
    gene_pair = get_gene_pair(rock, gene_name)

    if gene_pair is None:
        return []

    return [
        gene_pair.allele_a.value,
        gene_pair.allele_b.value,
    ]

def get_gene_allele_names(
    rock,
    gene_name: str
) -> list[str]:
    values = get_gene_values(rock, gene_name)
    spec = genetics.GENE_SPECS.get(gene_name)

    if spec is None:
        return [str(value) for value in values]

    names = []

    for value in values:
        option = spec.options.get(value)
        names.append(option.name if option is not None else str(value))

    return names

def get_gene_phenotype(
    rock,
    gene_name: str,
    fallback: str = "n/a"
) -> str:
    gene_pair = get_gene_pair(rock, gene_name)

    if gene_pair is None:
        return fallback

    if gene_pair.phenotype is None:
        return fallback

    return gene_pair.phenotype

def get_rock_gender_value(
    rock
) -> int:
    if getattr(rock, "sex", genetics.Sex.FEMALE) == genetics.Sex.MALE:
        return 1

    return 0

def get_visual_phenotype(
    rock
) -> dict[str, Any]:
    """
    Build the renderer-facing phenotype dictionary from genetics dataclasses.
    """

    v: dict[str, Any] = {}

    v["gender"] = "Male" if get_rock_gender_value(rock) == 1 else "Female"

    status = getattr(rock, "status", None)
    v["is_craisen"] = status == genetics.RockStatus.CRAISENED or bool(getattr(rock, "is_craisen", False))

    for gene_name in genetics.GENE_SPECS:
        values = get_gene_values(rock, gene_name)
        allele_names = get_gene_allele_names(rock, gene_name)

        v[f"{gene_name}_values"] = values
        v[f"{gene_name}_alleles"] = allele_names
        v[gene_name] = get_gene_phenotype(rock, gene_name)

    eye_values = v.get("eyes_values", [])
    fuzz_values = v.get("fuzz_values", [])
    hair_values = v.get("hair_values", [])

    v["eyes_count"] = sum(1 for value in eye_values if value == 1)
    v["fuzz_count"] = sum(1 for value in fuzz_values if value == 1)
    v["hair_count"] = sum(1 for value in hair_values if value == 1)

    arm_values = v.get("arms_values", [])
    v["normal_arm_pairs"] = arm_values.count(1)
    v["muscle_arm_pairs"] = arm_values.count(2)
    v["normal_arm_count"] = 2 * v["normal_arm_pairs"]
    v["muscle_arm_count"] = 2 * v["muscle_arm_pairs"]

    v["color"] = express_body_color_name(v.get("color_alleles", []))
    v["hair_color"] = express_hair_color_name(v.get("hair_color_alleles", []))

    if v["gender"] == "Female" and v.get("facial_hair", "n/a") != "n/a":
        v["facial_hair"] = "peach fuzz"

    return v

def get_body_color_from_alleles(
    color_alleles
):
    color_name = express_body_color_name(color_alleles)
    return BODY_COLOR_MAP.get(color_name, BODY_COLOR_MAP["n/a"])

def get_hair_color_from_alleles(hair_color_alleles):
    color_name = express_hair_color_name(hair_color_alleles)
    return HAIR_COLOR_MAP.get(color_name, HAIR_COLOR_MAP["white"])

def get_render_hair_color(ctx):
    return get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

