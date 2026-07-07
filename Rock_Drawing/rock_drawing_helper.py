#-----------------------------------------------------
"""
Rock Drawing Helper

Public compatibility surface for individual rock drawing.
The renderer internals live in smaller modules:
- rock_drawing_phenotype.py
- rock_drawing_geometry.py
- rock_render_context.py
- rock_feature_drawers.py
- rock_draw_machine.py
"""
#-----------------------------------------------------

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt

from Rock_Drawing.rock_drawing_phenotype import *
from Rock_Drawing.rock_drawing_geometry import *
from Rock_Drawing.rock_render_context import *
from Rock_Drawing.rock_feature_drawers import *
from Rock_Drawing.rock_draw_machine import DrawMachine, draw_rock

PAD_FRAC = 0.2

# -----------------------------
# SAVING ROCK AS AN IMAGE FOR ROCK
# -----------------------------

def pad_rock_axis(ax, pad_frac=PAD_FRAC):
    """
    Expand the current axes limits so external traits do not get clipped.

    Good for halos, ion stones, wings, tails, hair, horns, etc.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    dx = x1 - x0
    dy = y1 - y0

    ax.set_xlim(x0 - pad_frac * dx, x1 + pad_frac * dx)
    ax.set_ylim(y0 - pad_frac * dy, y1 + pad_frac * dy)
    ax.set_aspect("equal")

def rock_to_image_uri(rock, sprite_size=2.0, dpi=400, identity_layout=None):
    """
    Render a rock to a transparent PNG and return it as a base64 image URI
    that Plotly can place on the graph.
    """
    fig, ax = plt.subplots(figsize=(sprite_size, sprite_size), dpi=dpi)

    if identity_layout is None:
        draw_rock(rock, ax=ax)
    else:
        draw_rock(rock, ax=ax, identity_layout=identity_layout)

    pad_rock_axis(ax, pad_frac=PAD_FRAC)

    ax.set_title("")
    ax.axis("off")

    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        transparent=True,
        dpi=dpi,
    )
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

def get_rock_render_signature(rock):
    """
    Stable-enough cache key for a rock image.
    """
    gene_bits = []

    for gene_name, gene_pair in sorted(rock.genotype.genes.items()):
        gene_bits.append(
            (
                gene_name,
                gene_pair.allele_a.value,
                gene_pair.allele_b.value,
                gene_pair.phenotype,
            )
        )

    return (
        int(rock.id),
        str(getattr(rock, "name", "")),
        getattr(getattr(rock, "sex", None), "value", str(getattr(rock, "sex", ""))),
        getattr(getattr(rock, "status", None), "value", str(getattr(rock, "status", ""))),
        tuple(gene_bits),
    )

def ensure_rock_image_cache(game):
    """
    Runtime cache for id -> rendered data URI.
    """
    if not hasattr(game, "rock_image_cache") or game.rock_image_cache is None:
        game.rock_image_cache = {}

    return game.rock_image_cache

def render_game_rock_images(game, sprite_size=2.0, dpi=220, force=False, identity_layout=None):
    """
    Render all game rocks to a cache and return {rock_id: image_uri}.
    """
    cache = ensure_rock_image_cache(game)
    output = {}

    for rock_id, rock in game.rocks.items():
        signature = get_rock_render_signature(rock)
        cached = cache.get(int(rock_id))

        if force or cached is None or cached.get("signature") != signature:
            cache[int(rock_id)] = {
                "signature": signature,
                "image_uri": rock_to_image_uri(
                    rock,
                    sprite_size=sprite_size,
                    dpi=dpi,
                    identity_layout=identity_layout,
                ),
            }

        output[int(rock_id)] = cache[int(rock_id)]["image_uri"]

    return output
