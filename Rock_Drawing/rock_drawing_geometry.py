#-----------------------------------------------------
"""
Split-out module from rock_drawing_helper.py.
"""
#-----------------------------------------------------

from __future__ import annotations

import math
import random

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Arc
from matplotlib.path import Path

def make_body_points(shape_name, size_name, rng):
    """
    Generate the body outline points.
    """
    size_scale_map = {
        "medium": 1.00,
        "large": 1.60,
        "small": 0.70,
        "giant": 2.30,
        "missized": 1.30,
    }

    s = size_scale_map.get(size_name, 1.0)

    if shape_name == "triangle":
        base = np.array([
            [0.00, 1.05],
            [-1.05, -0.75],
            [1.05, -0.75],
        ]) * s

        points = []

        for i in range(3):
            a = base[i]
            b = base[(i + 1) % 3]

            for j in range(7):
                t = j / 7
                pt = (1 - t) * a + t * b
                #if size_name == "missized":
                #pt += rng.normal(0, 0.035 * s, size=2)
                points.append(pt)

        return np.array(points), s

    n_points = 34
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    if shape_name == "square":
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.42
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.42
    else:
        x = np.cos(theta)
        y = np.sin(theta)

    if shape_name == "circle":
        x_scale, y_scale = 1.00, 1.00
    elif shape_name == "oval":
        x_scale, y_scale = 0.82, 1.18
    elif shape_name == "oblong":
        x_scale, y_scale = 1.35, 0.72
    elif shape_name == "square":
        x_scale, y_scale = 1.00, 1.00
    else:
        x_scale, y_scale = 1.00, 1.00

    if size_name == "missized":
        x_scale *= 1.28
        y_scale *= 0.82

    wobble = 1 #
    if size_name == "missized":
      wobble += rng.normal(0, 0.055, n_points)

    points = np.column_stack([
        s * x_scale * x * wobble,
        s * y_scale * y * wobble
    ])

    return points, s



# -----------------------------
# DRAWING WINGS FOR ROCK
# -----------------------------

def get_wing_anchor_fraction(ctx):
    """
    Choose the vertical attachment height for wings.
    Sketch style: wings attach fairly high on the body.
    """
    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        return 0.60
    elif shape == "oblong":
        return 0.66
    elif shape == "oval":
        return 0.66
    elif shape == "square":
        return 0.67
    else:
        return 0.66

def polygon_x_span_at_y(points, y):
    """
    Finds the left/right x-intersections of a horizontal line through a polygon.

    Returns (x_left, x_right).
    """
    xs = []
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]

        # Skip horizontal edges to avoid duplicate weirdness.
        if abs(y2 - y1) < 1e-9:
            continue

        y_low = min(y1, y2)
        y_high = max(y1, y2)

        if y_low <= y <= y_high:
            t = (y - y1) / (y2 - y1)

            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                xs.append(x)

    if len(xs) < 2:
        # Fallback to bounding box.
        return float(np.min(points[:, 0])), float(np.max(points[:, 0]))

    xs = sorted(xs)
    return xs[0], xs[-1]

def body_span_at_fraction(ctx, y_frac):
    """
    Returns x-left, x-right, and y for a horizontal body slice.
    """
    y = ctx.ymin + y_frac * ctx.height
    x_left, x_right = ctx.body_x_span_at_ny(y_frac)

    return x_left, x_right, y

def clamp_inside_span(x, x_left, x_right, margin):
    return max(x_left + margin, min(x, x_right - margin))

def get_fuzz_color(body_color):
    """
    Adaptive inner fuzz color:
    - lighter on dark rocks
    - darker on light rocks

    This sits on top of a black under-stroke for visibility.
    """
    lum = color_luminance(body_color)

    if lum < 0.45:
        return mix_colors(body_color, (1, 1, 1), t=0.55)
    else:
        return mix_colors(body_color, (0, 0, 0), t=0.45)

def polygon_perimeter(points):
    """
    Perimeter of a closed polygon.
    """
    pts = np.asarray(points)
    shifted = np.roll(pts, -1, axis=0)
    return np.sum(np.sqrt(np.sum((shifted - pts) ** 2, axis=1)))

def sample_polygon_boundary(points, n_samples, offset_frac=0.0):
    """
    Evenly sample points along a closed polygon boundary.
    Returns an array of shape (n_samples, 2).
    """
    pts = np.asarray(points)
    shifted = np.roll(pts, -1, axis=0)

    seg_vecs = shifted - pts
    seg_lens = np.sqrt(np.sum(seg_vecs ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]

    if total <= 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)

    samples = []
    start_dist = offset_frac * total

    for k in range(n_samples):
        d = (start_dist + k * total / n_samples) % total

        seg_idx = np.searchsorted(cum, d, side="right") - 1
        seg_idx = min(seg_idx, len(seg_lens) - 1)

        seg_start = cum[seg_idx]
        seg_len = seg_lens[seg_idx]

        if seg_len <= 1e-12:
            samples.append(pts[seg_idx].copy())
            continue

        t = (d - seg_start) / seg_len
        p = pts[seg_idx] + t * seg_vecs[seg_idx]
        samples.append(p)

    return np.array(samples)

def transform_template_points(points, base, side=1, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    """
    Transform a canned local template:
    - scale x/y
    - mirror in x for left/right
    - translate to base
    """
    pts = np.array(points, dtype=float).copy()

    pts[:, 0] *= sx
    pts[:, 1] *= sy

    if side == -1:
        pts[:, 0] *= -1

    pts[:, 0] += dx + base[0]
    pts[:, 1] += dy + base[1]

    return pts

def make_round_ear_template():
    """
    Human-esque ear.
    Rounded with a slightly narrower lower attachment and fuller upper body.
    Local coordinates.
    """
    return np.array([
        [ 0.00, -0.10],   # lower attach
        [ 0.10, -0.02],
        [ 0.18,  0.10],
        [ 0.20,  0.26],
        [ 0.14,  0.40],
        [ 0.02,  0.48],   # top
        [-0.08,  0.42],
        [-0.14,  0.26],
        [-0.12,  0.10],
        [-0.06, -0.02],
    ])

def make_ogre_ear_template():
    """
    Shrek-like ogre ear.
    Broad, flared outward, with a trumpet/tube-like silhouette.
    """
    return np.array([
        [ 0.00, -0.06],   # attach point
        [ 0.10,  0.00],
        [ 0.28,  0.08],
        [ 0.46,  0.08],
        [ 0.60,  -0.18],   # outward flare
        [ 0.73,  0.08],
        [ 0.78,  0.35],
        [ 0.60,  0.58],
        [ 0.48,  0.38],
        [ 0.28,  0.34],
        [ 0.12,  0.24],
        [ 0.02,  0.12],
        [-0.02,  0.02],
    ])

def make_goblin_ear_template():
    """
    Elfish goblin ear.
    Taller, sharper, elegant point, slightly swept back.
    """
    return np.array([
        [ 0.00, -0.08],   # attach point
        [ 0.10,  0.02],
        [ 0.18,  0.18],
        [ 0.20,  0.38],
        [ 0.12,  0.62],
        [ 0.00,  0.86],   # main point
        [-0.10,  0.62],
        [-0.12,  0.34],
        [-0.08,  0.12],
    ])

def color_luminance(rgb):
    """
    Approximate perceived luminance for an RGB tuple in [0,1].
    """
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def mix_colors(c1, c2, t=0.5):
    """
    Linear blend between two RGB colors.
    t=0 -> c1
    t=1 -> c2
    """
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1 - t) * c1 + t * c2, 0, 1))

def adjust_color_brightness(color, factor=1.2):
    """
    Lighten/darken a matplotlib color.

    factor > 1 lightens
    factor < 1 darkens
    """
    try:
        rgb = np.array(mcolors.to_rgb(color))
    except Exception:
        rgb = np.array([0.1, 0.1, 0.1])

    if factor >= 1:
        rgb = rgb + (1 - rgb) * (factor - 1)
    else:
        rgb = rgb * factor

    return tuple(np.clip(rgb, 0, 1))

def rock_texture_is_curly(ctx):
    """
    True if this rock expresses curly hair texture.

    Supports phenotype strings and direct gene fallback.
    """
    texture = str(ctx.v.get("hair_texture", "n/a")).lower()

    if "curly" in texture:
        return True

    values = ctx.v.get("hair_texture_values", [])
    return any(value == 1 for value in values)

def deterministic_rng_for_rock(rock, salt="curl"):
    """
    Deterministic random generator so curls do not jump around every redraw.
    """
    genotype = getattr(rock, "genotype", None)

    if genotype is None:
        gene_signature = ""
    else:
        gene_signature = tuple(
            sorted(
                (
                    gene_name,
                    gene_pair.allele_a.value,
                    gene_pair.allele_b.value,
                )
                for gene_name, gene_pair in genotype.genes.items()
            )
        )

    seed_text = f"{getattr(rock, 'id', 0)}_{salt}_{gene_signature}"
    seed = abs(hash(seed_text)) % (2**32)
    return random.Random(seed)

def draw_curl_arc(
    ax,
    x,
    y,
    w,
    h,
    color,
    angle=0,
    theta1=0,
    theta2=180,
    linewidth=1.25,
    alpha=0.9,
    zorder=50
):
    """
    Draw one curl arc.
    """
    arc = Arc(
        (x, y),
        width=w,
        height=h,
        angle=angle,
        theta1=theta1,
        theta2=theta2,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

    ax.add_patch(arc)

    return arc

def make_diamond(cx, cy, w, h):
    """
    Return diamond polygon points centered at (cx, cy).
    """
    return [
        [cx, cy + h / 2],
        [cx + w / 2, cy],
        [cx, cy - h / 2],
        [cx - w / 2, cy],
    ]

