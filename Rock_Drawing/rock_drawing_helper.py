#-----------------------------------------------------
"""
Rock Drawing Helper 

This file answers:

- How does a rock look because of its genotype?
- What does the image of the rock get saved as?

- How do I display family trees of rocks?
- How do I display individual rocks, as needed?

"""
#-----------------------------------------------------

#-----------------------------------------------------
# IMPORT ZONE
#-----------------------------------------------------

import random, math
import numpy as np

from dataclasses import dataclass, field
from typing import List

#-----------------------------------------------------
# SPECIAL IMPORT ZONE
#-----------------------------------------------------

import Rock_Genetics.rock_genetic_helper as genetics

#-----------------------------------------------------
# START MAJOR REFACTOR: FEATURE PRESENCE 
#-----------------------------------------------------

@dataclass(frozen=True)
class FeaturePresence:
    eyes: bool = False
    brows: bool = False
    nose: bool = False
    mouth: bool = False
    facial_hair: bool = False
    hair: bool = False
    horns: bool = False
    ears: bool = False
    crown: bool = False
    halo: bool = False
    wings: bool = False
    tail: bool = False

#-----------------------------------------------------
# FEATURE SLOTS
#-----------------------------------------------------

@dataclass(frozen=True)
class FeatureSlot:
    name: str
    nx: float
    ny: float
    scale: float = 1.0

#-----------------------------------------------------
# FACE LAYOUT: WHAT SLOTS ARE ON THE ROCK
#-----------------------------------------------------

@dataclass(frozen=True)
class FaceLayout:
    slots: dict[str, FeatureSlot] = field(default_factory=dict)

    def has(self, feature_name: str) -> bool:
        return feature_name in self.slots

    def get(self, feature_name: str) -> FeatureSlot | None:
        return self.slots.get(feature_name)

#-----------------------------------------------------
# ROCK RENDERER CONTEXT: NEW AND IMPORVED
#-----------------------------------------------------

@dataclass
class RockRenderContext:

    ax: object

    v: dict
    rng: object
    py_rng: object

    body: object
    body_points: np.ndarray
    body_color: object

    s: float

    rock: genetics.Rock

    def __post_init__(
        self
    ):
        self.xmin = float(np.min(self.body_points[:, 0]))
        self.xmax = float(np.max(self.body_points[:, 0]))
        self.ymin = float(np.min(self.body_points[:, 1]))
        self.ymax = float(np.max(self.body_points[:, 1]))

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

        self.cx = 0.5 * (self.xmin + self.xmax)
        self.cy = 0.5 * (self.ymin + self.ymax)

        self.unit = min(self.width, self.height)

        self.presence = self._build_feature_presence()
        self.face_layout = self._build_face_layout()

    def xy(
        self, 
        nx: float, 
        ny: float,
    ) -> tuple[float, float]:
        
        """
        Convert normalized bounding-box body coordinates to actual plot coordinates.

        nx = 0 means body bounding-box left
        nx = 1 means body bounding-box right
        ny = 0 means body bottom
        ny = 1 means body top
        """

        x = self.xmin + nx * self.width
        y = self.ymin + ny * self.height
        return x, y

    def body_xy(
        self, 
        nx: float, 
        ny: float
    ) -> tuple[float, float]:
        
        """
        Convert normalized coordinates to actual plot coordinates,
        but use the real body width at that y-level.

        This is better than xy() for triangles, oblongs, and irregular bodies.
        """

        left_x, right_x = self.body_x_span_at_ny(ny)
        y = self.ymin + ny * self.height
        x = left_x + nx * (right_x - left_x)
        return x, y

    def body_x_span_at_ny(
        self, 
        ny: float
    ) -> tuple[float, float]:
        
        """
        Find the left and right body boundary at a normalized y-position.

        This lets face features fit inside the actual body shape instead of
        only using the bounding box.
        """

        y = self.ymin + ny * self.height
        points = self.body_points
        intersections = []

        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]

            crosses = (y1 <= y <= y2) or (y2 <= y <= y1)

            if not crosses:
                continue

            if y1 == y2:
                continue

            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            intersections.append(float(x))

        if len(intersections) < 2:
            return self.xmin, self.xmax

        intersections.sort()
        return intersections[0], intersections[-1]

    def phen(
        self, 
        gene_name: str, 
        default: str = "n/a"
    ) -> str:
        
        """
        Get the already-computed phenotype for a gene.
        """

        if self.rock is not None:
            genotype = self.rock.genotype.genes[gene_name]

            phenotype = genotype.phenotype

            if phenotype is not None:
                return phenotype
        
        else:
            return (default)

    def is_present(
        self, 
        gene_name: str
    ) -> bool:

        phenotype = self.phen(gene_name)

        absent_values = {
            #None,
            #"",
            "n/a",
            #"none",
            #"inactive",
            #"off",
            #False,
            #0,
        }

        return phenotype not in absent_values

    def feature_xy(
        self, 
        feature_name: str
    ) -> tuple[float, float]:

        """
        Return the planned x/y coordinate for a feature.
        """

        slot = self.face_layout.get(feature_name)

        if slot is None:
            return self.body_xy(0.5, 0.5)

        return self.body_xy(slot.nx, slot.ny)

    def feature_scale(
        self, 
        feature_name: str, 
        default: float = 1.0
    ) -> float:
        
        slot = self.face_layout.get(feature_name)

        if slot is None:
            return default

        return slot.scale

    def _build_feature_presence(
        self
    ) -> FeaturePresence:
        
        return FeaturePresence(
            eyes = self.is_present("eyes"),
            brows = self.is_present("brows"),
            nose = self.is_present("noses"),
            mouth = self.is_present("mouths"),
            facial_hair = self.is_present("facial_hair"),
            hair = self.is_present("hair"),
            horns = self.is_present("horns"),
            ears = self.is_present("ears"),
            crown = self.is_present("crowns"),
            halo = self.is_present("halos"),
            wings = self.is_present("wings"),
            tail = self.is_present("tails"),
        )

    def _build_face_layout(
        self
    ) -> FaceLayout:
        
        """
        Build dynamic facial feature positions.

        If only eyes are present, they sit more centrally.
        If many features are present, they spread vertically.
        """

        slots: dict[str, FeatureSlot] = {}

        face_top = 0.72
        face_bottom = 0.28

        # Reserve upper head space if upper features exist.
        if self.presence.hair:
            face_top -= 0.07

        if self.presence.crown:
            face_top -= 0.03

        if self.presence.horns:
            face_top -= 0.02

        # Give facial hair some lower room.
        if self.presence.facial_hair:
            face_bottom = 0.22

        ordered_face_features = []

        if self.presence.brows:
            ordered_face_features.append("brows")

        if self.presence.eyes:
            ordered_face_features.append("eyes")

        if self.presence.nose:
            ordered_face_features.append("nose")

        if self.presence.mouth:
            ordered_face_features.append("mouth")

        if self.presence.facial_hair:
            ordered_face_features.append("facial_hair")

        if len(ordered_face_features) == 0:
            return FaceLayout(slots = slots)

        if ordered_face_features == ["eyes"]:

            slots["eyes"] = FeatureSlot(
                name = "eyes",
                nx = 0.5,
                ny = 0.53,
                scale = 1.15,
            )

            return FaceLayout(slots=slots)

        if len(ordered_face_features) == 1:

            feature = ordered_face_features[0]

            slots[feature] = FeatureSlot(
                name = feature,
                nx = 0.5,
                ny = 0.50,
                scale = 1.05,
            )

            return FaceLayout(slots = slots)

        y_positions = np.linspace(
            face_top,
            face_bottom,
            len(ordered_face_features),
        )

        for feature, ny in zip(ordered_face_features, y_positions):

            scale = 1.0

            if feature == "eyes":
                scale = 1.0

            if feature == "brows":
                scale = 0.95

            if feature == "facial_hair":
                scale = 1.05

            slots[feature] = FeatureSlot(
                name = feature,
                nx = 0.5,
                ny = float(ny),
                scale = scale,
            )

        return FaceLayout(slots = slots)

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

def get_hair_color_from_alleles(hair_color_alleles):
    color_name = express_hair_color_name(hair_color_alleles)
    return HAIR_COLOR_MAP.get(color_name, HAIR_COLOR_MAP["white"])

def get_render_hair_color(ctx):
    return get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

# -----------------------------
# ROCK RENDERER
# -----------------------------

@dataclass
class RockRenderContext:
    ax: object
    rock: object
    v: dict
    rng: object
    py_rng: object
    body: object
    body_points: np.ndarray
    s: float
    body_color: object

    def __post_init__(self):
        self.xmin = float(np.min(self.body_points[:, 0]))
        self.xmax = float(np.max(self.body_points[:, 0]))
        self.ymin = float(np.min(self.body_points[:, 1]))
        self.ymax = float(np.max(self.body_points[:, 1]))

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

        self.cx = 0.5 * (self.xmin + self.xmax)
        self.cy = 0.5 * (self.ymin + self.ymax)

        self.unit = min(self.width, self.height)

    def xy(self, nx, ny):
        """
        Convert normalized body coordinates to actual plot coordinates.

        nx = 0 means body left edge
        nx = 1 means body right edge
        ny = 0 means body bottom
        ny = 1 means body top
        """
        x = self.xmin + nx * self.width
        y = self.ymin + ny * self.height
        return x, y
    
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

def get_body_color_from_alleles(color_alleles):
    color_name = express_body_color_name(color_alleles)
    return BODY_COLOR_MAP.get(color_name, BODY_COLOR_MAP["n/a"])

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
    x_left, x_right = polygon_x_span_at_y(ctx.body_points, y)

    return x_left, x_right, y

def clamp_inside_span(x, x_left, x_right, margin):
    return max(x_left + margin, min(x, x_right - margin))

def get_wing_layout(ctx):
    """
    Compute left/right wing anchors and scale information.
    Sketch style: rounded wings, slightly taller and moderately wide.
    """
    y_frac = get_wing_anchor_fraction(ctx)
    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    local_span = x_right - x_left

    wing_w = max(0.52 * ctx.unit, 0.62 * local_span)
    wing_h = max(0.62 * ctx.unit, 0.78 * ctx.height)

    return {
        "left_anchor": (x_left, y),
        "right_anchor": (x_right, y),
        "span": local_span,
        "wing_w": wing_w,
        "wing_h": wing_h,
        "y_frac": y_frac
    }

def draw_single_wing(ctx, anchor, side=1, wing_w=1.0, wing_h=1.0):
    """
    Draw one rounded cartoon wing attached to the body edge.

    side:
    -1 = left wing
    +1 = right wing

    Style target:
    - rounded top arch
    - drooping outer wing
    - 3 little feather/finger tips
    - black outline, light fill
    """

    ax = ctx.ax
    x0, y0 = anchor

    sgn = side

    # Anchor points on the body edge
    root_top = (x0, y0 + 0.06 * wing_h)
    root_bot = (x0, y0 - 0.08 * wing_h)

    # Main structure points
    top_peak = (x0 + sgn * 0.30 * wing_w, y0 + 0.72 * wing_h)
    outer_top = (x0 + sgn * 0.82 * wing_w, y0 + 0.68 * wing_h)
    outer_mid = (x0 + sgn * 1.00 * wing_w, y0 + 0.10 * wing_h)
    lower_outer = (x0 + sgn * 0.88 * wing_w, y0 - 0.20 * wing_h)

    # Three feather/finger tips, like the sketch
    tip1 = (x0 + sgn * 0.92 * wing_w, y0 - 0.40 * wing_h)
    valley1 = (x0 + sgn * 0.76 * wing_w, y0 - 0.30 * wing_h)

    tip2 = (x0 + sgn * 0.72 * wing_w, y0 - 0.52 * wing_h)
    valley2 = (x0 + sgn * 0.58 * wing_w, y0 - 0.36 * wing_h)

    tip3 = (x0 + sgn * 0.50 * wing_w, y0 - 0.46 * wing_h)
    inner_return = (x0 + sgn * 0.28 * wing_w, y0 - 0.18 * wing_h)

    # Build a rounded wing outline using bezier curves + line segments
    verts = [
        root_top,  # start
        (x0 + sgn * 0.06 * wing_w, y0 + 0.24 * wing_h),   # control
        (x0 + sgn * 0.16 * wing_w, y0 + 0.48 * wing_h),   # control
        top_peak,                                          # top rise

        (x0 + sgn * 0.45 * wing_w, y0 + 0.86 * wing_h),   # control
        (x0 + sgn * 0.66 * wing_w, y0 + 0.80 * wing_h),   # control
        outer_top,                                         # top arch

        (x0 + sgn * 0.96 * wing_w, y0 + 0.46 * wing_h),   # control
        (x0 + sgn * 1.04 * wing_w, y0 + 0.24 * wing_h),   # control
        outer_mid,                                         # descend

        lower_outer,   # line down
        tip1,
        valley1,
        tip2,
        valley2,
        tip3,
        inner_return,

        (x0 + sgn * 0.10 * wing_w, y0 - 0.12 * wing_h),   # control back in
        root_bot,                                          # return near body
        root_top                                           # close
    ]

    codes = [
        Path.MOVETO,

        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,

        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,

        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY
    ]

    wing_path = Path(verts, codes)

    wing_patch = PathPatch(
        wing_path,
        facecolor=(0.96, 0.96, 0.98, 0.95),
        edgecolor="black",
        linewidth=2.0,
        zorder=0,
        joinstyle="round",
        capstyle="round"
    )
    ax.add_patch(wing_patch)

    # A couple of inner feather support lines
    support_lines = [
        ((x0 + sgn * 0.08 * wing_w, y0 + 0.02 * wing_h), (x0 + sgn * 0.42 * wing_w, y0 + 0.56 * wing_h)),
        ((x0 + sgn * 0.12 * wing_w, y0 - 0.02 * wing_h), (x0 + sgn * 0.68 * wing_w, y0 + 0.10 * wing_h)),
        ((x0 + sgn * 0.16 * wing_w, y0 - 0.04 * wing_h), (x0 + sgn * 0.56 * wing_w, y0 - 0.26 * wing_h)),
    ]

    for p0, p1 in support_lines:
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="black",
            linewidth=1.0,
            alpha=0.45,
            zorder=1
        )

    return wing_patch

def draw_wings(ctx):
    """
    Draw wings using ctx.

    Current rule:
    if v["wings"] != "n/a", draw one left wing and one right wing.
    """
    wing_trait = ctx.v.get("wings", "n/a")

    if wing_trait == "n/a":
        return None

    layout = get_wing_layout(ctx)

    left_anchor = layout["left_anchor"]
    right_anchor = layout["right_anchor"]
    wing_w = layout["wing_w"]
    wing_h = layout["wing_h"]

    left_wing = draw_single_wing(
        ctx,
        left_anchor,
        side=-1,
        wing_w=wing_w,
        wing_h=wing_h
    )

    right_wing = draw_single_wing(
        ctx,
        right_anchor,
        side=1,
        wing_w=wing_w,
        wing_h=wing_h
    )

    return {
        "left": left_wing,
        "right": right_wing,
        "layout": layout
    }

# -----------------------------
# DRAWING FUZZ FOR ROCK
# -----------------------------

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

def draw_fuzz(ctx):
    """
    Draw fuzz around the rock boundary.

    Expression rule:
    - 0 active fuzz alleles -> no fuzz
    - 1 active fuzz allele  -> small fuzz
    - 2 active fuzz alleles -> urchin-like spines

    Drawn behind the body.
    """
    fuzz_count = ctx.v.get("fuzz_count", 0)

    if fuzz_count <= 0:
        return None

    perimeter = polygon_perimeter(ctx.body_points)

    # Use a stable but shape-aware number of spikes.
    if fuzz_count == 1:
        # Small fuzz
        n_spikes = max(18, int(perimeter / (0.22 * ctx.unit)))
        n_spikes = min(n_spikes, 40)

        len_min = 0.05 * ctx.unit
        len_max = 0.12 * ctx.unit

        inner_lw = 1.0
        outer_lw = 1.8

        # Slight random wiggle
        angle_jitter = 0.28
        bent = False

    else:
        # Urchin mode
        n_spikes = max(10, int(perimeter / (0.34 * ctx.unit)))
        n_spikes = min(n_spikes, 24)

        len_min = 0.16 * ctx.unit
        len_max = 0.34 * ctx.unit

        inner_lw = 2.0
        outer_lw = 3.0

        angle_jitter = 0.18
        bent = True

    # Sample the outline evenly, with a deterministic offset.
    offset_frac = ((ctx.rock.id * 0.137) % 1.0)
    boundary_pts = sample_polygon_boundary(ctx.body_points, n_spikes, offset_frac=offset_frac)

    inner_color = get_fuzz_color(ctx.body_color)
    lines_drawn = []

    for p in boundary_pts:
        x0, y0 = p

        # Outward direction = from center to boundary point.
        dx = x0 - ctx.cx
        dy = y0 - ctx.cy
        norm = math.sqrt(dx * dx + dy * dy) + 1e-12
        ux = dx / norm
        uy = dy / norm

        # Add a small angular jitter.
        theta = math.atan2(uy, ux) + ctx.py_rng.uniform(-angle_jitter, angle_jitter)
        ux_j = math.cos(theta)
        uy_j = math.sin(theta)

        spike_len = ctx.py_rng.uniform(len_min, len_max)

        if not bent:
            # Small fuzz = single straight little hair
            x1 = x0 + ux_j * spike_len
            y1 = y0 + uy_j * spike_len

            # Black under-stroke
            line_bg, = ctx.ax.plot(
                [x0, x1],
                [y0, y1],
                color="black",
                linewidth=outer_lw,
                alpha=0.95,
                zorder=0.30,
                solid_capstyle="round"
            )

            # Adaptive inner stroke
            line_fg, = ctx.ax.plot(
                [x0, x1],
                [y0, y1],
                color=inner_color,
                linewidth=inner_lw,
                alpha=0.95,
                zorder=0.35,
                solid_capstyle="round"
            )

            lines_drawn.extend([line_bg, line_fg])

        else:
            # Urchin fuzz = stronger, slightly kinked spines
            mid_len = 0.58 * spike_len

            mx = x0 + ux_j * mid_len
            my = y0 + uy_j * mid_len

            # Small second bend
            theta2 = theta + ctx.py_rng.uniform(-0.20, 0.20)
            ux2 = math.cos(theta2)
            uy2 = math.sin(theta2)

            x2 = mx + ux2 * (spike_len - mid_len)
            y2 = my + uy2 * (spike_len - mid_len)

            line_bg, = ctx.ax.plot(
                [x0, mx, x2],
                [y0, my, y2],
                color="black",
                linewidth=outer_lw,
                alpha=0.98,
                zorder=0.30,
                solid_capstyle="round"
            )

            line_fg, = ctx.ax.plot(
                [x0, mx, x2],
                [y0, my, y2],
                color=inner_color,
                linewidth=inner_lw,
                alpha=0.98,
                zorder=0.35,
                solid_capstyle="round"
            )

            lines_drawn.extend([line_bg, line_fg])

    return lines_drawn

# -----------------------------
# DRAWING HALO FOR ROCK
# -----------------------------

def get_halo_layout(ctx):
    """
    Compute halo placement above the rock.

    Goal:
    - centered above the head
    - about 40% of the rock height above the top
    - scaled to body width/height
    """
    halo_type = ctx.v.get("halos", "n/a")

    if halo_type == "n/a":
        return None

    cx = ctx.cx
    top_y = ctx.ymax

    # Place halo center about 40% of rock height above the top.
    cy = top_y + 0.40 * ctx.height

    # Halo size relative to rock.
    halo_w = 0.62 * ctx.width
    halo_h = 0.14 * ctx.height

    return {
        "type": halo_type,
        "center": (cx, cy),
        "width": halo_w,
        "height": halo_h,
        "top_y": top_y
    }

def draw_halo(ctx):
    """
    Draw a single golden halo with black outline.

    Style:
    - black outer outline for clarity
    - gold inner outline
    - simple ellipse
    """
    halo_type = ctx.v.get("halos", "n/a")

    if halo_type == "n/a":
        return None

    layout = get_halo_layout(ctx)

    cx, cy = layout["center"]
    hw = layout["width"]
    hh = layout["height"]

    # Outer black outline
    halo_black = Ellipse(
        (cx, cy),
        width=hw,
        height=hh,
        facecolor="none",
        edgecolor="black",
        linewidth=3.2,
        zorder=12
    )
    ctx.ax.add_patch(halo_black)

    # Inner gold outline
    halo_gold = Ellipse(
        (cx, cy),
        width=0.94 * hw,
        height=0.82 * hh,
        facecolor="none",
        edgecolor="gold",
        linewidth=2.2,
        zorder=13
    )
    ctx.ax.add_patch(halo_gold)

    return layout

# -----------------------------
# DRAWING ION STONE FOR ROCK
# -----------------------------

def get_ion_stone_layout(ctx):
    """
    Compute an orbit path above the rock head for an ion stone.
    """
    stone_trait = ctx.v.get("stones", "n/a")

    if stone_trait == "n/a":
        return None

    # Orbit sits above the top half of the rock.
    orbit_cx = ctx.cx
    orbit_cy = ctx.ymax + 0.20 * ctx.height

    orbit_w = 0.95 * ctx.width
    orbit_h = 0.42 * ctx.height

    # Arc span over the head.
    theta1 = 20
    theta2 = 160

    # Choose a deterministic stone position along the arc,
    # biased a bit to the upper-left like your sketch.
    stone_angle_deg = ctx.py_rng.uniform(130, 155)
    t = math.radians(stone_angle_deg)

    sx = orbit_cx + 0.5 * orbit_w * math.cos(t)
    sy = orbit_cy + 0.5 * orbit_h * math.sin(t)

    # Stone size varies a little, but not wildly.
    stone_size = ctx.py_rng.uniform(0.10, 0.15) * ctx.unit

    return {
        "orbit_center": (orbit_cx, orbit_cy),
        "orbit_w": orbit_w,
        "orbit_h": orbit_h,
        "theta1": theta1,
        "theta2": theta2,
        "stone_center": (sx, sy),
        "stone_size": stone_size,
    }

def get_ion_stone_color(ctx):
    """
    Pick a pleasant little gem color.
    Deterministic per rock through ctx.py_rng.
    """
    palette = [
        (0.74, 0.88, 1.00),  # pale blue
        (0.90, 0.78, 1.00),  # lavender
        (1.00, 0.82, 0.84),  # rose
        (0.82, 1.00, 0.86),  # mint
        (1.00, 0.90, 0.66),  # amber
        (0.92, 0.92, 1.00),  # pearl
    ]

    idx = ctx.py_rng.randint(0, len(palette) - 1)
    return palette[idx]

def make_ion_stone_polygon(center, size, rng, n_sides=None):
    """
    Make a small irregular gemstone polygon.
    """
    cx, cy = center

    if n_sides is None:
        n_sides = rng.randint(4, 6)

    theta0 = rng.uniform(0, 2 * math.pi)
    angles = np.linspace(0, 2 * math.pi, n_sides, endpoint=False) + theta0

    points = []
    for a in angles:
        r = size * rng.uniform(0.80, 1.18)
        points.append([
            cx + r * math.cos(a),
            cy + r * math.sin(a)
        ])

    return points

def draw_stones(ctx):
    """
    Draw an ion stone trait:
    - one small floating stone
    - one orbit arc over the head
    - slight variation in shape and size
    """
    stone_trait = ctx.v.get("stones", "n/a")

    if stone_trait == "n/a":
        return None

    layout = get_ion_stone_layout(ctx)

    orbit_cx, orbit_cy = layout["orbit_center"]
    orbit_w = layout["orbit_w"]
    orbit_h = layout["orbit_h"]
    theta1 = layout["theta1"]
    theta2 = layout["theta2"]
    stone_center = layout["stone_center"]
    stone_size = layout["stone_size"]

    # Orbit arc
    orbit_arc = Arc(
        (orbit_cx, orbit_cy),
        width=orbit_w,
        height=orbit_h,
        theta1=theta1,
        theta2=theta2,
        color="black",
        linewidth=1.8,
        zorder=14
    )
    ctx.ax.add_patch(orbit_arc)

    # Ion stone shape
    gem_points = make_ion_stone_polygon(
        stone_center,
        stone_size,
        ctx.py_rng
    )

    gem_color = get_ion_stone_color(ctx)

    gem = Polygon(
        gem_points,
        closed=True,
        facecolor=gem_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=15,
        joinstyle="round"
    )
    ctx.ax.add_patch(gem)

    # Tiny highlight for gem shine
    gx, gy = stone_center
    highlight = Circle(
        (gx - 0.20 * stone_size, gy + 0.20 * stone_size),
        radius=0.20 * stone_size,
        facecolor="white",
        edgecolor="none",
        alpha=0.45,
        zorder=16
    )
    ctx.ax.add_patch(highlight)

    return {
        "layout": layout,
        "orbit_arc": orbit_arc,
        "gem": gem
    }

# -----------------------------
# DRAWING TAIL FOR ROCK
# -----------------------------

def get_tail_layout(ctx):
    """
    Compute a tail anchor on the lower body edge.

    Tail comes from the bottom / lower-side region,
    with slight deterministic side variation.
    """
    tail_trait = ctx.v.get("tails", "n/a")

    if tail_trait == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Use a low body slice.
    if shape == "triangle":
        y_frac = 0.14
    elif shape == "oblong":
        y_frac = 0.18
    else:
        y_frac = 0.16

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
    span = x_right - x_left

    # Choose side deterministically but with variety.
    side = -1 if (ctx.rock.id % 2 == 0) else 1

    # Anchor can be centered-ish or offset toward a side.
    mode = ctx.rock.id % 3

    if mode == 0:
        anchor_x = ctx.cx
    elif mode == 1:
        anchor_x = ctx.cx + side * 0.18 * span
    else:
        anchor_x = ctx.cx + side * 0.28 * span

    anchor_x = max(x_left + 0.06 * span, min(x_right - 0.06 * span, anchor_x))

    # Tail scale
    tail_len = ctx.py_rng.uniform(0.42, 0.65) * ctx.unit
    tail_drop = ctx.py_rng.uniform(0.30, 0.48) * ctx.unit
    tip_size = ctx.py_rng.uniform(0.07, 0.11) * ctx.unit

    return {
        "anchor": (anchor_x, y),
        "side": side,
        "tail_len": tail_len,
        "tail_drop": tail_drop,
        "tip_size": tip_size,
        "y_frac": y_frac,
        "span": span,
    }

def make_tail_tip(center, size, side=1, rng=None):
    """
    Small tail tip shape.
    Slight variation between pebble/diamond-ish shapes.
    """
    if rng is None:
        rng = random.Random(0)

    cx, cy = center
    mode = rng.randint(0, 2)

    if mode == 0:
        # Little rounded pebble-like diamond
        pts = [
            [cx, cy + 1.00 * size],
            [cx + 0.85 * size, cy + 0.35 * size],
            [cx + 0.70 * size, cy - 0.75 * size],
            [cx - 0.55 * size, cy - 0.70 * size],
            [cx - 0.90 * size, cy + 0.10 * size],
        ]
    elif mode == 1:
        # Squatter polygon
        pts = [
            [cx - 0.75 * size, cy + 0.20 * size],
            [cx - 0.20 * size, cy + 0.95 * size],
            [cx + 0.80 * size, cy + 0.30 * size],
            [cx + 0.70 * size, cy - 0.65 * size],
            [cx - 0.35 * size, cy - 0.80 * size],
        ]
    else:
        # Simple diamond-ish point
        pts = [
            [cx, cy + 1.00 * size],
            [cx + 0.85 * size, cy],
            [cx, cy - 0.95 * size],
            [cx - 0.80 * size, cy],
        ]

    return pts

def draw_tail(ctx):
    """
    Draw a curved tail attached to the lower body.

    Style:
    - starts from bottom/lower-side of rock
    - curves outward and down
    - ends in a small tip shape
    - slight deterministic variation
    """
    tail_trait = ctx.v.get("tails", "n/a")

    if tail_trait == "n/a":
        return None

    layout = get_tail_layout(ctx)

    x0, y0 = layout["anchor"]
    side = layout["side"]
    tail_len = layout["tail_len"]
    tail_drop = layout["tail_drop"]
    tip_size = layout["tip_size"]

    # Control points for a nice curving tail.
    c1 = (
        x0 + side * 0.02 * ctx.unit,
        y0 - 0.18 * tail_drop
    )

    c2 = (
        x0 + side * 0.10 * tail_len,
        y0 - 0.68 * tail_drop
    )

    end = (
        x0 + side * tail_len,
        y0 - tail_drop
    )

    # Optional extra bend / curl
    bend = ctx.rock.id % 3

    if bend == 0:
        c2 = (c2[0] + side * 0.10 * tail_len, c2[1] - 0.05 * ctx.unit)
    elif bend == 1:
        c2 = (c2[0] - side * 0.06 * tail_len, c2[1] + 0.04 * ctx.unit)
    else:
        c1 = (c1[0] + side * 0.05 * tail_len, c1[1])

    tail_path = Path(
        [ (x0, y0), c1, c2, end ],
        [ Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4 ]
    )

    tail = PathPatch(
        tail_path,
        facecolor="none",
        edgecolor="black",
        linewidth=2.6,
        zorder=0.6,
        capstyle="round",
        joinstyle="round"
    )
    ctx.ax.add_patch(tail)

    # Tail tip
    tip_center = end
    tip_pts = make_tail_tip(tip_center, tip_size, side=side, rng=ctx.py_rng)

    tip = Polygon(
        tip_pts,
        closed=True,
        facecolor=ctx.body_color,
        edgecolor="black",
        linewidth=1.4,
        zorder=0.7,
        joinstyle="round"
    )
    ctx.ax.add_patch(tip)

    return {
        "layout": layout,
        "tail": tail,
        "tip": tip,
        "end": end
    }

# -----------------------------
# DRAWING HORNS FOR ROCK
# -----------------------------

def get_horn_layout(ctx):
    """
    Compute consistent horn placement from the head shape.

    Horns are placed near the top of the body using the width of the rock
    at a high body slice. Only placement and scale depend on the rock;
    horn shape itself stays canned/consistent.
    """
    horn_type = ctx.v.get("horns", "n/a")

    if horn_type == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Use a high slice to determine "head width"
    if shape == "triangle":
        y_frac = 0.80
    elif shape == "oblong":
        y_frac = 0.84
    else:
        y_frac = 0.83

    x_left, x_right, y_band = body_span_at_fraction(ctx, y_frac)
    local_width = x_right - x_left

    top_y = ctx.ymax

    # Horn anchors sit a bit inward from the high-side body edges.
    inset = 0.16 * local_width
    left_base = (x_left + inset, y_band + 0.02 * ctx.unit)
    right_base = (x_right - inset, y_band + 0.02 * ctx.unit)

    # Scale from rock size, but clamp so they stay visually consistent.
    horn_scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)

    return {
        "type": horn_type,
        "left_base": left_base,
        "right_base": right_base,
        "top_y": top_y,
        "local_width": local_width,
        "scale": horn_scale
    }

def make_canned_horn_template():
    """
    Canonical horn shape in local coordinates.

    Base is centered near (0, 0).
    Horn points upward with a slight outward lean.
    This template is used for both horns; right horn is mirrored.
    """
    pts = np.array([
        [-0.18,  0.00],   # base left
        [-0.10,  0.18],
        [-0.04,  0.42],
        [ 0.02,  0.74],   # tip
        [ 0.12,  0.48],
        [ 0.16,  0.20],
        [ 0.18,  0.00],   # base right
        [ 0.08,  0.10],   # inner ridge start
        [ 0.00,  0.28],   # inner ridge mid
    ])
    return pts

def draw_single_horn(ctx, base, side=1, horn_scale=1.0):
    """
    Draw one horn using a canned horn template.

    side:
    -1 = left horn
    +1 = right horn

    Only placement and scale vary. Shape stays consistent.
    """
    bx, by = base
    template = make_canned_horn_template().copy()

    # Scale horn to body size.
    w = 0.42 * horn_scale
    h = 0.62 * horn_scale

    # Transform template into horn coordinates.
    horn_pts = template.copy()
    horn_pts[:, 0] *= w
    horn_pts[:, 1] *= h

    # Mirror for left horn so both point outward.
    if side == -1:
        horn_pts[:, 0] *= -1

    # Small outward lean shift.
    horn_pts[:, 0] += side * 0.03 * horn_scale

    # Translate to base point.
    horn_pts[:, 0] += bx
    horn_pts[:, 1] += by

    # Outer horn polygon uses first 7 points.
    poly_pts = horn_pts[:7]

    horn = Polygon(
        poly_pts,
        closed=True,
        facecolor="tan",
        edgecolor="black",
        linewidth=1.3,
        zorder=12,
        joinstyle="round"
    )
    ctx.ax.add_patch(horn)

    # Inner ridge line from remaining points.
    ridge = horn_pts[7:]
    ctx.ax.plot(
        ridge[:, 0],
        ridge[:, 1],
        color="black",
        linewidth=0.8,
        alpha=0.35,
        zorder=13
    )

    return horn

def draw_horns(ctx):
    """
    Draw classic two horns using a canned horn shape and head-based placement.
    """
    horn_type = ctx.v.get("horns", "n/a")

    if horn_type == "n/a":
        return None

    layout = get_horn_layout(ctx)

    left_horn = draw_single_horn(
        ctx,
        layout["left_base"],
        side=-1,
        horn_scale=layout["scale"]
    )

    right_horn = draw_single_horn(
        ctx,
        layout["right_base"],
        side=1,
        horn_scale=layout["scale"]
    )

    return {
        "type": horn_type,
        "left": left_horn,
        "right": right_horn,
        "layout": layout
    }

# -----------------------------
# DRAWING PATCHWORK FOR ROCK
# -----------------------------

def draw_patchwork(ax, body_patch, color_alleles, s, rng):
    """
    Draw dispersed random patches only if body color expresses as patchwork.

    Since patchwork is recessive, this should only happen for:
    patchwork + patchwork.
    """
    body_color_name = express_body_color_name(color_alleles)

    if body_color_name != "patchwork":
        return

    patch_palette = [
        BODY_COLOR_MAP["white"],
        BODY_COLOR_MAP["black"],
        BODY_COLOR_MAP["silver"],
        BODY_COLOR_MAP["brown"],
        BODY_COLOR_MAP["red"],
        BODY_COLOR_MAP["yellow"],
        BODY_COLOR_MAP["blue"],
        BODY_COLOR_MAP["orange"],
        BODY_COLOR_MAP["green"],
        BODY_COLOR_MAP["purple"],
    ]

    n_patches = random.randint(1,5) * random.randint(1,4) + random.randint(1,6)

    for i in range(n_patches):
        cx = rng.uniform(-0.85 * s, 0.85 * s)
        cy = rng.uniform(-0.75 * s, 0.85 * s)

        r = rng.uniform(0.16 * s, 0.40 * s)
        n_sides = int(rng.integers(5, 9))

        theta0 = rng.uniform(0, 2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + theta0

        points = []

        for a in angles:
            rr = r * rng.uniform(0.55, 1.15)
            points.append([
                cx + rr * np.cos(a),
                cy + rr * np.sin(a)
            ])

        patch = Polygon(
            points,
            closed=True,
            facecolor=patch_palette[i % len(patch_palette)],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.55,
            zorder=2.5
        )

        patch.set_clip_path(body_patch)
        ax.add_patch(patch)

# -----------------------------
# DRAWING HAIR + CURLY FOR ROCK
# -----------------------------

def get_hair_layout(ctx):
    """
    Compute head-aware anchor geometry for hair.
    """

    hair_type = ctx.v.get("hair", "n/a")

    if hair_type == "n/a":
        return None

    # Top cap width
    top_left, top_right, top_y = body_span_at_fraction(ctx, 0.88)

    # Forehead band for front hairline
    head_left, head_right, head_y = body_span_at_fraction(ctx, 0.74)

    top_span = top_right - top_left
    head_span = head_right - head_left

    return {
        "type": hair_type,
        "top_left": top_left,
        "top_right": top_right,
        "top_y": top_y,
        "head_left": head_left,
        "head_right": head_right,
        "head_y": head_y,
        "top_span": top_span,
        "head_span": head_span,
        "cx": ctx.cx,
        "cy": ctx.cy
    }

def get_render_hair_color(ctx):
    return get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

def draw_curly_overlay_in_box(
    ctx,
    x_min,
    x_max,
    y_min,
    y_max,
    hair_color,
    n_curls=10,
    curl_scale=0.10,
    zorder=60,
    salt="head_curls"
):
    """
    Draw randomized semicircle curl marks inside an approximate hair region.

    The region is a simple bounding box, but the arcs are decorative and
    read well over both head hair and facial hair.
    """
    if x_max <= x_min or y_max <= y_min:
        return []

    rng = deterministic_rng_for_rock(ctx.rock, salt=salt)

    light = adjust_color_brightness(hair_color, 1.45)
    dark = adjust_color_brightness(hair_color, 0.65)

    curls = []

    box_w = x_max - x_min
    box_h = y_max - y_min

    for i in range(n_curls):
        x = rng.uniform(x_min + 0.08 * box_w, x_max - 0.08 * box_w)
        y = rng.uniform(y_min + 0.15 * box_h, y_max - 0.10 * box_h)

        size = rng.uniform(0.65, 1.15) * curl_scale * ctx.unit

        # Alternate light/dark so curls show on many colors.
        color = light if i % 2 == 0 else dark

        # Mostly semicircles, slightly rotated.
        angle = rng.uniform(-25, 25)

        # Randomize arc direction a bit.
        if rng.random() < 0.5:
            theta1, theta2 = 0, 180
        else:
            theta1, theta2 = 180, 360

        arc = draw_curl_arc(
            ctx.ax,
            x=x,
            y=y,
            w=size,
            h=0.72 * size,
            color=color,
            angle=angle,
            theta1=theta1,
            theta2=theta2,
            linewidth=max(0.8, 1.05 * ctx.unit),
            alpha=0.9,
            zorder=zorder
        )

        curls.append(arc)

    return curls

def draw_head_hair_curls(ctx, hair_color, layout, hair_type):
    """
    Add curl marks over head hair if hair_texture is curly.
    """
    if not rock_texture_is_curly(ctx):
        return []

    head_left = layout["head_left"]
    head_right = layout["head_right"]
    head_y = layout["head_y"]
    top_y = layout["top_y"]
    cx = layout["cx"]
    head_span = layout["head_span"]

    # Approximate curl region depending on hairstyle.
    if hair_type == "hair":
        x_min = head_left + 0.03 * head_span
        x_max = head_right - 0.03 * head_span
        y_min = head_y - 0.10 * ctx.unit
        y_max = top_y + 0.25 * ctx.unit
        n_curls = 8

    elif hair_type == "double hair":
        x_min = head_left - 0.20 * head_span
        x_max = head_right + 0.20 * head_span
        y_min = head_y - 0.18 * ctx.unit
        y_max = top_y + 0.30 * ctx.unit
        n_curls = 13

    else:
        x_min = cx - 0.40 * head_span
        x_max = cx + 0.40 * head_span
        y_min = head_y - 0.10 * ctx.unit
        y_max = top_y + 0.25 * ctx.unit
        n_curls = 8

    return draw_curly_overlay_in_box(
        ctx,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        hair_color=hair_color,
        n_curls=n_curls,
        curl_scale=0.095,
        zorder=72,
        salt=f"head_curls_{hair_type}"
    )

def draw_hair(ctx, rock, v):
    """
    Draw canned hairstyles inspired by the sketch.

    Styles:
    - femme_side
    - femme_long
    - masc_short
    - masc_spike
    """
    hair_type = ctx.v.get("hair", "n/a")
    #print(hair_type)
    #gender = rock.genes["gender"]
    gender = v["gender"]
    #print(gender)

    if hair_type == "n/a":
        return None

    layout = get_hair_layout(ctx)
    hair_color = get_render_hair_color(ctx)

    top_left = layout["top_left"]
    top_right = layout["top_right"]
    top_y = layout["top_y"]

    head_left = layout["head_left"]
    head_right = layout["head_right"]
    head_y = layout["head_y"]

    cx = layout["cx"]
    head_span = layout["head_span"]
    top_span = layout["top_span"]

    z = 11

    pieces = []

    def add_arc_cap(
        left_x,
        right_x,
        base_y,
        top_bump=0.12,
        lower_dip=0.08,
        zorder=11
    ):
        """
        Shared hair cap.

        It uses:
        - a top arc rising over the head
        - a lower arc dipping across the forehead

        This makes the visible top hairline smooth from left to right.
        """

        top_peak_y = top_y + top_bump * ctx.unit
        lower_y = base_y - lower_dip * ctx.unit

        verts = [
            # Start at left hairline
            (left_x, base_y),

            # Top arc: left -> right
            (left_x + 0.20 * (right_x - left_x), top_peak_y),
            (cx - 0.15 * (right_x - left_x), top_peak_y),
            (cx, top_peak_y),

            (cx + 0.15 * (right_x - left_x), top_peak_y),
            (right_x - 0.20 * (right_x - left_x), top_peak_y),
            (right_x, base_y),

            # Lower forehead arc: right -> left
            (right_x - 0.18 * (right_x - left_x), lower_y),
            (cx + 0.18 * (right_x - left_x), lower_y),
            (cx, lower_y),

            (cx - 0.18 * (right_x - left_x), lower_y),
            (left_x + 0.18 * (right_x - left_x), lower_y),
            (left_x, base_y),
        ]

        codes = [
            Path.MOVETO,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,

            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        cap = PathPatch(
            Path(verts, codes),
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=zorder,
            joinstyle="round",
            capstyle="round"
        )

        ctx.ax.add_patch(cap)
        return cap

    # Shared cap geometry.
    cap_left = head_left + 0.03 * head_span
    cap_right = head_right - 0.03 * head_span
    cap_base_y = head_y + 0.04 * ctx.unit

    # --------------------------------------------------
    # 1) FEMME SIDE — side sweep with one long lock
    # --------------------------------------------------
    if hair_type == "hair" and gender == "Female":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        # Long side lock to the right
        lock_pts = [
            [cx + 0.08 * head_span, top_y + 0.16 * ctx.unit],
            [cx + 0.34 * head_span, top_y + 0.26 * ctx.unit],
            [cx + 0.74 * head_span, head_y + 0.12 * ctx.unit],
            [cx + 0.86 * head_span, head_y - 0.08 * ctx.unit],
            [cx + 0.62 * head_span, head_y - 0.04 * ctx.unit],
            [cx + 0.46 * head_span, head_y + 0.10 * ctx.unit],
            [cx + 0.28 * head_span, head_y + 0.18 * ctx.unit],
        ]

        lock = Polygon(
            lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )
        ctx.ax.add_patch(lock)
        pieces.append(lock)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 2) FEMME LONG — twin long draping locks
    # --------------------------------------------------
    elif hair_type == "double hair" and gender == "Female":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        left_lock_pts = [
            [head_left + 0.18 * head_span, top_y + 0.14 * ctx.unit],
            [head_left - 0.14 * head_span, top_y + 0.24 * ctx.unit],
            [head_left - 0.40 * head_span, head_y + 0.06 * ctx.unit],
            [head_left - 0.54 * head_span, head_y - 0.16 * ctx.unit],
            [head_left - 0.26 * head_span, head_y - 0.14 * ctx.unit],
            [head_left - 0.08 * head_span, head_y + 0.02 * ctx.unit],
            [head_left + 0.12 * head_span, head_y + 0.14 * ctx.unit],
        ]

        right_lock_pts = [
            [head_right - 0.18 * head_span, top_y + 0.14 * ctx.unit],
            [head_right + 0.14 * head_span, top_y + 0.24 * ctx.unit],
            [head_right + 0.40 * head_span, head_y + 0.06 * ctx.unit],
            [head_right + 0.54 * head_span, head_y - 0.16 * ctx.unit],
            [head_right + 0.26 * head_span, head_y - 0.14 * ctx.unit],
            [head_right + 0.08 * head_span, head_y + 0.02 * ctx.unit],
            [head_right - 0.12 * head_span, head_y + 0.14 * ctx.unit],
        ]

        left_lock = Polygon(
            left_lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )

        right_lock = Polygon(
            right_lock_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )

        ctx.ax.add_patch(left_lock)
        ctx.ax.add_patch(right_lock)
        pieces.extend([left_lock, right_lock])

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 3) MASC SHORT — short top hair
    # --------------------------------------------------
    elif hair_type == "hair" and gender == "Male":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
        )
        pieces.append(cap)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    # --------------------------------------------------
    # 4) MASC SPIKE — fuller cap with side spike
    # --------------------------------------------------
    elif hair_type == "double hair" and gender == "Male":
        cap = add_arc_cap(
            cap_left,
            cap_right,
            cap_base_y,
            zorder=z
          )
        pieces.append(cap)

        spike_pts = [
            [cx + 0.08 * head_span, top_y + 0.10 * ctx.unit],
            [cx + 0.25 * head_span, top_y + 0.32 * ctx.unit],
            [head_right + 0.16 * head_span, top_y + 0.42 * ctx.unit],
            [head_right + 0.06 * head_span, head_y + 0.12 * ctx.unit],
            [cx + 0.18 * head_span, head_y + 0.02 * ctx.unit],
        ]

        spike = Polygon(
            spike_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.4,
            zorder=z + 1,
            joinstyle="round"
        )
        ctx.ax.add_patch(spike)
        pieces.append(spike)

        pieces.extend(
            draw_head_hair_curls(
                ctx,
                hair_color=hair_color,
                layout=layout,
                hair_type=hair_type
            )
        )

        return pieces

    return pieces

# -----------------------------
# DRAWING EAR FOR ROCK
# -----------------------------

def get_ear_layout(ctx):
    """
    Compute ear anchors and scale from the rock head.

    Ears use canned shapes, while placement/scale comes from the body.
    """
    ear_type = ctx.v.get("ears", "n/a")

    if ear_type == "n/a":
        return None

    shape = ctx.v.get("shape", "circle")

    # Different ear types want different vertical bands.
    if ear_type in ["antannae", "antanna"]:
        y_frac = 0.82 if shape != "triangle" else 0.76
    elif ear_type in ["ears", "ear"]:
        y_frac = 0.66 if shape != "triangle" else 0.58
    elif ear_type in ["ogre", "ogres"]:
        y_frac = 0.80 if shape != "triangle" else 0.74
    elif ear_type in ["goblins", "goblin"]:
        y_frac = 0.82 if shape != "triangle" else 0.76
    else:
        y_frac = 0.75

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
    local_width = x_right - x_left

    # Inset slightly from the exact edge for top-attached ear types.
    if ear_type in ["antannae", "antanna", "ogre", "ogres", "goblins", "goblin"]:
        inset = 0.10 * local_width
        left_base = (x_left + inset, y + 0.01 * ctx.unit)
        right_base = (x_right - inset, y + 0.01 * ctx.unit)
    else:
        # Rounded side ears sit right at the side region.
        left_base = (x_left, y)
        right_base = (x_right, y)

    # Moderate size, not too wild.
    if ear_type in ["ears", "ear"]:
        scale = np.clip(0.85 * ctx.unit, 0.70, 1.25)
    elif ear_type in ["ogre", "ogres"]:
        scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)
    elif ear_type in ["goblins", "goblin"]:
        scale = np.clip(1.00 * ctx.unit, 0.80, 1.40)
    else:
        scale = np.clip(0.95 * ctx.unit, 0.75, 1.35)

    return {
        "type": ear_type,
        "left_base": left_base,
        "right_base": right_base,
        "y_frac": y_frac,
        "local_width": local_width,
        "scale": scale
    }

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

def draw_single_filled_ear(ctx, base, side=1, template="round", ear_scale=1.0):
    """
    Draw one filled ear using a canned polygon template.
    """

    if template == "round":
        tmpl = make_round_ear_template()
        sx = 0.42 * ear_scale
        sy = 0.42 * ear_scale
        z = 11
    elif template == "ogre":
        tmpl = make_ogre_ear_template()
        sx = 0.50 * ear_scale
        sy = 0.50 * ear_scale
        z = 11
    else:  # goblin
        tmpl = make_goblin_ear_template()
        sx = 0.48 * ear_scale
        sy = 0.54 * ear_scale
        z = 11

    pts = transform_template_points(
        tmpl,
        base=base,
        side=side,
        sx=sx,
        sy=sy
    )

    ear = Polygon(
        pts,
        closed=True,
        facecolor=ctx.body_color,
        edgecolor="black",
        linewidth=1.5,
        zorder=z,
        joinstyle="round"
    )
    ctx.ax.add_patch(ear)

    # Small inner ear accent
    inner_pts = transform_template_points(
        tmpl * np.array([0.55, 0.55]),
        base=base,
        side=side,
        sx=sx,
        sy=sy,
        dx=0.01 * side * ear_scale,
        dy=0.04 * ear_scale
    )

    inner = Polygon(
        inner_pts,
        closed=False,
        fill=False,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.35,
        zorder=z + 1,
        joinstyle="round"
    )
    ctx.ax.add_patch(inner)

    return ear

def draw_single_antanna(ctx, base, side=1, ear_scale=1.0):
    """
    Draw one antenna:
    - short stalk
    - 3 prong-like tips
    """
    bx, by = base

    stalk_len = 0.52 * ear_scale
    stalk_rise = 0.38 * ear_scale

    # Main elbow / tip direction
    mx = bx + side * 0.18 * stalk_len
    my = by + 0.45 * stalk_rise

    tx = bx + side * 0.55 * stalk_len
    ty = by + stalk_rise

    # Main stalk
    ctx.ax.plot(
        [bx, mx, tx],
        [by, my, ty],
        color="black",
        linewidth=2.2,
        zorder=11,
        solid_capstyle="round"
    )

    # Three little prongs
    prong_len = 0.12 * ear_scale
    prong_angles = [2.3, 1.75, 1.2]  # visually nice spread

    for ang in prong_angles:
        # flip horizontally for left side
        dx = side * prong_len * math.cos(ang)
        dy = prong_len * math.sin(ang)

        ctx.ax.plot(
            [tx, tx + dx],
            [ty, ty + dy],
            color="black",
            linewidth=2.0,
            zorder=12,
            solid_capstyle="round"
        )

    return {
        "base": base,
        "tip": (tx, ty)
    }

def draw_ears(ctx):
    """
    Draw ear traits using canned shapes + ctx-based placement.

    Supported:
    - antennae
    - ears
    - ogre
    - goblins
    """
    ear_type = ctx.v.get("ears", "n/a")

    if ear_type == "n/a":
        return None

    layout = get_ear_layout(ctx)
    left_base = layout["left_base"]
    right_base = layout["right_base"]
    scale = layout["scale"]

    if ear_type in ["antannae", "antanna"]:
        left = draw_single_antanna(ctx, left_base, side=-1, ear_scale=scale)
        right = draw_single_antanna(ctx, right_base, side=1, ear_scale=scale)

    elif ear_type in ["ears", "ear"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="round", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="round", ear_scale=scale)

    elif ear_type in ["ogre", "ogres"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="ogre", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="ogre", ear_scale=scale)

    elif ear_type in ["goblins", "goblin"]:
        left = draw_single_filled_ear(ctx, left_base, side=-1, template="goblin", ear_scale=scale)
        right = draw_single_filled_ear(ctx, right_base, side=1, template="goblin", ear_scale=scale)

    else:
        return None

    return {
        "type": ear_type,
        "left": left,
        "right": right,
        "layout": layout
    }

# -----------------------------
# DRAWING WRINKLES FOR ROCK
# -----------------------------

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

    try:
        gene = str(ctx.rock.genes.get("hair_texture", "00"))
        return gene == "11"
    except Exception:
        return False

def deterministic_rng_for_rock(rock, salt="curl"):
    """
    Deterministic random generator so curls do not jump around every redraw.
    """
    seed_text = f"{getattr(rock, 'id', 0)}_{salt}_{str(getattr(rock, 'genes', {}))}"
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

def get_wrinkle_color(body_color):
    """
    Adaptive wrinkle color:
    - lighten dark rocks
    - darken light rocks

    This keeps wrinkle lines visible on all body colors.
    """
    lum = color_luminance(body_color)

    if lum < 0.45:
        # Dark body -> lighter wrinkles
        return mix_colors(body_color, (1, 1, 1), t=0.45)
    else:
        # Light body -> darker wrinkles
        return mix_colors(body_color, (0, 0, 0), t=0.35)

def get_wrinkle_y_fractions(ctx, n_lines=5):
    """
    Choose wrinkle bands across the body.
    Focus on the broad middle/interior regions.
    """
    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    elif shape == "oblong":
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    else:
        base = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]

    if n_lines <= len(base):
        return base[:n_lines]

    extra = list(np.linspace(0.9, 0.1, n_lines))
    return extra

def draw_wrinkles(ctx):
    """
    Draw surface wrinkle lines across the rock.

    Style:
    - irregular short/medium lines
    - body-aware span
    - clipped to body
    - adaptive color so visible on all body colors
    """

    wrinkle_type = ctx.v.get("wrinkles", "n/a")

    if wrinkle_type == "n/a":
        return None

    wrinkle_color = get_wrinkle_color(ctx.body_color)
    lines_drawn = []

    # Number of wrinkle rows.
    # If you later want stronger expression for double-alleles,
    # we can increase this based on allele count.
    y_fracs = get_wrinkle_y_fractions(ctx, n_lines=5)

    for idx, y_frac in enumerate(y_fracs):
        x_left, x_right, y = body_span_at_fraction(ctx, y_frac)
        local_width = x_right - x_left

        # Keep wrinkles a bit inside the edges.
        margin = 0.01 * local_width
        usable_left = x_left + margin
        usable_right = x_right - margin

        if usable_right <= usable_left:
            continue

        # Start point and total line length.
        total_len = ctx.py_rng.uniform(0.5 * local_width, 0.95 * local_width)
        start_x = ctx.py_rng.uniform(usable_left - total_len / 2 + local_width / 2, usable_left - total_len / 2 + local_width / 2 + 0.05 * local_width)

        # Build a small jagged / wavy wrinkle path.
        n_segments = ctx.py_rng.randint(4, 7)

        xs = [start_x]
        ys = [y + ctx.py_rng.uniform(-0.015, 0.015) * ctx.unit]

        current_x = start_x
        current_y = ys[0]

        for s in range(n_segments):
            dx = total_len / n_segments
            dy = ctx.py_rng.uniform(-0.06, 0.06) * ctx.unit

            # Sometimes make a more angular “step”.
            if s % 2 == 1 and ctx.py_rng.random() < 0.35:
                dy *= 0.4

            current_x += dx
            current_y += dy

            if current_x > usable_right:
                break

            xs.append(current_x)
            ys.append(current_y)

        line, = ctx.ax.plot(
            xs,
            ys,
            color=wrinkle_color,
            linewidth=1.6,
            alpha=0.95,
            zorder=4,
            solid_capstyle="round"
        )
        line.set_clip_path(ctx.body)
        lines_drawn.append(line)

    return lines_drawn

# -----------------------------
# DRAWING FRECKLES FOR ROCK
# -----------------------------

def get_freckle_color(body_color):
    """
    Adaptive freckle color.

    Freckles should look like small mineral inclusions.
    - on light rocks: darker freckles
    - on dark rocks: lighter freckles
    """
    lum = color_luminance(body_color)

    if lum < 0.42:
        return mix_colors(body_color, (1, 1, 1), t=0.55)
    else:
        return mix_colors(body_color, (0, 0, 0), t=0.45)

def random_point_in_body(ctx, max_attempts=100):
    """
    Sample a random point inside the body polygon.
    """
    body_path = Path(ctx.body_points)

    for _ in range(max_attempts):
        x = ctx.py_rng.uniform(ctx.xmin, ctx.xmax)
        y = ctx.py_rng.uniform(ctx.ymin, ctx.ymax)

        if body_path.contains_point((x, y)):
            return x, y

    # Fallback to center if sampling somehow fails.
    return ctx.cx, ctx.cy

def draw_freckles(ctx):
    """
    Draw freckles/mineral speckles on the body surface.

    Current expression:
    - if freckles phenotype is n/a, draw none
    - if freckles are active, draw scattered small dots

    Uses active allele count as intensity:
    - one active allele: fewer freckles
    - two active alleles: more freckles

    If your phenotype says freckles only express at 11, this still works.
    """
    freckle_trait = ctx.v.get("freckles", "n/a")

    if freckle_trait == "n/a":
        return None

    values = ctx.v.get("freckles_values", [])
    active_count = sum(1 for val in values if val != 0)

    # If somehow active_count is zero but phenotype says freckles, default to 1.
    active_count = max(1, active_count)

    # Number and size scale.
    if active_count == 1:
        n_freckles = 10
        r_min = 0.010 * ctx.unit
        r_max = 0.024 * ctx.unit
    else:
        n_freckles = 20
        r_min = 0.012 * ctx.unit
        r_max = 0.032 * ctx.unit

    freckle_color = get_freckle_color(ctx.body_color)

    freckles = []

    for _ in range(n_freckles):
        x, y = random_point_in_body(ctx)

        # Bias freckles slightly toward the visible/front middle,
        # so they do not all vanish near the silhouette.
        if ctx.py_rng.random() < 0.55:
            x = 0.72 * x + 0.28 * ctx.cx
            y = 0.80 * y + 0.20 * ctx.cy

        radius = ctx.py_rng.uniform(r_min, r_max)

        dot = Circle(
            (x, y),
            radius=radius,
            facecolor=freckle_color,
            edgecolor="none",
            alpha=0.80,
            zorder=5
        )

        dot.set_clip_path(ctx.body)
        ctx.ax.add_patch(dot)
        freckles.append(dot)

    return freckles

# -----------------------------
# DRAWING ARMS FOR ROCK
# -----------------------------

def get_arm_y_fractions(ctx, n_pairs):
    """
    Choose vertical attachment positions for arm pairs.

    The values are body-height fractions:
    0 = bottom of body
    1 = top of body

    We keep arms in the middle band to avoid eyes/mouth/hair/feet-space.
    """

    if n_pairs <= 0:
        return []

    shape = ctx.v.get("shape", "circle")

    if shape == "triangle":
        # Triangle is narrow high up, so arms attach slightly lower.
        if n_pairs == 1:
            return [0.50]
        if n_pairs == 2:
            return [0.50, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "oblong":
        # Oblongs are wide, can support a clean mid-band.
        if n_pairs == 1:
            return [0.50]
        if n_pairs == 2:
            return [0.50, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "oval":
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    elif shape == "square":
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

    else:
        if n_pairs == 1:
            return [0.5]
        if n_pairs == 2:
            return [0.5, 0.2]
        return np.linspace(0.5, 0.2, n_pairs)

def get_arm_anchor_points(ctx, n_pairs):
    """
    Returns arm anchor points on the actual left and right body edges.

    Output:
    [
        {
            "left":  (x_left, y),
            "right": (x_right, y),
            "y_frac": y_frac,
            "span": x_right - x_left
        },
        ...
    ]
    """
    y_fracs = get_arm_y_fractions(ctx, n_pairs)
    anchors = []

    for y_frac in y_fracs:
        x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

        anchors.append({
            "left": (x_left, y),
            "right": (x_right, y),
            "y_frac": y_frac,
            "span": x_right - x_left
        })

    return anchors

def draw_single_normal_arm(ctx, attach, side=1, layer_offset=0):
    """
    Draw one stick-style arm connected to the body edge.

    side = -1 for left, +1 for right
    """

    ax = ctx.ax
    x0, y0 = attach

    # Scale arm length to rock size but not wildly.
    arm_len = 0.42 * ctx.unit
    forearm_len = 0.28 * ctx.unit

    # Slight deterministic pose variation.
    bend = ctx.py_rng.uniform(-0.1, 0.1) * ctx.unit

    elbow_x = x0 + side * arm_len
    elbow_y = y0 + 0.10 * ctx.unit + bend

    hand_x = elbow_x + side * forearm_len
    hand_y = elbow_y - 0.16 * ctx.unit

    # Tiny shoulder dot at exact edge connection.
    ax.add_patch(
        Circle(
            (x0, y0),
            radius=0.030 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3 + layer_offset
        )
    )

    # Upper arm and forearm.
    ax.plot(
        [x0, elbow_x],
        [y0, elbow_y],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    ax.plot(
        [elbow_x, hand_x],
        [elbow_y, hand_y],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Hand.
    ax.add_patch(
        Circle(
            (hand_x, hand_y),
            radius=0.055 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3 + layer_offset
        )
    )

    return {
        "shoulder": (x0, y0),
        "elbow": (elbow_x, elbow_y),
        "hand": (hand_x, hand_y)
    }

def draw_single_muscle_arm(ctx, attach, side=1, layer_offset=0):
    """
    Draw one thicker muscle arm connected to the body edge.
    """

    ax = ctx.ax
    x0, y0 = attach

    upper_len = 0.42 * ctx.unit
    fore_len = 0.32 * ctx.unit

    elbow_x = x0 + side * upper_len
    elbow_y = y0 + 0.08 * ctx.unit

    hand_x = elbow_x + side * fore_len
    hand_y = elbow_y - 0.13 * ctx.unit

    # Shoulder connector.
    ax.add_patch(
        Circle(
            (x0, y0),
            radius=0.045 * ctx.unit,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    # Thick upper arm.
    ax.plot(
        [x0, elbow_x],
        [y0, elbow_y],
        color="black",
        linewidth=5,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Bicep bulge.
    bicep_x = x0 + side * 0.58 * upper_len
    bicep_y = y0 + 0.05 * ctx.unit

    ax.add_patch(
        Ellipse(
            (bicep_x, bicep_y),
            width=0.22 * ctx.unit,
            height=0.12 * ctx.unit,
            angle=side * 15,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    # Thick forearm.
    ax.plot(
        [elbow_x, hand_x],
        [elbow_y, hand_y],
        color="black",
        linewidth=4,
        solid_capstyle="round",
        zorder=2 + layer_offset
    )

    # Fist.
    ax.add_patch(
        Ellipse(
            (hand_x, hand_y),
            width=0.18 * ctx.unit,
            height=0.14 * ctx.unit,
            angle=side * -12,
            facecolor=ctx.body_color,
            edgecolor="black",
            linewidth=1.1,
            zorder=3 + layer_offset
        )
    )

    return {
        "shoulder": (x0, y0),
        "elbow": (elbow_x, elbow_y),
        "hand": (hand_x, hand_y)
    }

def draw_arms(ctx):
    """
    Draw arms based on co-dominant arm alleles.

    arms gene:
    0 = no arm
    1 = normal arm pair
    2 = muscle arm pair

    Examples:
    00 -> no arms
    01 -> 2 normal arms
    11 -> 4 normal arms
    02 -> 2 muscle arms
    12 -> 2 normal arms + 2 muscle arms
    22 -> 4 muscle arms
    """

    normal_pairs = ctx.v.get("normal_arm_pairs", 0)
    muscle_pairs = ctx.v.get("muscle_arm_pairs", 0)

    total_pairs = normal_pairs + muscle_pairs

    if total_pairs <= 0:
        return []

    anchors = get_arm_anchor_points(ctx, total_pairs)

    drawn = []

    pair_types = []

    # Put muscle arms first so they usually appear slightly behind normal arms.
    for _ in range(muscle_pairs):
        pair_types.append("muscle")

    for _ in range(normal_pairs):
        pair_types.append("normal")

    for i, pair_type in enumerate(pair_types):
        anchor = anchors[i]

        left_attach = anchor["left"]
        right_attach = anchor["right"]

        # Lower pairs draw slightly behind upper pairs.
        layer_offset = i

        if pair_type == "muscle":
            drawn.append(draw_single_muscle_arm(ctx, left_attach, side=-1, layer_offset=layer_offset))
            drawn.append(draw_single_muscle_arm(ctx, right_attach, side=1, layer_offset=layer_offset))

        else:
            drawn.append(draw_single_normal_arm(ctx, left_attach, side=-1, layer_offset=layer_offset))
            drawn.append(draw_single_normal_arm(ctx, right_attach, side=1, layer_offset=layer_offset))

    return drawn

# -----------------------------
# DRAWING CROWNS FOR ROCK
# -----------------------------

def get_crown_layout(ctx):
    """
    Compute a top-of-head anchor and scale for crowns.

    Returns a dict with:
    - center x
    - top y of the body
    - local width near the top
    - diamond size scale
    """
    crown_type = ctx.v.get("crowns", "n/a")

    if crown_type == "n/a":
        return None

    # Sample a band slightly below the very top to estimate width.
    x_left, x_right, y_band = body_span_at_fraction(ctx, 0.90)

    top_x = ctx.cx
    top_y = ctx.ymax

    local_width = max(1e-6, x_right - x_left)

    # Base diamond size from local width and overall body size.
    diamond_w = min(0.34 * local_width, 0.24 * ctx.unit)
    diamond_h = 0.18 * ctx.unit

    return {
        "type": crown_type,
        "cx": top_x,
        "top_y": top_y,
        "local_width": local_width,
        "diamond_w": diamond_w,
        "diamond_h": diamond_h,
        "x_left": x_left,
        "x_right": x_right,
        "y_band": y_band
    }

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

def draw_crown(ctx):
    """
    Draw crown traits using ctx.

    Crown types:
    - small  -> one black diamond with gold outline
    - medium -> black / white stacked diamonds with gold outline
    - large  -> black / white / black stacked diamonds with gold outline
    - indent -> a divot cut into the top of the rock, ending at the rock edge
    """

    crown_type = ctx.v.get("crowns", "n/a")

    if crown_type == "n/a":
        return None

    layout = get_crown_layout(ctx)
    cx = layout["cx"]
    top_y = layout["top_y"]
    dw = layout["diamond_w"]
    dh = layout["diamond_h"]

    gold_edge = "gold"
    gold_lw = 1.0

    # -----------------------------
    # Indent crown: divot into head
    # -----------------------------
    if crown_type == "indent":
        bg = ctx.ax.get_facecolor()

        notch_w = 0.42 * ctx.unit
        notch_d = 0.18 * ctx.unit

        # Cut a notch exactly from the top edge downward.
        notch = Polygon(
            [
                [cx - notch_w / 2, top_y],
                [cx, top_y - notch_d],
                [cx + notch_w / 2, top_y],
            ],
            closed=True,
            facecolor=bg,
            edgecolor="none",
            zorder=15
        )
        notch.set_clip_path(ctx.body)
        ctx.ax.add_patch(notch)

        # Draw only the two sloped sides of the divot.
        # These begin and end exactly on the rock edge.
        ctx.ax.plot(
            [cx - notch_w / 2, cx],
            [top_y, top_y - notch_d],
            color="black",
            linewidth=2.1,
            zorder=16,
            clip_path=ctx.body
        )
        ctx.ax.plot(
            [cx, cx + notch_w / 2],
            [top_y - notch_d, top_y],
            color="black",
            linewidth=2.1,
            zorder=16,
            clip_path=ctx.body
        )

        return {
            "type": crown_type,
            "center": (cx, top_y),
        }

    # -----------------------------
    # Stacked diamond crowns
    # -----------------------------
    if crown_type == "small":
        stack_colors = ["black"]
    elif crown_type == "medium":
        stack_colors = ["black", "white"]
    elif crown_type == "large":
        stack_colors = ["black", "white", "black"]
    else:
        stack_colors = ["black"]

    # Bottom diamond sits slightly into the head, like your sketch.
    base_cy = top_y + 0.04 * ctx.unit

    drawn = []

    for i, fill_color in enumerate(stack_colors):
        cy = base_cy + i * (0.58 * dh)

        # Slight taper up the stack.
        scale = 1.00 - 0.06 * i
        w = dw * scale
        h = dh * scale

        diamond = Polygon(
            make_diamond(cx, cy, w, h),
            closed=True,
            facecolor=fill_color,
            edgecolor=gold_edge,
            linewidth=gold_lw,
            zorder=15 + i,
            joinstyle="miter"
        )
        ctx.ax.add_patch(diamond)
        drawn.append(diamond)

    return {
        "type": crown_type,
        "center": (cx, top_y),
        "count": len(stack_colors)
    }

# -----------------------------
# DRAWING EYES FOR ROCK
# -----------------------------

def get_eye_layout(ctx, eye_count):
    """
    Returns eye positions and eye radius that respect body shape and size.
    """
    shape = ctx.v.get("shape", "circle")

    # Triangles are narrow near the top, so eyes sit lower.
    # Oblong rocks can support wider-set eyes.
    if shape == "triangle":
        y_frac = 0.50
        spread_factor = 0.28
    elif shape == "oblong":
        y_frac = 0.58
        spread_factor = 0.34
    elif shape == "oval":
        y_frac = 0.58
        spread_factor = 0.28
    elif shape == "square":
        y_frac = 0.58
        spread_factor = 0.30
    else:
        y_frac = 0.58
        spread_factor = 0.30

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    available_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Eye radius scales with local rock size.
    eye_radius = min(
        0.085 * ctx.height,
        0.13 * available_width
    )

    eye_radius = max(eye_radius, 0.045 * ctx.unit)

    margin = 1.35 * eye_radius

    if eye_count <= 0:
        return [], eye_radius

    if eye_count == 1:
        return [(center_x, y)], eye_radius * 1.08

    # Two eyes.
    separation = available_width * spread_factor

    left_x = clamp_inside_span(center_x - separation, x_left, x_right, margin)
    right_x = clamp_inside_span(center_x + separation, x_left, x_right, margin)

    return [(left_x, y), (right_x, y)], eye_radius

def get_eye_color(color_name):
    return EYE_COLOR_MAP.get(color_name, "white")

def draw_eyes(ctx):
    """
    Draw eyes using the new shape-aware layout.
    """
    eye_count = ctx.v.get("eyes_count", 0)

    eye_positions, eye_radius = get_eye_layout(ctx, eye_count)

    if len(eye_positions) == 0:
        return []

    eye_color_name = ctx.v.get("eye_color", "black")
    eye_color = get_eye_color(eye_color_name)

    drawn_positions = []

    for ex, ey in eye_positions:
        sclera_color = eye_color #"white"

        if eye_color_name == "callus":
            sclera_color = "tan"

        eye = Circle(
            (ex, ey),
            radius=eye_radius,
            facecolor=sclera_color,
            edgecolor="black",
            linewidth=1.2,
            zorder=8
        )
        ctx.ax.add_patch(eye)

        pupil = Circle(
            (ex, ey),
            radius=0.43 * eye_radius,
            facecolor=eye_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=9
        )
        ctx.ax.add_patch(pupil)

        # Optional evil eye accent.
        if eye_color_name == "evil":
            ctx.ax.plot(
                [ex, ex],
                [ey + 0.45 * eye_radius, ey - 0.45 * eye_radius],
                color="black",
                linewidth=1.4,
                zorder=10
            )

        drawn_positions.append((ex, ey, eye_radius))

    return drawn_positions

# -----------------------------
# DRAWING BROWS FOR ROCK
# -----------------------------

def draw_brows(ctx, drawn_eye_positions):
    """
    Draw brows using eye positions from draw_eyes(ctx).

    Supports:
    - brows
    - eyehair
    - unibrows

    Uses ctx so brows respect shape, size, and actual eye placement.
    """

    brow_type = ctx.v.get("brows", "n/a")

    if brow_type == "n/a":
        return

    # Brows need eyes. If there are no eyes, skip for now.
    # Later we can make "orphan brows" into a cursed rare visual.
    if len(drawn_eye_positions) == 0:
        return

    hair_color = get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

    curly = ctx.v.get("hair_texture", "straight") == "curly"

    # Use eye radius as local scale.
    avg_eye_radius = sum(r for _, _, r in drawn_eye_positions) / len(drawn_eye_positions)

    # Brow y-position sits above the eye.
    brow_lift = 1.35 * avg_eye_radius

    # Body slice near brow height for clamping.
    # Use the first eye to estimate face band.
    sample_eye_y = drawn_eye_positions[0][1]
    y_frac = (sample_eye_y - ctx.ymin) / max(ctx.height, 1e-9)
    brow_frac = min(0.92, y_frac + 0.12)

    x_left, x_right, brow_band_y = body_span_at_fraction(ctx, brow_frac)

    def clamp_brow_x(x, margin):
        return clamp_inside_span(x, x_left, x_right, margin)

    # -----------------------------
    # Normal separate brows
    # -----------------------------

    if brow_type == "brows":
        for ex, ey, er in drawn_eye_positions:
            brow_y = ey + brow_lift
            brow_half_width = 1.05 * er

            x0 = clamp_brow_x(ex - brow_half_width, 0.20 * er)
            x1 = clamp_brow_x(ex + brow_half_width, 0.20 * er)

            # Slight angry/expressive slant.
            if ex < ctx.cx:
                y0 = brow_y + 0.12 * er
                y1 = brow_y + 0.28 * er
            elif ex > ctx.cx:
                y0 = brow_y + 0.28 * er
                y1 = brow_y + 0.12 * er
            else:
                y0 = brow_y + 0.18 * er
                y1 = brow_y + 0.18 * er

            if curly:
                arc = Arc(
                    (ex, brow_y + 0.12 * er),
                    2.2 * er,
                    0.9 * er,
                    theta1=20,
                    theta2=160,
                    color=hair_color,
                    linewidth=2.2,
                    zorder=10
                )
                ctx.ax.add_patch(arc)
            else:
                ctx.ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=hair_color,
                    linewidth=2.4,
                    solid_capstyle="round",
                    zorder=10
                )

    # -----------------------------
    # Eyehair: little lashes/tufts over each eye
    # -----------------------------

    elif brow_type == "eyehair":
        for ex, ey, er in drawn_eye_positions:
            n_hairs = 5
            spread = 1.45 * er

            for k in range(n_hairs):
                t = 0 if n_hairs == 1 else k / (n_hairs - 1)
                hx = ex - spread / 2 + t * spread
                hy = ey + 1.15 * er

                hx = clamp_brow_x(hx, 0.15 * er)

                if curly:
                    arc = Arc(
                        (hx, hy + 0.32 * er),
                        0.50 * er,
                        0.55 * er,
                        theta1=0,
                        theta2=300,
                        color=hair_color,
                        linewidth=1.4,
                        zorder=10
                    )
                    ctx.ax.add_patch(arc)
                else:
                    lean = ctx.py_rng.uniform(-0.25, 0.25) * er
                    length = ctx.py_rng.uniform(0.65, 1.05) * er

                    ctx.ax.plot(
                        [hx, hx + lean],
                        [hy, hy + length],
                        color=hair_color,
                        linewidth=1.3,
                        solid_capstyle="round",
                        zorder=10
                    )

    # -----------------------------
    # Unibrow: one connected brow across eye region
    # -----------------------------

    elif brow_type == "unibrows":
        eye_xs = [x for x, y, r in drawn_eye_positions]
        eye_ys = [y for x, y, r in drawn_eye_positions]

        center_y = sum(eye_ys) / len(eye_ys)
        brow_y = center_y + brow_lift

        if len(drawn_eye_positions) == 1:
            # Single-eye unibrow becomes a thick brow over the cyclops eye.
            ex, ey, er = drawn_eye_positions[0]
            x0 = ex - 1.65 * er
            x1 = ex + 1.65 * er
        else:
            left_eye = min(eye_xs)
            right_eye = max(eye_xs)
            x0 = left_eye - 1.1 * avg_eye_radius
            x1 = right_eye + 1.1 * avg_eye_radius

        x0 = clamp_brow_x(x0, 0.2 * avg_eye_radius)
        x1 = clamp_brow_x(x1, 0.2 * avg_eye_radius)

        if curly:
            n_curls = 5
            for k in range(n_curls):
                t = 0 if n_curls == 1 else k / (n_curls - 1)
                cx = x0 + t * (x1 - x0)

                arc = Arc(
                    (cx, brow_y),
                    0.55 * avg_eye_radius,
                    0.50 * avg_eye_radius,
                    theta1=0,
                    theta2=320,
                    color=hair_color,
                    linewidth=2.0,
                    zorder=10
                )
                ctx.ax.add_patch(arc)
        else:
            xs = np.linspace(x0, x1, 80)
            ys = brow_y + 0.10 * avg_eye_radius * np.sin(
                np.linspace(0, 2 * np.pi, 80)
            )

            ctx.ax.plot(
                xs,
                ys,
                color=hair_color,
                linewidth=3.0,
                solid_capstyle="round",
                zorder=10
            )

# -----------------------------
# DRAWING NOSE FOR ROCK
# -----------------------------

def get_nose_layout(ctx, drawn_eye_positions=None):
    """
    Compute a nose position between eyes and mouth.

    Goals:
    - Stay below eyes
    - Stay above mouth
    - Stay inside the body span
    - Leave mouth/facial-hair space clean
    """

    shape = ctx.v.get("shape", "circle")
    nose_type = ctx.v.get("noses", "n/a")

    # Estimate mouth position without drawing it.
    mouth_cx, mouth_cy, mouth_w, mouth_h, mouth_x_left, mouth_x_right = get_mouth_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions
    )

    # Estimate eye lower bound.
    if drawn_eye_positions is not None and len(drawn_eye_positions) > 0:
        avg_eye_y = sum(y for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)
        avg_eye_r = sum(r for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)

        eye_bottom = avg_eye_y - 1.10 * avg_eye_r
        eye_center_y = avg_eye_y
    else:
        avg_eye_r = 0.07 * ctx.unit
        eye_center_y = ctx.ymin + 0.58 * ctx.height
        eye_bottom = eye_center_y - avg_eye_r

    # Estimate mouth top. Smiles/chips/smeagol occupy a little more vertical room.
    if ctx.v.get("mouths", "n/a") in ["smile", "chip", "smeagol"]:
        mouth_top = mouth_cy + 0.35 * mouth_h
    else:
        mouth_top = mouth_cy + 0.15 * mouth_h

    # Available vertical lane between eyes and mouth.
    top_limit = eye_bottom - 0.08 * ctx.unit
    bottom_limit = mouth_top + 0.10 * ctx.unit

    # If there is room, place nose in the lane.
    if top_limit > bottom_limit:
        nose_y = 0.56 * top_limit + 0.44 * bottom_limit
    else:
        # Emergency fallback: squeeze nose around body mid-face.
        nose_y = ctx.ymin + 0.47 * ctx.height

    # Shape-specific correction.
    # Triangles get narrow near the top, so nose sits lower.
    if shape == "triangle":
        nose_y = min(nose_y, ctx.ymin + 0.49 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.35 * ctx.height)
    elif shape == "oblong":
        nose_y = min(nose_y, ctx.ymin + 0.55 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.38 * ctx.height)
    else:
        nose_y = min(nose_y, ctx.ymin + 0.56 * ctx.height)
        nose_y = max(nose_y, ctx.ymin + 0.36 * ctx.height)

    # Get body width at chosen nose height.
    y_frac = (nose_y - ctx.ymin) / max(ctx.height, 1e-9)
    x_left, x_right, nose_y = body_span_at_fraction(ctx, y_frac)

    local_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Nose base scale. Different nose styles get different sizes.
    base_w = 0.13 * local_width
    base_h = 0.10 * ctx.height

    if nose_type == "nub":
        nw = 0.85 * base_w
        nh = 0.85 * base_h
    elif nose_type == "honk":
        nw = 1.55 * base_w
        nh = 1.15 * base_h
    elif nose_type == "holes":
        nw = 1.10 * base_w
        nh = 0.80 * base_h
    elif nose_type == "concave":
        nw = 1.65 * base_w
        nh = 1.00 * base_h
    else:
        nw = base_w
        nh = base_h

    # Clamp to sane limits.
    nw = max(0.08 * ctx.unit, min(nw, 0.30 * local_width))
    nh = max(0.05 * ctx.unit, min(nh, 0.18 * ctx.height))

    return {
        "type": nose_type,
        "center": (center_x, nose_y),
        "width": nw,
        "height": nh,
        "x_left": x_left,
        "x_right": x_right,
        "mouth_layout": {
            "center": (mouth_cx, mouth_cy),
            "width": mouth_w,
            "height": mouth_h,
            "x_left": mouth_x_left,
            "x_right": mouth_x_right,
        }
    }

def draw_nose(ctx, drawn_eye_positions=None):
    """
    Draw noses with ctx-based placement.

    Supported:
    - nub
    - honk
    - holes
    - concave
    """

    nose_type = ctx.v.get("noses", "n/a")

    if nose_type == "n/a":
        return None

    layout = get_nose_layout(ctx, drawn_eye_positions=drawn_eye_positions)

    cx, cy = layout["center"]
    nw = layout["width"]
    nh = layout["height"]

    # Use body color for protruding nose types.
    nose_fill = ctx.body_color

    # -----------------------------
    # Nub: small round bump
    # -----------------------------
    if nose_type == "nub":
        nose = Circle(
            (cx, cy),
            radius=0.50 * min(nw, nh),
            facecolor=nose_fill,
            edgecolor="black",
            linewidth=1.1,
            zorder=10
        )
        nose.set_clip_path(ctx.body)
        ctx.ax.add_patch(nose)

        # Tiny highlight
        ctx.ax.add_patch(
            Circle(
                (cx - 0.15 * nw, cy + 0.12 * nh),
                radius=0.10 * min(nw, nh),
                facecolor="white",
                edgecolor="none",
                alpha=0.35,
                zorder=11
            )
        )

    # -----------------------------
    # Honk: big goofy oval nose
    # -----------------------------
    elif nose_type == "honk":
        nose = Ellipse(
            (cx, cy),
            width=nw,
            height=nh,
            facecolor=nose_fill,
            edgecolor="black",
            linewidth=1.2,
            zorder=10
        )
        nose.set_clip_path(ctx.body)
        ctx.ax.add_patch(nose)

        # Nostril dot
        ctx.ax.add_patch(
            Circle(
                (cx + 0.20 * nw, cy - 0.02 * nh),
                radius=0.08 * min(nw, nh),
                facecolor="black",
                edgecolor="none",
                zorder=11
            )
        )

    # -----------------------------
    # Holes: two nostril holes only
    # -----------------------------
    elif nose_type == "holes":
        hole_r = 0.17 * min(nw, nh)

        for side in [-1, 1]:
            hole = Circle(
                (cx + side * 0.25 * nw, cy),
                radius=hole_r,
                facecolor="black",
                edgecolor="none",
                zorder=10
            )
            hole.set_clip_path(ctx.body)
            ctx.ax.add_patch(hole)

    # -----------------------------
    # Concave: inward curved nose/notch
    # -----------------------------
    elif nose_type == "concave":
        # A downward-facing shallow arc reads like a dent.
        arc = Arc(
            (cx, cy + 0.08 * nh),
            width=nw,
            height=nh,
            theta1=200,
            theta2=340,
            color="black",
            linewidth=1.8,
            zorder=10
        )
        arc.set_clip_path(ctx.body)
        ctx.ax.add_patch(arc)

        # Small shadow mark under it
        ctx.ax.plot(
            [cx - 0.20 * nw, cx + 0.20 * nw],
            [cy - 0.18 * nh, cy - 0.12 * nh],
            color="black",
            alpha=0.25,
            linewidth=1.0,
            zorder=9,
            clip_path=ctx.body
        )

    return layout

# -----------------------------
# DRAWING MOUTH FOR ROCK
# -----------------------------

def get_mouth_layout(ctx, drawn_eye_positions=None):
    """
    Compute a shape-aware mouth position and size.

    Returns:
    center_x, center_y, mouth_width, mouth_height, x_left, x_right
    """

    shape = ctx.v.get("shape", "circle")

    # If eyes exist, place mouth below them.
    if drawn_eye_positions is not None and len(drawn_eye_positions) > 0:
        avg_eye_y = sum(y for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)
        avg_eye_r = sum(r for x, y, r in drawn_eye_positions) / len(drawn_eye_positions)

        # Mouth sits below eyes.
        desired_y = avg_eye_y - 2.25 * avg_eye_r

        # Convert desired y to body fraction.
        y_frac = (desired_y - ctx.ymin) / max(ctx.height, 1e-9)

    else:
        # Fallback if the rock has no eyes.
        if shape == "triangle":
            y_frac = 0.34
        elif shape == "oblong":
            y_frac = 0.42
        else:
            y_frac = 0.38

    # Keep mouth inside a reasonable lower-face band.
    if shape == "triangle":
        y_frac = max(0.25, min(0.48, y_frac))
    else:
        y_frac = max(0.26, min(0.48, y_frac))

    x_left, x_right, y = body_span_at_fraction(ctx, y_frac)

    local_width = x_right - x_left
    center_x = 0.5 * (x_left + x_right)

    # Mouth size scales with local body span.
    mouth_width = 0.36 * local_width
    mouth_height = 0.13 * ctx.height

    # Avoid absurdly small or giant mouths.
    mouth_width = max(0.18 * ctx.unit, min(mouth_width, 0.58 * ctx.unit))
    mouth_height = max(0.06 * ctx.unit, min(mouth_height, 0.18 * ctx.unit))

    return center_x, y, mouth_width, mouth_height, x_left, x_right

def draw_mouth(ctx, drawn_eye_positions=None):
    """
    Shape-aware mouth drawer, style v2.

    mouth:
        simple neutral line

    smile:
        large orange-slice toothy grin

    chip:
        Patrick-style curved smile with one big off-center tooth

    smeagol:
        cursed jagged-tooth grin
    """

    mouth_type = ctx.v.get("mouths", "n/a")

    if mouth_type == "n/a":
        return None

    cx, cy, mw, mh, x_left, x_right = get_mouth_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions
    )

    # Make cartoon mouths more readable.
    if mouth_type in ["smile", "chip", "smeagol"]:
        mw *= 1.25
        mh *= 1.35

    line_color = "black"

    # Keep mouth width inside the body span.
    max_width = 0.86 * (x_right - x_left)
    mw = min(mw, max_width)

    # -----------------------------
    # Basic neutral mouth
    # -----------------------------
    if mouth_type == "mouth":
        x0 = cx - 0.50 * mw
        x1 = cx + 0.50 * mw

        x0 = clamp_inside_span(x0, x_left, x_right, 0.04 * ctx.unit)
        x1 = clamp_inside_span(x1, x_left, x_right, 0.04 * ctx.unit)

        ctx.ax.plot(
            [x0, x1],
            [cy, cy],
            color=line_color,
            linewidth=2.1,
            solid_capstyle="round",
            zorder=10
        )

    # -----------------------------
    # Smile: orange-slice toothy grin
    # -----------------------------
    elif mouth_type == "smile":
        left = cx - 0.50 * mw
        right = cx + 0.50 * mw
        top_y = cy + 0.12 * mh
        bottom_y = cy - 0.58 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Orange-slice / crescent mouth shape.
        verts = [
            (left, top_y),
            (right, top_y),
            (right - 0.08 * mw, cy - 0.40 * mh),
            (cx, bottom_y),
            (left + 0.08 * mw, cy - 0.40 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        mouth_patch = PathPatch(
            Path(verts, codes),
            facecolor="white",
            edgecolor="black",
            linewidth=2.2,
            zorder=10
        )

        mouth_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(mouth_patch)

        # Tooth grid clipped inside the smile.
        n_vertical = 4

        for i in range(1, n_vertical):
            tx = left + i * (right - left) / n_vertical

            ctx.ax.plot(
                [tx, tx],
                [top_y, bottom_y + 0.10 * mh],
                color="black",
                linewidth=1.1,
                zorder=11,
                clip_path=mouth_patch
            )

        # One curved-ish horizontal tooth separator.
        xs = np.linspace(left + 0.05 * mw, right - 0.05 * mw, 80)
        ys = cy - 0.22 * mh + 0.04 * mh * np.cos(np.linspace(0, np.pi, 80))

        ctx.ax.plot(
            xs,
            ys,
            color="black",
            linewidth=1.1,
            zorder=11,
            clip_path=mouth_patch
        )

    # -----------------------------
    # Chip: Patrick smile with one big off-center tooth
    # -----------------------------
    elif mouth_type == "chip":
        left = cx - 0.46 * mw
        right = cx + 0.46 * mw
        top_y = cy + 0.05 * mh
        bottom_y = cy - 0.48 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Big bowl smile.
        verts = [
            (left, top_y),
            (right, top_y),
            (right, cy - 0.38 * mh),
            (cx, bottom_y),
            (left, cy - 0.38 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        smile_patch = PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="black",
            linewidth=2.2,
            zorder=10
        )

        smile_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(smile_patch)

        # One big off-center tooth hanging from the top line.
        tooth_cx = cx + 0.13 * mw
        tooth_top = top_y - 0.02 * mh
        tooth_bottom = cy - 0.30 * mh
        tooth_half_w = 0.105 * mw

        tooth = Polygon(
            [
                [tooth_cx - tooth_half_w, tooth_top],
                [tooth_cx + tooth_half_w, tooth_top],
                [tooth_cx + 0.72 * tooth_half_w, tooth_bottom],
                [tooth_cx - 0.72 * tooth_half_w, tooth_bottom],
            ],
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=11
        )

        tooth.set_clip_path(ctx.body)
        ctx.ax.add_patch(tooth)

    # -----------------------------
    # Smeagol: jagged chip smile with 2-3 teeth
    # -----------------------------
    elif mouth_type == "smeagol":
        left = cx - 0.50 * mw
        right = cx + 0.48 * mw
        top_y = cy + 0.04 * mh
        bottom_y = cy - 0.45 * mh

        left = clamp_inside_span(left, x_left, x_right, 0.04 * ctx.unit)
        right = clamp_inside_span(right, x_left, x_right, 0.04 * ctx.unit)

        # Uneven bowl-like smile.
        verts = [
            (left, top_y),
            (right, top_y + 0.03 * mh),
            (right - 0.02 * mw, cy - 0.34 * mh),
            (cx + 0.08 * mw, bottom_y),
            (left + 0.04 * mw, cy - 0.33 * mh),
            (left, top_y),
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ]

        mouth_patch = PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="black",
            linewidth=2.0,
            zorder=10
        )

        mouth_patch.set_clip_path(ctx.body)
        ctx.ax.add_patch(mouth_patch)

        # Jagged teeth.
        tooth_count = 2 + (ctx.rock.id % 2)

        tooth_centers = np.linspace(
            cx - 0.18 * mw,
            cx + 0.22 * mw,
            tooth_count
        )

        for i, tx in enumerate(tooth_centers):
            tooth_height = (0.27 + 0.08 * ((i + ctx.rock.id) % 2)) * mh
            tooth_width = 0.075 * mw

            tooth = Polygon(
                [
                    [tx - tooth_width, top_y - 0.02 * mh],
                    [tx + tooth_width, top_y - 0.02 * mh],
                    [tx + ctx.py_rng.uniform(-0.04, 0.04) * mw, top_y - tooth_height],
                ],
                closed=True,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=11
            )

            tooth.set_clip_path(ctx.body)
            ctx.ax.add_patch(tooth)

        # Little cursed wrinkles below mouth.
        for i in range(2):
            yy = cy - (0.50 + 0.20 * i) * mh
            ctx.ax.plot(
                [cx - 0.22 * mw, cx + 0.18 * mw],
                [yy, yy + 0.04 * mh * ((-1) ** i)],
                color="black",
                linewidth=0.9,
                alpha=0.45,
                zorder=9,
                clip_path=ctx.body
            )

    return {
        "type": mouth_type,
        "center": (cx, cy),
        "width": mw,
        "height": mh
    }

# -----------------------------
# DRAWING FACIAL HAIR FOR ROCK
# -----------------------------

def get_facial_hair_layout(ctx, drawn_eye_positions=None, nose_info=None, mouth_info=None):
    """
    Compute safe layout bands for facial hair.

    We use:
    - nose info
    - mouth info
    - body spans at relevant heights

    So facial hair avoids overlapping the nose and mouth too aggressively.
    """
    fh_type = ctx.v.get("facial_hair", "n/a")

    if fh_type == "n/a":
        return None

    # If not supplied, estimate them.
    if mouth_info is None:
        mcx, mcy, mw, mh, mxl, mxr = get_mouth_layout(ctx, drawn_eye_positions)
        mouth_info = {
            "center": (mcx, mcy),
            "width": mw,
            "height": mh,
            "x_left": mxl,
            "x_right": mxr
        }

    if nose_info is None:
        nose_info = get_nose_layout(ctx, drawn_eye_positions)

    mouth_cx, mouth_cy = mouth_info["center"]
    mouth_w = mouth_info["width"]
    mouth_h = mouth_info["height"]

    nose_cx, nose_cy = nose_info["center"]
    nose_w = nose_info["width"]
    nose_h = nose_info["height"]

    # Useful vertical reference levels
    nose_bottom = nose_cy - 0.50 * nose_h
    mouth_top = mouth_cy + 0.18 * mouth_h
    mouth_bottom = mouth_cy - 0.18 * mouth_h

    # Mustache / stubble band between nose and mouth
    upper_band_y = 0.55 * nose_bottom + 0.45 * mouth_top

    # Patch / goatee / beard band below mouth
    lower_band_y = mouth_cy - 0.35 * mouth_h
    chin_band_y = mouth_cy - 0.75 * mouth_h

    # Clamp body-safe heights
    def clamp_y(y):
        return max(ctx.ymin + 0.08 * ctx.height, min(ctx.ymax - 0.08 * ctx.height, y))

    upper_band_y = clamp_y(upper_band_y)
    lower_band_y = clamp_y(lower_band_y)
    chin_band_y = clamp_y(chin_band_y)

    # Get local spans at these bands
    def span_at_y(y):
        y_frac = (y - ctx.ymin) / max(ctx.height, 1e-9)
        x_left, x_right, yy = body_span_at_fraction(ctx, y_frac)
        return x_left, x_right, yy

    upper_left, upper_right, upper_y = span_at_y(upper_band_y)
    lower_left, lower_right, lower_y = span_at_y(lower_band_y)
    chin_left, chin_right, chin_y = span_at_y(chin_band_y)

    return {
        "type": fh_type,
        "mouth_info": mouth_info,
        "nose_info": nose_info,

        "upper_band": {
            "y": upper_y,
            "x_left": upper_left,
            "x_right": upper_right
        },
        "lower_band": {
            "y": lower_y,
            "x_left": lower_left,
            "x_right": lower_right
        },
        "chin_band": {
            "y": chin_y,
            "x_left": chin_left,
            "x_right": chin_right
        }
    }

def draw_facial_hair_curls(
    ctx,
    hair_color,
    mouth_cx,
    mouth_cy,
    mouth_w,
    mouth_h,
    fh_type,
    zorder=62
):
    """
    Add curl marks over facial hair if hair_texture is curly.

    Works best for beard/goatee/curly styles.
    """
    if not rock_texture_is_curly(ctx):
        return []

    if fh_type in ["n/a", "peach_fuzz"]:
        return []

    # Small facial hair styles get fewer curls.
    if fh_type in ["beard"]:
        x_min = mouth_cx - 0.75 * mouth_w
        x_max = mouth_cx + 0.75 * mouth_w
        y_min = mouth_cy - 2.05 * mouth_h
        y_max = mouth_cy + 0.15 * mouth_h
        n_curls = 9
        curl_scale = 0.05

    elif fh_type in ["goatee"]:
        x_min = mouth_cx - 0.62 * mouth_w
        x_max = mouth_cx + 0.62 * mouth_w
        y_min = mouth_cy - 1.20 * mouth_h
        y_max = mouth_cy + 0.10 * mouth_h
        n_curls = 6
        curl_scale = 0.05

    elif fh_type in ["curly_mustache"]:
        x_min = mouth_cx - 0.65 * mouth_w
        x_max = mouth_cx + 0.65 * mouth_w
        y_min = mouth_cy - 0.15 * mouth_h
        y_max = mouth_cy + 0.35 * mouth_h
        n_curls = 5
        curl_scale = 0.05

    else:
        x_min = mouth_cx - 0.50 * mouth_w
        x_max = mouth_cx + 0.50 * mouth_w
        y_min = mouth_cy - 0.65 * mouth_h
        y_max = mouth_cy + 0.15 * mouth_h
        n_curls = 4
        curl_scale = 0.05

    return draw_curly_overlay_in_box(
        ctx,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        hair_color=hair_color,
        n_curls=n_curls,
        curl_scale=curl_scale,
        zorder=zorder,
        salt=f"facial_curls_{fh_type}"
    )

def draw_facial_hair(ctx, rock, v, drawn_eye_positions=None, nose_info=None, mouth_info=None):
    """
    Draw facial-hair styles:
    - goatee
    - beard
    - pedo
    - curl
    - chapman
    - sol
    """
    fh_type = ctx.v.get("facial_hair", "n/a")

    if fh_type == "n/a":
        return None

    layout = get_facial_hair_layout(
        ctx,
        drawn_eye_positions=drawn_eye_positions,
        nose_info=nose_info,
        mouth_info=mouth_info
    )

    mouth_info = layout["mouth_info"]
    nose_info = layout["nose_info"]

    mouth_cx, mouth_cy = mouth_info["center"]
    mouth_w = mouth_info["width"]
    mouth_h = mouth_info["height"]

    nose_cx, nose_cy = nose_info["center"]
    nose_w = nose_info["width"]
    nose_h = nose_info["height"]

    hair_color = get_hair_color_from_alleles(
        ctx.v.get("hair_color_alleles", [ctx.v.get("hair_color", "black")])
    )

    gender = v["gender"]

    z = 10

    # --------------------------------------------------
    # 0) PEACH FUZZ — soft small fuzz for females
    # --------------------------------------------------
        # --------------------------------------------------
    # 0) PEACH FUZZ — tiny fuzz spikes at ~7:30 and ~4:30
    # --------------------------------------------------
    if fh_type == "peach fuzz":
        fh_z = z - 1

        # Keep it subtle.
        fuzz_color = hair_color

        # Put the fuzz clusters slightly below mouth center,
        # out toward the lower-left and lower-right "cheek" areas.
        cluster_y = mouth_cy - 0.10 * mouth_h
        y_frac = (cluster_y - ctx.ymin) / max(ctx.height, 1e-9)
        x_left, x_right, cluster_y = body_span_at_fraction(ctx, y_frac)
        local_span = x_right - x_left

        # Anchor the fuzz on the face, not at the extreme edge.
        left_anchor = (
            x_left + 0.28 * local_span,
            cluster_y - 0.01 * ctx.unit
        )
        right_anchor = (
            x_right - 0.28 * local_span,
            cluster_y - 0.01 * ctx.unit
        )

        # 7:30 and 4:30 directions
        left_angle = math.radians(225)   # down-left
        right_angle = math.radians(-45)  # down-right

        cluster_specs = [
            (left_anchor, left_angle),
            (right_anchor, right_angle),
        ]

        for (ax0, ay0), base_angle in cluster_specs:
            n_spikes = 5

            for i in range(n_spikes):
                # Spread the spikes in a tiny fan
                ang = base_angle + np.linspace(-0.28, 0.28, n_spikes)[i]

                # Slightly offset each spike base so the cluster looks natural
                perp = base_angle + math.pi / 2
                base_offset = (i - (n_spikes - 1) / 2) * 0.018 * ctx.unit

                x0 = ax0 + base_offset * math.cos(perp)
                y0 = ay0 + base_offset * math.sin(perp)

                # Small lengths, like subtle fuzz
                L = (0.055 + 0.010 * i) * ctx.unit

                x1 = x0 + L * math.cos(ang)
                y1 = y0 + L * math.sin(ang)

                line, = ctx.ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=fuzz_color,
                    linewidth=1.0,
                    alpha=0.42,
                    zorder=fh_z,
                    solid_capstyle="round"
                )

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 1) GOATEE — Homer-Simpson-like muzzle/chin beard
    # --------------------------------------------------
    elif fh_type == "goatee":
        # Put facial hair behind the mouth
        fh_z = z - 2

        # Build a filled "muzzle" around the mouth.
        # Top sits a little above mouth center so the mouth can cut across it.
        top_y = mouth_cy + 0.22 * mouth_h
        side_y = mouth_cy - 0.18 * mouth_h
        bottom_y = mouth_cy - 1.05 * mouth_h

        goatee_pts = [
            [mouth_cx - 0.55 * mouth_w, top_y],
            [mouth_cx + 0.55 * mouth_w, top_y],

            [mouth_cx + 0.68 * mouth_w, side_y],
            [mouth_cx + 0.40 * mouth_w, bottom_y],

            [mouth_cx,                 bottom_y - 0.18 * mouth_h],

            [mouth_cx - 0.40 * mouth_w, bottom_y],
            [mouth_cx - 0.68 * mouth_w, side_y],
        ]

        goatee = Polygon(
            goatee_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=fh_z,
            joinstyle="round"
        )
        ctx.ax.add_patch(goatee)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 2) BEARD — Homer muzzle + draping lower beard
    # --------------------------------------------------
    elif fh_type == "beard":
        fh_z = z - 2

        # Upper part: same Homer-like face beard
        top_y = mouth_cy + 0.24 * mouth_h
        side_y = mouth_cy - 0.16 * mouth_h
        chin_y = mouth_cy - 0.95 * mouth_h

        # Lower drape extends off the face more like a beard wedge
        beard_tip_y = mouth_cy - 2.05 * mouth_h

        beard_pts = [
            [mouth_cx - 0.58 * mouth_w, top_y],
            [mouth_cx + 0.58 * mouth_w, top_y],

            [mouth_cx + 0.74 * mouth_w, side_y],
            [mouth_cx + 0.54 * mouth_w, chin_y],
            [mouth_cx + 0.24 * mouth_w, mouth_cy - 1.45 * mouth_h],

            [mouth_cx, beard_tip_y],

            [mouth_cx - 0.24 * mouth_w, mouth_cy - 1.45 * mouth_h],
            [mouth_cx - 0.54 * mouth_w, chin_y],
            [mouth_cx - 0.74 * mouth_w, side_y],
        ]

        beard = Polygon(
            beard_pts,
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=fh_z,
            joinstyle="round"
        )
        ctx.ax.add_patch(beard)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout
    # --------------------------------------------------
    # 3) PEDO — two small marks
    # --------------------------------------------------
    elif fh_type == "pedo":
        y = layout["upper_band"]["y"]
        for side in [-1, 1]:
            dot = Ellipse(
                (mouth_cx + side * 0.22 * mouth_w, y),
                width=0.4 * mouth_w,
                height=0.25 * mouth_h,
                angle=side * -10,
                facecolor=hair_color,
                edgecolor="black",
                linewidth=0.8,
                zorder=z
            )
            ctx.ax.add_patch(dot)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 4) CURLY — two-lobed
    # --------------------------------------------------
    elif fh_type == "curly":
        y = layout["upper_band"]["y"]

        for side in [-1, 1]:
            x0 = mouth_cx + side * 0.05 * mouth_w
            x1 = mouth_cx + side * 0.58 * mouth_w

            verts = [
                (x0, y),
                (mouth_cx + side * 0.18 * mouth_w, y - 0.33 * mouth_h),
                (mouth_cx + side * 0.42 * mouth_w, y - 0.12 * mouth_h),
                (x1, y + 0.35 * mouth_h),
            ]

            patch = PathPatch(
                Path(
                    verts,
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                ),
                facecolor="none",
                edgecolor=hair_color,
                linewidth=3.0,
                zorder=z,
                capstyle="round"
            )
            ctx.ax.add_patch(patch)

            """
            # little curl tip
            curl = Arc(
                (x1, y + 0.2 * mouth_h),
                width=0.18 * mouth_w,
                height=0.20 * mouth_h,
                theta1=210 if side == -1 else -30,
                theta2=360 if side == -1 else 120,
                color=hair_color,
                linewidth=2.4,
                zorder=z
            )
            ctx.ax.add_patch(curl)
            """

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 5) CHAPMAN — little soul patch
    # --------------------------------------------------
    elif fh_type == "chapman":
        y = layout["upper_band"]["y"] - 0.05 * mouth_h

        patch = Polygon(
            [
                [mouth_cx - 0.18 * mouth_w, y + 0.05 * mouth_h],
                [mouth_cx + 0.18 * mouth_w, y + 0.05 * mouth_h],
                [mouth_cx + 0.22 * mouth_w, y - 0.18 * mouth_h],
                [mouth_cx - 0.22 * mouth_w, y - 0.18 * mouth_h],
            ],
            closed=True,
            facecolor="none",
            edgecolor=hair_color,
            linewidth=2.6,
            zorder=z,
            joinstyle="round"
        )
        ctx.ax.add_patch(patch)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    # --------------------------------------------------
    # 6) SOL — small chin tuft
    # --------------------------------------------------
    elif fh_type == "sol":
        y = layout["chin_band"]["y"]

        goatee = Polygon(
            [
                [mouth_cx - 0.20 * mouth_w, y + 0.20 * mouth_h],
                [mouth_cx + 0.20 * mouth_w, y + 0.20 * mouth_h],
                [mouth_cx + 0.12 * mouth_w, y - 0.44 * mouth_h],
                [mouth_cx,                 y - 0.68 * mouth_h],
                [mouth_cx - 0.12 * mouth_w, y - 0.44 * mouth_h],
            ],
            closed=True,
            facecolor=hair_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=z,
            joinstyle="round"
        )
        ctx.ax.add_patch(goatee)

        draw_facial_hair_curls(
            ctx,
            hair_color=hair_color,
            mouth_cx=mouth_cx,
            mouth_cy=mouth_cy,
            mouth_w=mouth_w,
            mouth_h=mouth_h,
            fh_type=fh_type,
            zorder=z + 3
        )

        return layout

    return layout

# -----------------------------
# DRAWING A FULL ROCK
# -----------------------------

def draw_rock(rock, ax=None, show_genes=False, normalize_size=True):
    """
    Trait-based rock renderer.

    Uses the new categorical/co-dominant visual phenotype.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    rng = np.random.default_rng(1_000_000 + rock.id)
    py_rng = random.Random(200_000 + rock.id)

    v = get_visual_phenotype(rock)

    shape_name = v.get("shape", "circle")
    size_name = v.get("size", "medium")
    color_name = v.get("color", "brown")
    color_alleles = v.get("color_alleles", [color_name])

    body_points, s = make_body_points(shape_name, size_name, rng)
    body_color = get_body_color_from_alleles(color_alleles)

    body = Polygon(
        body_points,
        closed=True,
        facecolor=body_color,
        edgecolor="black",
        linewidth=2.0,
        zorder=1
    )

    ctx = RockRenderContext(
        ax=ax,
        rock=rock,
        v=v,
        rng=rng,
        py_rng=py_rng,
        body=body,
        body_points=body_points,
        s=s,
        body_color=body_color
        )

    draw_wings(ctx)
    draw_fuzz(ctx)
    draw_halo(ctx)

    draw_stones(ctx)
    draw_tail(ctx)

    draw_horns(ctx)

    ax.add_patch(body)
    draw_patchwork(ax, body, color_alleles, s, rng)
    draw_hair(ctx, rock, v)

    draw_ears(ctx)

    draw_wrinkles(ctx)
    draw_freckles(ctx)

    draw_arms(ctx)
    draw_crown(ctx)

    drawn_eye_positions = draw_eyes(ctx)
    draw_brows(ctx, drawn_eye_positions)
    nose_info = draw_nose(ctx, drawn_eye_positions)
    mouth_info = draw_mouth(ctx, drawn_eye_positions)
    draw_facial_hair(
      ctx,
      rock,
      v,
      drawn_eye_positions=drawn_eye_positions,
      nose_info=nose_info,
      mouth_info=mouth_info
      )

    # Temporary compatibility for any older code below this point.
    eye_positions = [(x, y) for x, y, r in drawn_eye_positions]

    # -----------------------------
    # Craisen overlay
    # -----------------------------

    if v.get("is_craisen", False):
        ax.plot([-0.75 * s, 0.75 * s], [-0.75 * s, 0.75 * s], color="crimson", linewidth=4, zorder=20)
        ax.plot([-0.75 * s, 0.75 * s], [0.75 * s, -0.75 * s], color="crimson", linewidth=4, zorder=20)
        ax.text(0, -1.25 * s, "CRAISEN", color="crimson", ha="center", va="center", fontsize=10, fontweight="bold")

    # -----------------------------
    # Labels / formatting
    # -----------------------------

    ax.set_title(f"{rock.name} #{rock.id}\nGen {rock.generation}")
    ax.set_aspect("equal")

    if normalize_size:
        # Portrait mode:
        # each rock fills its own frame.
        ax.set_xlim(-2.0 * s, 2.0 * s)
        ax.set_ylim(-1.65 * s, 1.75 * s)
    else:
        # Comparison mode:
        # every rock uses the same camera.
        # This makes small/large/giant visibly different.
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-2.55, 2.85)

    ax.axis("off")

    if show_genes:
        gene_text = "\n".join([f"{k}: {v}" for k, v in rock.genes.items()])
        ax.text(
            1.55 * s,
            1.35 * s,
            gene_text,
            fontsize=7,
            va="top",
            family="monospace"
        )

    return ax

# -----------------------------
# SHOWING SOME ROCK
# -----------------------------

def show_rocks(
    rock_items,
    rock_source=None,
    cols=6,
    figsize_per_rock=3.2,
    show_genes=False,
    show_traits=False,
    title=None,
    sort_by_generation=False,
    normalize_size=True
):
    """
    Display a grid of rocks.

    Accepts:
    - a dictionary of rocks: show_rocks(rocks)
    - a list of rock IDs: show_rocks([1, 2, 3], rock_source=rocks)
    - a list of Rock objects: show_rocks([rock1, rock2])
    - a dictionary of test rocks: show_rocks(test_rocks)

    Parameters
    ----------
    rock_items:
        Dict[int, Rock], list[int], tuple[int], list[Rock], or tuple[Rock]

    rock_source:
        Optional dictionary used when rock_items is a list of IDs.
        If None, the function tries to use the global `rocks`.

    cols:
        Number of columns in the display grid.

    figsize_per_rock:
        Size multiplier for each rock subplot.

    show_genes:
        Passes show_genes=True into draw_rock.

    show_traits:
        Adds a compact trait label under each rock.

    title:
        Optional figure title.

    sort_by_generation:
        If True, sorts rocks by generation, then ID.
    """

    # -----------------------------
    # Resolve input into Rock objects
    # -----------------------------

    if isinstance(rock_items, dict):
        rock_list = list(rock_items.values())

    else:
        rock_list = []

        for item in list(rock_items):
            if isinstance(item, Rock):
                rock_list.append(item)

            elif isinstance(item, int):
                source = rock_source

                if source is None:
                    try:
                        source = rocks
                    except NameError:
                        raise ValueError(
                            "You passed rock IDs, but no rock_source was provided "
                            "and no global `rocks` dictionary exists."
                        )

                if item not in source:
                    raise KeyError(f"Rock ID {item} was not found in the provided rock source.")

                rock_list.append(source[item])

            else:
                raise TypeError(
                    "show_rocks expects a dict of rocks, a list of Rock objects, "
                    "or a list of integer rock IDs."
                )

    if sort_by_generation:
        rock_list = sorted(rock_list, key=lambda r: (r.generation, r.id))

    n = len(rock_list)

    if n == 0:
        print("No rocks to show.")
        return None, None

    # -----------------------------
    # Create grid
    # -----------------------------

    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_rock, rows * figsize_per_rock)
    )

    axes = np.array(axes).reshape(-1)

    # Turn all axes off first.
    for ax in axes:
        ax.axis("off")

    # -----------------------------
    # Draw rocks
    # -----------------------------

    for ax, rock in zip(axes, rock_list):
        draw_rock(rock, ax=ax, show_genes=show_genes, normalize_size=normalize_size)

        pad_rock_axis(ax, pad_frac=PAD_FRAC)

        if show_traits:
            v = get_visual_phenotype(rock)

            trait_text = (
                f"{v.get('shape', 'n/a')} | {v.get('size', 'n/a')} | {v.get('color', 'n/a')}\n"
                f"eyes: {v.get('eyes', 'n/a')} | hair: {v.get('hair', 'n/a')} | {v.get('hair_color', 'n/a')}"
            )

            if v.get("is_craisen", False):
                trait_text += "\nCRAISEN"

            ax.text(
                0.5,
                -0.08,
                trait_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8
            )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

    return fig, axes

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

def rock_to_image_uri(rock, sprite_size=2.0, dpi=400):
    """
    Render a rock to a transparent PNG and return it as a base64 image URI
    that Plotly can place on the graph.
    """
    fig, ax = plt.subplots(figsize=(sprite_size, sprite_size), dpi=dpi)

    draw_rock(rock, ax=ax)

    pad_rock_axis(ax, pad_frac=PAD_FRAC)

    # Remove the title from the mini image.
    ax.set_title("")
    ax.axis("off")

    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi = dpi
    )
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

# -----------------------------
# COMPUTING LAYOUT FOR ROCK TREE
# -----------------------------

def compute_lineage_positions_christmas(
    rocks,
    x_gap=5,
    gen_gap=5,
    parent_pair_gap=3,
    min_node_gap=3,
    anti_overlap=True,
    layout_passes=4
):
    """
    Robust Christmas-tree lineage layout.

    Goals:
    - all rocks get positions
    - parents are pulled toward their children
    - missing/orphan parent links do not crash the tree
    - sold/dead/puffed/spore clones can remain visible
    - works even when weird generation relationships appear

    Returns:
        pos = {rock_id: (x, y)}
    """

    if rocks is None or len(rocks) == 0:
        return {}

    # Make sure IDs are normal ints when possible.
    rock_ids = list(rocks.keys())

    # --------------------------------------------------
    # Group rocks by generation
    # --------------------------------------------------
    by_generation = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        by_generation.setdefault(gen, []).append(rid)

    generations = sorted(by_generation.keys())

    # --------------------------------------------------
    # Initial centered layout within each generation
    # --------------------------------------------------
    pos_x = {}

    for gen in generations:
        ids = sorted(by_generation[gen])
        n = len(ids)

        if n == 1:
            pos_x[ids[0]] = 0.0
        else:
            start_x = -0.5 * (n - 1) * x_gap

            for i, rid in enumerate(ids):
                pos_x[rid] = start_x + i * x_gap

    # Ensure every rock has an x position.
    for rid in rock_ids:
        if rid not in pos_x:
            pos_x[rid] = 0.0

    # --------------------------------------------------
    # Pull parents toward their children, bottom-up
    # --------------------------------------------------
    for _ in range(layout_passes):
        desired_x = {rid: [] for rid in rock_ids}

        # For each child, request parent positions around child center.
        for child_id, child in rocks.items():
            parents = getattr(child, "parents", None)

            if parents is None:
                continue

            if len(parents) != 2:
                continue

            p1, p2 = parents

            # Defensive skip: parent might not exist in current tree dictionary.
            if p1 not in rocks or p2 not in rocks:
                continue

            if child_id not in pos_x:
                continue

            child_center_x = pos_x[child_id]

            # Make sure keys exist even if the old data is odd.
            desired_x.setdefault(p1, [])
            desired_x.setdefault(p2, [])

            desired_x[p1].append(child_center_x - parent_pair_gap / 2)
            desired_x[p2].append(child_center_x + parent_pair_gap / 2)

        # Update positions from desired child-centered positions.
        # Work from older generations first to keep the tree stable.
        for gen in generations:
            for rid in by_generation[gen]:
                if rid in desired_x and len(desired_x[rid]) > 0:
                    old_x = pos_x.get(rid, 0.0)
                    target_x = sum(desired_x[rid]) / len(desired_x[rid])

                    # Blend instead of snapping to reduce wild oscillations.
                    pos_x[rid] = 0.45 * old_x + 0.55 * target_x

        # --------------------------------------------------
        # Anti-overlap pass within each generation
        # --------------------------------------------------
        if anti_overlap:
            for gen in generations:
                ids = sorted(by_generation[gen], key=lambda rid: pos_x.get(rid, 0.0))

                if len(ids) <= 1:
                    continue

                # Left-to-right push
                for i in range(1, len(ids)):
                    prev_id = ids[i - 1]
                    curr_id = ids[i]

                    if pos_x[curr_id] - pos_x[prev_id] < min_node_gap:
                        pos_x[curr_id] = pos_x[prev_id] + min_node_gap

                # Recenter the generation around zero-ish
                mean_x = sum(pos_x[rid] for rid in ids) / len(ids)

                for rid in ids:
                    pos_x[rid] -= mean_x

    # --------------------------------------------------
    # Final positions
    # --------------------------------------------------
    pos = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        x = pos_x.get(rid, 0.0)
        y = -gen * gen_gap

        pos[rid] = (x, y)

    return pos

FAMILY_PALETTE = (
    px.colors.qualitative.Safe
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Bold
)

def family_color(parent_pair):
    """
    Deterministic color for a parent pair.
    Same parent pair -> same color every time.
    """
    key = f"{min(parent_pair)}-{max(parent_pair)}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return FAMILY_PALETTE[h % len(FAMILY_PALETTE)]

FAMILY_COLORS = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#9C755F",  # brown
    "#FF9DA7",  # pink
    "#BAB0AC",  # gray
]

FAMILY_DASHES = [
    "solid",
    "dash",
    "dot",
    "longdash",
    "dashdot",
]

def get_family_styles(rocks):
    """
    Assign a stable color/dash style to each parent pair.
    """
    families = []

    for child_id, rock in rocks.items():
        if rock.parents is not None:
            families.append(tuple(sorted(rock.parents)))

    families = sorted(set(families))

    style_map = {}

    for i, fam in enumerate(families):
        style_map[fam] = {
            "color": FAMILY_COLORS[i % len(FAMILY_COLORS)],
            "dash": FAMILY_DASHES[(i // len(FAMILY_COLORS)) % len(FAMILY_DASHES)]
        }

    return style_map

def build_family_segments(pos, parent_pair, child_ids):
    """
    Builds pedigree line segments for one parent pair and its displayed children.
    """
    p1, p2 = parent_pair

    if p1 not in pos or p2 not in pos:
        return [], []

    child_ids = [cid for cid in child_ids if cid in pos]

    if len(child_ids) == 0:
        return [], []

    child_ids = sorted(child_ids, key=lambda cid: pos[cid][0])

    x1, y1 = pos[p1]
    x2, y2 = pos[p2]

    child_xs = [pos[cid][0] for cid in child_ids]
    child_ys = [pos[cid][1] for cid in child_ids]

    child_y = child_ys[0]
    parent_y = min(y1, y2)

    parent_bar_y = parent_y - 0.48
    sibling_bar_y = child_y + 0.68

    parent_center_x = (x1 + x2) / 2
    child_center_x = sum(child_xs) / len(child_xs)

    line_segments_x = []
    line_segments_y = []

    def add_segment(xa, ya, xb, yb):
        line_segments_x.extend([xa, xb, None])
        line_segments_y.extend([ya, yb, None])

    # Parent drops.
    add_segment(x1, y1 - 0.45, x1, parent_bar_y)
    add_segment(x2, y2 - 0.45, x2, parent_bar_y)

    # Parent pair bar.
    add_segment(x1, parent_bar_y, x2, parent_bar_y)

    # Descent toward children.
    add_segment(parent_center_x, parent_bar_y, child_center_x, sibling_bar_y)

    # Sibling bar and child drops.
    if len(child_ids) > 1:
        add_segment(min(child_xs), sibling_bar_y, max(child_xs), sibling_bar_y)

        for cx in child_xs:
            add_segment(cx, sibling_bar_y, cx, child_y + 0.45)
    else:
        cx = child_xs[0]
        add_segment(child_center_x, sibling_bar_y, cx, child_y + 0.45)

    return line_segments_x, line_segments_y

SIZE_SCALE_MAP = {
    "medium": 1.00,
    "large": 1.22,
    "small": 0.78,
    "giant": 1.55,
    "missized": 1.10,
}

def get_rock_size_scale(rock):
    """
    Returns the expressed visual size scale for a rock.
    """
    v = get_visual_phenotype(rock)
    return SIZE_SCALE_MAP.get(v.get("size", "medium"), 1.0)

def get_gender_symbol(rock):
    """
    Return display symbol for rock gender.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "♂"

    return "♀"

def get_gender_color(rock):
    """
    Display color for gender symbol.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "royalblue"

    return "deeppink"

# -----------------------------
# GETTING HOVER VALUES FOR ROCK TREE
# -----------------------------

def clean_hover_value(value):
    """
    Make phenotype values readable in Plotly hover text.
    """
    if isinstance(value, float):
        return f"{value:.3g}"

    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)

    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())

    return str(value)

def format_full_phenotype_hover(rock):
    """
    Build a full phenotype printout for hover boxes.
    """
    v = get_visual_phenotype(rock)

    lines = []

    for key in sorted(v.keys()):
        value = clean_hover_value(v[key])
        lines.append(f"{key}: {value}")

    return "<br>".join(lines)

def format_selected_phenotype_hover(rock):
    v = get_visual_phenotype(rock)

    preferred_keys = [
        "gender",
        "shape",
        "size",
        "color",
        "eyes",
        "eye_color",
        "mouths",
        "noses",
        "arms",
        "wings",
        "horns",
        "halos",
        "ears",
        "hair",
        "hair_color",
        "facial_hair",
        "wrinkles",
        "fuzz",
        "freckles",
        "stones",
        "tails",
        "splitting",
    ]

    lines = []

    for key in preferred_keys:
        if key in v:
            lines.append(f"{key}: {clean_hover_value(v[key])}")

    return "<br>".join(lines)

# -----------------------------
# DRAWING DA ROCK TREE
# -----------------------------

def normalize_parent_pair_for_tree(parent_pair):
    """
    Convert a parent-pair-like value into a clean (p1, p2) tuple.

    Returns None if the value is empty, malformed, or not exactly two ids.

    Handles:
        None
        ()
        []
        (1, 2)
        ["1", "2"]
        {"a": 1, "b": 2}
        Rock objects with .id
    """
    if parent_pair is None:
        return None

    if isinstance(parent_pair, dict):
        raw_values = list(parent_pair.values())
    elif isinstance(parent_pair, (list, tuple, set)):
        raw_values = list(parent_pair)
    else:
        raw_values = [parent_pair]

    cleaned = []

    for value in raw_values:
        if value is None:
            continue

        if hasattr(value, "id"):
            value = value.id

        try:
            cleaned.append(int(value))
        except Exception:
            continue

    if len(cleaned) != 2:
        return None

    p1, p2 = cleaned

    if p1 == p2:
        return None

    return (p1, p2)

def draw_game_tree(
    game,
    selected_ids=None,
    x_gap=3.2,
    gen_gap=3.2,
    parent_pair_gap=1.7,
    rock_image_size=1.15,
    canvas_width=1800,
    canvas_height=1100,
    show_labels=True,
    show_sold=True,
    inactive_sold_opacity=0.55,
    show = False,
    highlight_breeding_queue=False
):
    """
    Draw the full game lineage tree.

    Features:
    - all rocks shown
    - sold rocks marked with green $
    - craisen rocks marked with red X
    - bred parents marked with gray circle if not sold/craisen
    - selected rocks highlighted
    """
    evaluate_all_rocks(game)

    selected_ids = set(selected_ids or [])

    rocks_dict = game.rocks

    if len(rocks_dict) == 0:
        print("No rocks to draw.")
        return None

    pos = compute_lineage_positions_christmas(
        rocks_dict,
        x_gap=x_gap,
        gen_gap=gen_gap,
        parent_pair_gap=parent_pair_gap,
        anti_overlap=True
    )

    fig = go.Figure()

    # Group children by parent pair.
    families = {}

    for child_id, rock in rocks_dict.items():
        if rock.parents is not None:
            key = tuple(sorted(rock.parents))
            families.setdefault(key, []).append(child_id)

    family_styles = get_family_styles(rocks_dict)

    # Draw family lines.
    for parent_pair, child_ids in sorted(families.items()):
        parent_pair = normalize_parent_pair_for_tree(parent_pair)

        if parent_pair is None:
            continue

        p1, p2 = parent_pair

        if p1 not in pos or p2 not in pos:
            continue

        x_line, y_line = build_family_segments(pos, parent_pair, child_ids)

        if len(x_line) == 0:
            continue

        style = family_styles.get(parent_pair, {"color": "#4E79A7", "dash": "solid"})

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=7,
                    color="rgba(255,255,255,0.95)",
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=3,
                    color=style["color"],
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Add rock images.
    image_cache = {}

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        if rid not in image_cache:
            image_cache[rid] = rock_to_image_uri_cached(game, rock)

        size_scale = get_rock_size_scale(rock)

        opacity = 1.0
        if getattr(rock, "sold", False):
            opacity = inactive_sold_opacity

        fig.add_layout_image(
            dict(
                source=image_cache[rid],
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=rock_image_size * size_scale,
                sizey=rock_image_size * size_scale,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=opacity
            )
        )

    # Selected rings.
    for rid in selected_ids:
        if rid not in pos:
            continue

        x, y = pos[rid]

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=rock_image_size * 85,
                    color="rgba(255,255,255,0)",
                    line=dict(
                        color="gold",
                        width=6
                    )
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Hover points, labels, and status symbols.
    hover_x = []
    hover_y = []
    hover_text = []
    labels = []

    status_x = []
    status_y = []
    status_text = []
    status_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]
        v = get_visual_phenotype(rock)

        parent_text = "Founder/import"
        if rock.parents is not None:
            parent_text = f"Parents: #{rock.parents[0]} and #{rock.parents[1]}"

        flags = []
        if getattr(rock, "sold", False):
            flags.append("SOLD")
        if getattr(rock, "used_as_parent", False):
            flags.append("BRED PARENT")
        if getattr(rock, "is_craisen", 0) == 1:
            flags.append("CRAISEN")
        if getattr(rock, "imported", False):
            flags.append("IMPORTED")

        flag_text = ", ".join(flags) if flags else "OK"

        full_phenotype_text = format_selected_phenotype_hover(rock)

        text = (
            f"<b>{rock.name} #{rock.id}</b><br>"
            f"Generation: {rock.generation}<br>"
            f"{parent_text}<br>"
            f"Gender: {v.get('gender', 'n/a')} {get_gender_symbol(rock)}<br>"
            f"Base value: ${rock.base_value}<br>"
            f"Sell value: ${rock.sell_value}<br>"
            f"Score value: ${rock.score_value}<br>"
            f"Status: {flag_text}<br>"
            f"<br>"
            f"<b>Full phenotype</b><br>"
            f"{full_phenotype_text}"
        )

        hover_x.append(x)
        hover_y.append(y)
        hover_text.append(text)

        if show_labels:
            labels.append(f"{rock.name}<br>#{rock.id}")
        else:
            labels.append("")

        symbol = get_rock_status_symbol(rock)

        if symbol != "":
            status_x.append(x)
            status_y.append(y - 0.72 * rock_image_size)
            status_text.append(symbol)
            status_colors.append(get_rock_status_color(rock))

            # Queued breeding pair markers.
        if highlight_breeding_queue and len(game.breeding_queue) > 0:
            queue_labels_by_rock = get_queue_labels_by_rock(game)

            # Draw dashed red lines between currently queued future parents.
            for i, entry in enumerate(game.breeding_queue, start=1):
                a, b = get_queue_entry_pair(entry)

                if a not in pos or b not in pos:
                    continue

                xa, ya = pos[a]
                xb, yb = pos[b]

                # Slightly above rocks so it reads as a planned pair, not lineage.
                ya2 = ya + 0.78 * rock_image_size
                yb2 = yb + 0.78 * rock_image_size

                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[ya2, yb2],
                        mode="lines",
                        line=dict(
                            color="crimson",
                            width=3,
                            dash="dot"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

                mid_x = 0.5 * (xa + xb)
                mid_y = 0.5 * (ya2 + yb2)

                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y + 0.12 * rock_image_size],
                        mode="text",
                        text=[f"❤{i}"],
                        textfont=dict(
                            size=28,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            # Draw heart labels on each queued rock.
            for rid, labels in queue_labels_by_rock.items():
                if rid not in pos:
                    continue

                x, y = pos[rid]
                rock = rocks_dict[rid]
                size_scale = get_rock_size_scale(rock)

                fig.add_trace(
                    go.Scatter(
                        x=[x - 0.42 * rock_image_size * size_scale],
                        y=[y + 0.42 * rock_image_size * size_scale],
                        mode="text",
                        text=[" ".join(labels)],
                        textfont=dict(
                            size=24,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

    # Invisible hover/labels.
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=rock_image_size * 48,
                color="rgba(0,0,0,0)"
            ),
            text=labels,
            textposition="bottom center",
            textfont=dict(size=10, color="black"),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        )
    )

    # Status symbols.
    for sx, sy, st, sc in zip(status_x, status_y, status_text, status_colors):
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="text",
                text=[st],
                textfont=dict(
                    size=30,
                    color=sc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

        # Gender symbols near top-right of each rock.
    gender_x = []
    gender_y = []
    gender_text = []
    gender_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        size_scale = get_rock_size_scale(rock)

        gender_x.append(x + 0.42 * rock_image_size * size_scale)
        gender_y.append(y + 0.42 * rock_image_size * size_scale)
        gender_text.append(get_gender_symbol(rock))
        gender_colors.append(get_gender_color(rock))

    for gx, gy, gt, gc in zip(gender_x, gender_y, gender_text, gender_colors):
        fig.add_trace(
            go.Scatter(
                x=[gx],
                y=[gy],
                mode="text",
                text=[gt],
                textfont=dict(
                    size=26,
                    color=gc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    margin_x = 3.0
    margin_y = 3.0

    fig.update_layout(
        title=(
            f"Rock Game Tree — Generation {game.generation}/{game.max_generation} "
            f"| Cash ${game.money} | Score Estimate ${get_final_score_estimate(game)}"
        ),
        width=canvas_width,
        height=canvas_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            visible=False,
            range=[min(xs) - margin_x, max(xs) + margin_x]
        ),
        yaxis=dict(
            visible=False,
            range=[min(ys) - margin_y, max(ys) + margin_y],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=20, r=20, t=70, b=20),
        dragmode="pan"
    )

    if show:
        fig.show(config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        })

    return fig

# -----------------------------
# ROCK FLAGS AND SYMBOLS
# -----------------------------

def is_rock_sold_flag(rock):
    return bool(getattr(rock, "sold", False))

def get_rock_status_symbol(rock):
    """
    Symbol shown near rocks in game views.
    """
    if getattr(rock, "puffed", False):
        return "☁"

    if getattr(rock, "dead", False):
        return "†"

    if is_rock_sold_flag(rock):
        return "$"

    if getattr(rock, "is_craisen", 0) == 1:
        return "X"

    if getattr(rock, "used_as_parent", False):
        return "○"

    if getattr(rock, "market_guest", False):
        return(("NPC", "darkviolet"))

    return ""

def get_rock_status_color(rock):
    """
    Color for status symbols.
    """
    if getattr(rock, "puffed", False):
        return "dimgray"

    if getattr(rock, "dead", False):
        return "black"

    if is_rock_sold_flag(rock):
        return "green"

    if getattr(rock, "is_craisen", 0) == 1:
        return "crimson"

    if getattr(rock, "used_as_parent", False):
        return "gray"

    return "black"









