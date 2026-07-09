#-----------------------------------------------------
"""
Split-out module from rock_drawing_helper.py.
"""
#-----------------------------------------------------

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Drawing.rock_drawing_phenotype import BODY_COLOR_MAP, get_visual_phenotype
from Rock_Drawing.rock_drawing_geometry import make_body_points

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
# BODY COLOR MAP FOR CONTEXT
#-----------------------------------------------------

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

#-----------------------------------------------------
# ROCK RENDERER CONTEXT: NEW AND IMPORVED
#-----------------------------------------------------

@dataclass
class RockRenderContext:

    ax: Axes

    body: object
    body_points: np.ndarray
    body_color: object

    size_scale: float

    rng: object
    py_rng: object

    rock: genetics.Rock
    v: dict[str, Any] | None = None

    presence: FeaturePresence = field(default_factory = FeaturePresence)
    face_layout: FaceLayout = field(default_factory = FaceLayout)

    body_color_map = BODY_COLOR_MAP

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

        if self.v is None:
            self.v = get_visual_phenotype(self.rock)

        self.presence = self.build_feature_presence()
        self.face_layout = self.build_face_layout()

    @property
    def s(
        self
    ) -> float:
        """
        Compatibility alias for old drawing functions.
        """

        return self.size_scale

    #-----------------------------------------------------
    # BODY FINDING HELPERS
    #-----------------------------------------------------

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

    #-----------------------------------------------------
    # FEATURE HELPERS
    #-----------------------------------------------------

    def phen(
        self, 
        gene_name: str, 
        default: str = "n/a"
    ) -> str:
        
        """
        Get the already-computed phenotype for a gene.
        """

        if self.rock is not None:
            genotype = self.rock.genotype.genes.get(gene_name)

            if genotype is None:
                return default

            phenotype = genotype.phenotype

            if phenotype is not None:
                return phenotype
            else:
                return (default)
        
        else:
            return (default)

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

    def feature_body_span(
        self,
        feature_name: str,
        fallback_ny: float = 0.5
    ) -> tuple[float, float, float, FeatureSlot | None]:
        """
        Return the body-safe horizontal span for a feature's planned layout slot.
        """

        slot = self.face_layout.get(feature_name)
        ny = fallback_ny if slot is None else slot.ny
        x_left, x_right = self.body_x_span_at_ny(ny)
        y = self.ymin + ny * self.height

        return x_left, x_right, y, slot
    
    #-----------------------------------------------------
    # FEATURE PRESENCE BUILDING
    #-----------------------------------------------------

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

    def build_feature_presence(
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

    #-----------------------------------------------------
    # FACE LAYOUT BUILDING
    #-----------------------------------------------------

    def build_face_layout(
        self
    ) -> FaceLayout:
        
        """
        Build dynamic facial feature positions.

        If only eyes are present, they sit more centrally.
        If many features are present, they spread vertically.
        """

        slots: dict[str, FeatureSlot] = {}

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

        upper_feature_pressure = 0.0

        if self.presence.hair:
            upper_feature_pressure += 0.045

        if self.presence.crown:
            upper_feature_pressure += 0.025

        if self.presence.horns:
            upper_feature_pressure += 0.025

        upper_feature_pressure = min(upper_feature_pressure, 0.055)

        base_slots = {
            "brows": FeatureSlot("brows", 0.5, 0.76 - upper_feature_pressure, 0.95),
            "eyes": FeatureSlot("eyes", 0.5, 0.62 - upper_feature_pressure, 1.0),
            "nose": FeatureSlot("nose", 0.5, 0.47, 0.95),
            "mouth": FeatureSlot("mouth", 0.5, 0.32, 1.0),
            "facial_hair": FeatureSlot("facial_hair", 0.5, 0.17, 1.05),
        }

        if ordered_face_features == ["eyes"]:

            slots["eyes"] = FeatureSlot(
                name = "eyes",
                nx = 0.5,
                ny = 0.56,
                scale = 1.15,
            )

            return FaceLayout(slots=slots)

        if len(ordered_face_features) == 1:

            feature = ordered_face_features[0]
            slot = base_slots.get(
                feature,
                FeatureSlot(feature, 0.5, 0.50, 1.05),
            )

            slots[feature] = FeatureSlot(
                name = feature,
                nx = slot.nx,
                ny = slot.ny,
                scale = max(slot.scale, 1.05),
            )

            return FaceLayout(slots = slots)

        previous_ny = None

        for feature in ordered_face_features:
            slot = base_slots[feature]
            ny = slot.ny

            if previous_ny is not None:
                min_gap = 0.13

                if feature == "facial_hair":
                    min_gap = 0.15

                if previous_ny - ny < min_gap:
                    ny = previous_ny - min_gap

            lower_limit = 0.13 if feature == "facial_hair" else 0.22
            upper_limit = 0.82
            ny = max(lower_limit, min(upper_limit, ny))

            slots[feature] = FeatureSlot(
                name = feature,
                nx = slot.nx,
                ny = float(ny),
                scale = slot.scale,
            )

            previous_ny = ny

        return FaceLayout(slots = slots)

    #-----------------------------------------------------
    # BUILDING ROCK RENDER CONTEXT BODY POINTS
    #-----------------------------------------------------
    
    @staticmethod
    def get_body_color_from_rock(
        rock,
        body_color_map = BODY_COLOR_MAP
    ):

        color_name = get_visual_phenotype(rock).get("color", "n/a")

        return body_color_map.get(color_name, body_color_map["n/a"])

    @staticmethod
    def get_phenotype_from_rock(
        rock, 
        gene_name: str, 
        fallback = "n/a"
    ):
        
        gene = rock.genotype.genes[gene_name]

        if gene is not None and hasattr(gene, "phenotype"):
            phenotype = gene.phenotype
            if phenotype is not None:
                return phenotype

        return fallback
    
    @classmethod
    def from_rock(
        cls, 
        rock, 
        ax
    ):

        rng = np.random.default_rng(
            2_000 + abs(rock.id)
        
        )
        py_rng = random.Random(
            1_000 + abs(rock.id)
        )

        shape_name = cls.get_phenotype_from_rock(
            rock = rock,
            gene_name = "shape",
            fallback = "circle",
        )

        size_name = cls.get_phenotype_from_rock(
            rock = rock,
            gene_name = "size",
            fallback = "medium",
        )

        body_points, size_scale = make_body_points(
            shape_name = shape_name,
            size_name = size_name,
            rng = rng,
        )

        body_color = cls.get_body_color_from_rock(
            rock = rock,
        )

        body = Polygon(
            body_points,
            closed = True,
            facecolor = body_color,
            edgecolor = "black",
            linewidth = 2.0,
            zorder = 1,
        )

        return cls(
            ax = ax,

            body = body,
            body_points = body_points,
            body_color = body_color,

            size_scale = size_scale,

            rng = rng,
            py_rng = py_rng,

            rock = rock,
            v = get_visual_phenotype(rock),
            
        )

    #-----------------------------------------------------
    # HELPFUL IMAGE SETUPS
    #-----------------------------------------------------

    def apply_camera(
        self, 
        normalize_size: bool = True
    ):
        
        self.ax.set_title(f"{self.rock.name} #{self.rock.id} \n Gen {self.rock.generation}")
        self.ax.set_aspect("equal")

        if normalize_size:
            self.ax.set_xlim(-2.35 * self.size_scale, 2.35 * self.size_scale)
            self.ax.set_ylim(-1.85 * self.size_scale, 1.95 * self.size_scale)
        else:
            self.ax.set_xlim(-3.35, 3.35)
            self.ax.set_ylim(-2.75, 3.05)

        self.ax.axis("off")

    def apply_labels(
        self, 
        show_genes: bool = False
    ):
        
        if not show_genes:
            return

        gene_text = "\n".join(
            f"{gene_name}: {gene_pair}"
            for gene_name, gene_pair in self.rock.genotype.genes.items()
        )

        self.ax.text(
            1.55 * self.size_scale,
            1.35 * self.size_scale,
            gene_text,
            fontsize = 7,
            va = "top",
            family = "monospace",
        )

