#-----------------------------------------------------
"""
Draw orchestration for individual rock renders.
"""
#-----------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Drawing.rock_render_context import RockRenderContext
from Rock_Drawing.rock_feature_drawers import (
    draw_arms, draw_brows, draw_crown, draw_ears, draw_eyes, draw_facial_hair,
    draw_freckles, draw_fuzz, draw_hair, draw_halo, draw_horns, draw_mouth,
    draw_nose, draw_patchwork, draw_stones, draw_tail, draw_wings, draw_wrinkles,
)

GIANT_SIZE_SCALE = 2.30

@dataclass(frozen=True)
class IdentityLabelLayout:
    """
    Controls identity-label placement in reference body-radius units.
    """

    label_reference_size_scale: float = GIANT_SIZE_SCALE
    name_below_body_radius: float = GIANT_SIZE_SCALE #1.0
    gender_right_body_radius: float = GIANT_SIZE_SCALE #1.0
    gender_above_body_radius: float = GIANT_SIZE_SCALE #0.95
    status_left_body_radius: float = GIANT_SIZE_SCALE #1.0
    status_above_body_radius: float = GIANT_SIZE_SCALE #0.95
    name_font_size: int = 8
    gender_font_size: int = 18
    status_font_size: int = 16
    fallback_name_line_chars: int = 20
    structured_name_line_chars: int = 24
    font_weight: str = "bold"


DEFAULT_IDENTITY_LABEL_LAYOUT = IdentityLabelLayout()

@dataclass
class DrawMachine:
    """
    Orchestrates one rock render.
    """

    rock: genetics.Rock
    ax: Axes | None = None
    show_genes: bool = False
    normalize_size: bool = True
    identity_layout: IdentityLabelLayout = DEFAULT_IDENTITY_LABEL_LAYOUT

    ctx: RockRenderContext | None = None
    drawn_eye_positions: list[tuple[float, float, float]] = field(default_factory = list)
    nose_info: Any = None
    mouth_info: Any = None

    def ensure_axis(
        self
    ) -> Axes:
        if self.ax is None:
            _, self.ax = plt.subplots(figsize = (4, 4))

        return self.ax

    def make_context(
        self
    ) -> RockRenderContext:
        self.ctx = RockRenderContext.from_rock(
            rock = self.rock,
            ax = self.ensure_axis(),
        )

        return self.ctx

    def draw_external_traits(
        self
    ):
        ctx = self.ctx

        draw_wings(ctx)
        draw_fuzz(ctx)
        draw_halo(ctx)
        draw_stones(ctx)
        draw_tail(ctx)
        draw_horns(ctx)

    def draw_body(
        self
    ):
        ctx = self.ctx

        ctx.ax.add_patch(ctx.body)
        draw_patchwork(ctx)

    def draw_body_attached_traits(
        self
    ):
        ctx = self.ctx

        draw_hair(ctx)
        draw_ears(ctx)
        draw_wrinkles(ctx)
        draw_freckles(ctx)
        draw_arms(ctx)
        draw_crown(ctx)

    def draw_face(
        self
    ):
        ctx = self.ctx

        self.drawn_eye_positions = draw_eyes(ctx)
        draw_brows(ctx, self.drawn_eye_positions)
        self.nose_info = draw_nose(ctx, self.drawn_eye_positions)
        self.mouth_info = draw_mouth(ctx, self.drawn_eye_positions)

        draw_facial_hair(
            ctx,
            drawn_eye_positions = self.drawn_eye_positions,
            nose_info = self.nose_info,
            mouth_info = self.mouth_info,
        )

    def draw_identity_label(
        self
    ):
        ctx = self.ctx
        gender_symbol = "\u2642" if self.rock.sex == genetics.Sex.MALE else "\u2640"
        gender_color = "royalblue" if self.rock.sex == genetics.Sex.MALE else "deeppink"
        name = self.format_rock_name()
        status_symbol, status_color = self.get_status_symbol_and_color()
        layout = self.identity_layout
        reference_body_radius = 0.5 * layout.label_reference_size_scale

        gender_x = ctx.xmax + layout.gender_right_body_radius * reference_body_radius
        gender_y = ctx.ymax + layout.gender_above_body_radius * reference_body_radius
        name_y = ctx.ymin - layout.name_below_body_radius * reference_body_radius
        status_x = ctx.xmin - layout.status_left_body_radius * reference_body_radius
        status_y = ctx.ymax + layout.status_above_body_radius * reference_body_radius

        ctx.ax.text(
            gender_x,
            gender_y,
            gender_symbol,
            color=gender_color,
            ha="center",
            va="center",
            fontsize=layout.gender_font_size,
            fontweight=layout.font_weight,
            zorder=30,
        )

        if status_symbol:
            ctx.ax.text(
                status_x,
                status_y,
                status_symbol,
                color=status_color,
                ha="center",
                va="center",
                fontsize=layout.status_font_size,
                fontweight=layout.font_weight,
                zorder=31,
                bbox={
                    "boxstyle": "circle,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": status_color,
                    "linewidth": 1.4,
                    "alpha": 0.85,
                },
            )

        ctx.ax.text(
            ctx.cx,
            name_y,
            f"#{self.rock.id}: {name}",
            color=gender_color,
            ha="center",
            va="center",
            fontsize=layout.name_font_size,
            fontweight=layout.font_weight,
            linespacing=1.0,
            zorder=30,
        )

    def format_rock_name(
        self
    ) -> str:
        name = self.rock.name

        if hasattr(name, "given"):
            lines = []

            if getattr(name, "honorific", None):
                lines.append(str(name.honorific))

            core = str(name.given)
            if getattr(name, "family", None):
                core = f"{core} {name.family}"
            lines.extend(
                textwrap.wrap(
                    core,
                    width=max(8, self.identity_layout.structured_name_line_chars),
                )
                or [core]
            )

            if getattr(name, "epithet", None):
                epithet = str(name.epithet)
                lines.extend(
                    textwrap.wrap(
                        epithet,
                        width=max(8, self.identity_layout.structured_name_line_chars),
                    )
                    or [epithet]
                )

            return "\n".join(lines)

        raw_name = str(name)
        wrapped = textwrap.wrap(
            raw_name,
            width=max(6, self.identity_layout.fallback_name_line_chars),
        )
        return "\n".join(wrapped) if wrapped else raw_name

    def get_status_symbol_and_color(
        self
    ) -> tuple[str, str]:
        status = self.rock.status

        if bool(getattr(self.rock, "puffed", False)):
            return "p", "royalblue"
        if status == genetics.RockStatus.SOLD:
            return "$", "green"
        if status == genetics.RockStatus.DEAD:
            return "\u271d", "black"
        if status == genetics.RockStatus.CRAISENED:
            return "x", "crimson"
        if status == genetics.RockStatus.BRED:
            return "o", "gray"
        if bool(getattr(self.rock, "is_market", False)):
            return "I", "darkorange"

        return "", "black"

    def finalize(
        self
    ):
        self.ctx.apply_camera(normalize_size = self.normalize_size)
        self.ctx.apply_labels(show_genes = self.show_genes)

    def draw(
        self
    ) -> Axes:
        self.make_context()
        self.draw_external_traits()
        self.draw_body()
        self.draw_body_attached_traits()
        self.draw_face()
        self.draw_identity_label()
        self.finalize()

        return self.ctx.ax

def draw_rock(rock, ax=None, show_genes=False, normalize_size=True, identity_layout=DEFAULT_IDENTITY_LABEL_LAYOUT):
    """
    Trait-based rock renderer.
    """

    machine = DrawMachine(
        rock = rock,
        ax = ax,
        show_genes = show_genes,
        normalize_size = normalize_size,
        identity_layout = identity_layout,
    )

    return machine.draw()



