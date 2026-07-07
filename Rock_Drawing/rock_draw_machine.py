#-----------------------------------------------------
"""
Draw orchestration for individual rock renders.
"""
#-----------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
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

@dataclass(frozen=True)
class IdentityLabelLayout:
    """
    Controls identity-label placement in body-radius units.
    """

    name_below_body_radius: float = 0.5
    gender_right_body_radius: float = 0.5
    gender_above_body_radius: float = 0.5
    name_font_size: int = 8
    gender_font_size: int = 18
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

    def draw_craisen_overlay(
        self
    ):
        ctx = self.ctx

        if not ctx.v.get("is_craisen", False):
            return

        ctx.ax.plot(
            [-0.75 * ctx.s, 0.75 * ctx.s],
            [-0.75 * ctx.s, 0.75 * ctx.s],
            color = "crimson",
            linewidth = 4,
            zorder = 20,
        )
        ctx.ax.plot(
            [-0.75 * ctx.s, 0.75 * ctx.s],
            [0.75 * ctx.s, -0.75 * ctx.s],
            color = "crimson",
            linewidth = 4,
            zorder = 20,
        )
        ctx.ax.text(
            0,
            -1.25 * ctx.s,
            "CRAISEN",
            color = "crimson",
            ha = "center",
            va = "center",
            fontsize = 10,
            fontweight = "bold",
        )

    def draw_identity_label(
        self
    ):
        ctx = self.ctx
        gender_symbol = "\u2642" if self.rock.sex == genetics.Sex.MALE else "\u2640"
        gender_color = "royalblue" if self.rock.sex == genetics.Sex.MALE else "deeppink"
        name = self.rock.name.full_name if hasattr(self.rock.name, "full_name") else str(self.rock.name)
        body_radius = 0.5 * ctx.unit
        layout = self.identity_layout

        gender_x = ctx.xmax + layout.gender_right_body_radius * body_radius
        gender_y = ctx.ymax + layout.gender_above_body_radius * body_radius
        name_y = ctx.ymin - layout.name_below_body_radius * body_radius

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
        ctx.ax.text(
            ctx.cx,
            name_y,
            f"#{self.rock.id}: {name}",
            color=gender_color,
            ha="center",
            va="center",
            fontsize=layout.name_font_size,
            fontweight=layout.font_weight,
            zorder=30,
        )

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
        self.draw_craisen_overlay()
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



