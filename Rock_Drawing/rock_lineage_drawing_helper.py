"""
Grid and lineage-tree drawing helpers for split-module rock games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import Rock_Genetics.rock_genetic_helper as genetics
from Rock_Drawing.rock_drawing_helper import (
    PAD_FRAC,
    pad_rock_axis,
    render_game_rock_images,
)
from Rock_Drawing.rock_draw_machine import draw_rock


def rock_display_name(rock: genetics.Rock) -> str:
    if hasattr(rock.name, "full_name"):
        return rock.name.full_name
    return str(rock.name)


def get_gender_symbol(rock: genetics.Rock) -> str:
    return "\u2642" if rock.sex == genetics.Sex.MALE else "\u2640"


def get_gender_color(rock: genetics.Rock) -> str:
    return "royalblue" if rock.sex == genetics.Sex.MALE else "deeppink"


def get_rock_status_symbol(rock: genetics.Rock) -> str:
    if rock.status == genetics.RockStatus.SOLD:
        return "$"
    if rock.status == genetics.RockStatus.DEAD:
        return "X"
    if rock.status == genetics.RockStatus.CRAISENED:
        return "!"
    if rock.status == genetics.RockStatus.BRED:
        return "o"
    if bool(getattr(rock, "is_market", False)):
        return "NPC"
    return ""


def get_rock_status_color(rock: genetics.Rock) -> str:
    if rock.status == genetics.RockStatus.SOLD:
        return "green"
    if rock.status in {genetics.RockStatus.DEAD, genetics.RockStatus.CRAISENED}:
        return "crimson"
    if rock.status == genetics.RockStatus.BRED:
        return "gray"
    if bool(getattr(rock, "is_market", False)):
        return "darkviolet"
    return "black"


def show_rocks(
    rock_items,
    rock_source=None,
    cols=6,
    figsize_per_rock=3.2,
    show_genes=False,
    title=None,
    sort_by_generation=False,
    normalize_size=True,
):
    """
    Display a Matplotlib grid of individual rocks.
    """
    if isinstance(rock_items, dict):
        rock_list = list(rock_items.values())
    else:
        rock_list = []
        for item in list(rock_items):
            if isinstance(item, genetics.Rock):
                rock_list.append(item)
            elif isinstance(item, int):
                if rock_source is None:
                    raise ValueError("rock_source is required when passing rock ids.")
                rock_list.append(rock_source[item])
            else:
                raise TypeError("show_rocks expects rocks, rock ids, or a rock dictionary.")

    if sort_by_generation:
        rock_list = sorted(rock_list, key=lambda rock: (rock.generation, rock.id))

    if not rock_list:
        return None, None

    cols = max(1, min(cols, len(rock_list)))
    rows = math.ceil(len(rock_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_rock, rows * figsize_per_rock))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, rock in zip(axes, rock_list):
        draw_rock(rock, ax=ax, show_genes=show_genes, normalize_size=normalize_size)
        pad_rock_axis(ax, pad_frac=PAD_FRAC)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    return fig, axes


@dataclass
class TreeHelper:
    """
    Calculates lineage-tree generations, positions, family links, and labels.
    """

    rocks: dict[int, genetics.Rock]
    x_gap: float = 3.2
    y_gap: float = 3.2
    min_node_gap: float = 2.2
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_game(cls, game, **kwargs) -> "TreeHelper":
        return cls(rocks=dict(game.rocks), **kwargs)

    def generation_groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for rock_id, rock in self.rocks.items():
            groups.setdefault(int(getattr(rock, "generation", 0)), []).append(int(rock_id))

        for ids in groups.values():
            ids.sort()

        return dict(sorted(groups.items()))

    def compute_positions(self) -> dict[int, tuple[float, float]]:
        groups = self.generation_groups()
        positions: dict[int, tuple[float, float]] = {}

        for generation, rock_ids in groups.items():
            count = len(rock_ids)
            start_x = -0.5 * (count - 1) * self.x_gap
            for index, rock_id in enumerate(rock_ids):
                positions[rock_id] = (start_x + index * self.x_gap, -generation * self.y_gap)

        for _ in range(3):
            positions = self.pull_children_toward_parent_centers(positions)
            positions = self.resolve_generation_overlap(positions, groups)

        self.positions = positions
        return positions

    def pull_children_toward_parent_centers(
        self,
        positions: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        updated = dict(positions)

        for child_id, rock in self.rocks.items():
            parents = self.parent_ids(rock)
            if len(parents) != 2:
                continue
            if parents[0] not in positions or parents[1] not in positions or child_id not in positions:
                continue

            parent_center = 0.5 * (positions[parents[0]][0] + positions[parents[1]][0])
            child_x, child_y = positions[child_id]
            updated[child_id] = (0.55 * child_x + 0.45 * parent_center, child_y)

        return updated

    def resolve_generation_overlap(
        self,
        positions: dict[int, tuple[float, float]],
        groups: dict[int, list[int]],
    ) -> dict[int, tuple[float, float]]:
        updated = dict(positions)

        for rock_ids in groups.values():
            ordered = sorted(rock_ids, key=lambda rock_id: updated[rock_id][0])
            if len(ordered) <= 1:
                continue

            for index in range(1, len(ordered)):
                previous_id = ordered[index - 1]
                current_id = ordered[index]
                previous_x = updated[previous_id][0]
                current_x, current_y = updated[current_id]
                if current_x - previous_x < self.min_node_gap:
                    updated[current_id] = (previous_x + self.min_node_gap, current_y)

            mean_x = sum(updated[rock_id][0] for rock_id in ordered) / len(ordered)
            for rock_id in ordered:
                x, y = updated[rock_id]
                updated[rock_id] = (x - mean_x, y)

        return updated

    def family_links(self) -> list[tuple[int, int, int]]:
        links = []
        for child_id, rock in self.rocks.items():
            parents = self.parent_ids(rock)
            if len(parents) == 2 and parents[0] in self.rocks and parents[1] in self.rocks:
                links.append((parents[0], parents[1], int(child_id)))
        return links

    @staticmethod
    def parent_ids(rock: genetics.Rock) -> list[int]:
        return [int(parent_id) for parent_id in getattr(rock, "parent_ids", []) if parent_id is not None]

    def bounds(self, padding: float = 2.5) -> dict[str, list[float]]:
        if not self.positions:
            self.compute_positions()

        xs = [pos[0] for pos in self.positions.values()] or [0.0]
        ys = [pos[1] for pos in self.positions.values()] or [0.0]
        return {
            "x": [min(xs) - padding, max(xs) + padding],
            "y": [min(ys) - padding, max(ys) + padding],
        }


@dataclass
class TreeDrawer:
    """
    Uses TreeHelper output and cached rock images to build a Plotly lineage tree.
    """

    game: Any
    helper: TreeHelper | None = None
    rock_image_size: float = 1.35
    canvas_width: int = 1200
    canvas_height: int = 800

    def __post_init__(self):
        if self.helper is None:
            self.helper = TreeHelper.from_game(self.game)

    def draw(self, show: bool = False) -> go.Figure:
        positions = self.helper.compute_positions()
        image_by_id = render_game_rock_images(self.game)
        fig = go.Figure()

        self.add_family_lines(fig, positions)
        self.add_rock_images(fig, positions, image_by_id)
        self.add_node_text(fig, positions)
        self.apply_layout(fig)

        if show:
            fig.show(config={"scrollZoom": True, "displayModeBar": True, "responsive": True})

        return fig

    def add_family_lines(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        for parent_a_id, parent_b_id, child_id in self.helper.family_links():
            xa, ya = positions[parent_a_id]
            xb, yb = positions[parent_b_id]
            xc, yc = positions[child_id]
            parent_bar_y = min(ya, yb) - 0.65
            child_bar_y = yc + 0.75
            parent_center_x = 0.5 * (xa + xb)

            x_values = [
                xa, xa, None,
                xb, xb, None,
                xa, xb, None,
                parent_center_x, parent_center_x, None,
                parent_center_x, xc, None,
                xc, xc,
            ]
            y_values = [
                ya - 0.45, parent_bar_y, None,
                yb - 0.45, parent_bar_y, None,
                parent_bar_y, parent_bar_y, None,
                parent_bar_y, child_bar_y, None,
                child_bar_y, child_bar_y, None,
                child_bar_y, yc + 0.45,
            ]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"color": "#4E79A7", "width": 3},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def add_rock_images(
        self,
        fig: go.Figure,
        positions: dict[int, tuple[float, float]],
        image_by_id: dict[int, str],
    ) -> None:
        for rock_id, rock in self.helper.rocks.items():
            x, y = positions[rock_id]
            opacity = 0.45 if rock.status == genetics.RockStatus.SOLD else 1.0

            fig.add_layout_image(
                {
                    "source": image_by_id[rock_id],
                    "xref": "x",
                    "yref": "y",
                    "x": x,
                    "y": y,
                    "sizex": self.rock_image_size,
                    "sizey": self.rock_image_size,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "layer": "above",
                    "opacity": opacity,
                }
            )

    def add_node_text(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        for rock_id, rock in self.helper.rocks.items():
            x, y = positions[rock_id]
            label = f"{get_gender_symbol(rock)}<br>#{rock.id}: {rock_display_name(rock)}"
            status_symbol = get_rock_status_symbol(rock)

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y - 1.0],
                    mode="text",
                    text=[label],
                    textfont={"size": 12, "color": get_gender_color(rock)},
                    hovertext=[self.hover_text(rock)],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            if status_symbol:
                fig.add_trace(
                    go.Scatter(
                        x=[x + 0.75],
                        y=[y + 0.65],
                        mode="text",
                        text=[status_symbol],
                        textfont={"size": 14, "color": get_rock_status_color(rock)},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    @staticmethod
    def hover_text(rock: genetics.Rock) -> str:
        parents = ", ".join(str(parent_id) for parent_id in rock.parent_ids) or "founder"
        return (
            f"<b>#{rock.id}: {rock_display_name(rock)}</b><br>"
            f"sex: {rock.sex.value}<br>"
            f"generation: {rock.generation}<br>"
            f"parents: {parents}<br>"
            f"status: {rock.status.value}<br>"
            f"value: ${rock.value}<br>"
            f"sell: ${rock.sell_value}"
        )

    def apply_layout(self, fig: go.Figure) -> None:
        bounds = self.helper.bounds()
        fig.update_layout(
            title=f"Rock Lineage Tree | Generation {self.game.generation} | ${self.game.money}",
            width=self.canvas_width,
            height=self.canvas_height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"visible": False, "range": bounds["x"]},
            yaxis={"visible": False, "range": bounds["y"], "scaleanchor": "x", "scaleratio": 1},
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            dragmode="pan",
        )


def draw_game_tree(game, show: bool = False, **kwargs) -> go.Figure:
    return TreeDrawer(game=game, **kwargs).draw(show=show)
