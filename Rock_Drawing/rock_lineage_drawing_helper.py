"""
Grid and lineage-tree drawing helpers for split-module rock games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import math
import random
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


FAMILY_LINE_COLORS = (
    "#4E79A7",
    "#59A14F",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
    "#2F4B7C",
    "#A05195",
    "#665191",
    "#003F5C",
    "#1B9E77",
    "#66A61E",
    "#3288BD",
    "#5E4FA2",
    "#8DA0CB",
    "#A6D854",
)

FAMILY_LINE_DASHES = (
    "solid",
    "dash",
    "dot",
    "dashdot",
    "longdash",
    "longdashdot",
)


def shuffled_family_colors(count: int, seed: int | None = None) -> list[str]:
    """
    Pick visually distinct non-red family colors for one generation's pods.
    """
    colors = list(FAMILY_LINE_COLORS)
    random.Random(seed).shuffle(colors)
    if count <= len(colors):
        return colors[:count]

    return [colors[index % len(colors)] for index in range(count)]


def rock_display_name(rock: genetics.Rock) -> str:
    if hasattr(rock.name, "full_name"):
        return rock.name.full_name
    return str(rock.name)


def get_gender_symbol(rock: genetics.Rock) -> str:
    return "\u2642" if rock.sex == genetics.Sex.MALE else "\u2640"


def get_gender_color(rock: genetics.Rock) -> str:
    return "royalblue" if rock.sex == genetics.Sex.MALE else "deeppink"


def get_rock_status_symbol(rock: genetics.Rock) -> str:
    if bool(getattr(rock, "puffed", False)):
        return "p"
    if rock.status == genetics.RockStatus.SOLD:
        return "$"
    if rock.status == genetics.RockStatus.DEAD:
        return "\u271d"
    if rock.status == genetics.RockStatus.CRAISENED:
        return "x"
    if rock.status == genetics.RockStatus.BRED:
        return "o"
    if bool(getattr(rock, "is_market", False)):
        return "I"
    return ""


def get_rock_status_color(rock: genetics.Rock) -> str:
    if bool(getattr(rock, "puffed", False)):
        return "royalblue"
    if rock.status == genetics.RockStatus.SOLD:
        return "green"
    if rock.status == genetics.RockStatus.DEAD:
        return "black"
    if rock.status == genetics.RockStatus.CRAISENED:
        return "crimson"
    if rock.status == genetics.RockStatus.BRED:
        return "gray"
    if bool(getattr(rock, "is_market", False)):
        return "darkorange"
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
    node_width: float = 1.35
    node_height: float = 1.35
    node_margin_x: float = 0.85
    branch_clearance: float = 0.28
    generation_gap_rocks: float = 1.8
    route_fudge: float = 0.25
    current_generation: int = 0
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_game(cls, game, **kwargs) -> "TreeHelper":
        kwargs.setdefault("current_generation", int(getattr(game, "generation", 0)))
        return cls(rocks=dict(game.rocks), **kwargs)

    def graph_generations(self) -> dict[int, int]:
        """
        Calculate visual-only generations for lineage layout.

        Gameplay generation stays untouched. Market/import founders with no
        parents render one lane above generation 0, and parent-child links are
        forced to move downward visually even when imported market pod records
        share the same gameplay generation.
        """
        graph_generations: dict[int, int] = {}
        for rock_id, rock in self.rocks.items():
            generation = int(getattr(rock, "generation", 0))
            if bool(getattr(rock, "is_market", False)) and not self.parent_ids(rock):
                generation = min(generation, self.current_generation - 1)
            graph_generations[int(rock_id)] = generation

        for _ in range(max(1, len(self.rocks))):
            changed = False
            for parent_a_id, parent_b_id, child_id in self.family_links():
                required_generation = max(
                    graph_generations[parent_a_id],
                    graph_generations[parent_b_id],
                ) + 1
                if graph_generations[child_id] < required_generation:
                    graph_generations[child_id] = required_generation
                    changed = True

            if not changed:
                break

        return graph_generations

    def generation_groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        graph_generations = self.graph_generations()
        for rock_id, rock in self.rocks.items():
            groups.setdefault(graph_generations[int(rock_id)], []).append(int(rock_id))

        for ids in groups.values():
            ids.sort()

        return dict(sorted(groups.items()))

    def compute_positions(self) -> dict[int, tuple[float, float]]:
        groups = self.generation_groups()
        positions: dict[int, tuple[float, float]] = {}
        x_gap = max(self.x_gap, self.required_x_gap())
        y_gap = max(self.y_gap, self.required_y_gap())

        for generation, rock_ids in groups.items():
            count = len(rock_ids)
            start_x = -0.5 * (count - 1) * x_gap
            for index, rock_id in enumerate(rock_ids):
                positions[rock_id] = (start_x + index * x_gap, -generation * y_gap)

        for _ in range(3):
            positions = self.pull_children_toward_parent_centers(positions)
            positions = self.resolve_generation_overlap(positions, groups)

        self.positions = positions
        return positions

    def required_x_gap(self) -> float:
        return self.node_width + self.node_margin_x

    def required_y_gap(self) -> float:
        return self.node_height * self.generation_gap_rocks

    def pull_children_toward_parent_centers(
        self,
        positions: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        updated = dict(positions)

        graph_generations = self.graph_generations()
        for child_id, rock in self.rocks.items():
            parents = self.parent_ids(rock)
            if len(parents) != 2:
                continue
            if parents[0] not in positions or parents[1] not in positions or child_id not in positions:
                continue
            if graph_generations[child_id] <= max(graph_generations[parents[0]], graph_generations[parents[1]]):
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
                required_gap = max(self.min_node_gap, self.required_x_gap())
                if current_x - previous_x < required_gap:
                    updated[current_id] = (previous_x + required_gap, current_y)

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

    def family_groups(self) -> dict[tuple[int, int], list[int]]:
        groups: dict[tuple[int, int], list[int]] = {}

        for parent_a_id, parent_b_id, child_id in self.family_links():
            key = tuple(sorted((parent_a_id, parent_b_id)))
            groups.setdefault(key, []).append(child_id)

        for child_ids in groups.values():
            child_ids.sort(key=lambda rock_id: (self.rocks[rock_id].generation, rock_id))

        return dict(sorted(groups.items()))

    def family_styles(
        self,
        seed: int | None = None,
        use_dash_styles: bool = False,
    ) -> dict[tuple[int, int], dict[str, str]]:
        parent_pairs = sorted({tuple(sorted((parent_a, parent_b))) for parent_a, parent_b, _ in self.family_links()})
        colors = shuffled_family_colors(len(parent_pairs), seed=seed)
        dashes = list(FAMILY_LINE_DASHES)
        random.Random(seed).shuffle(dashes)
        styles = {}

        for index, pair in enumerate(parent_pairs):
            styles[pair] = {
                "color": colors[index],
                "dash": dashes[index % len(dashes)] if use_dash_styles else "solid",
            }

        return styles

    def occupied_bounds(
        self,
        rock_id: int,
        positions: dict[int, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        if positions is None:
            positions = self.positions or self.compute_positions()

        x, y = positions[int(rock_id)]
        half_width = 0.5 * self.node_width + self.route_fudge
        half_height = 0.5 * self.node_height + self.route_fudge
        return {
            "left": x - half_width,
            "right": x + half_width,
            "bottom": y - half_height,
            "top": y + half_height,
        }

    def child_branch_y(
        self,
        parent_ids: tuple[int, int],
        child_ids: list[int],
        positions: dict[int, tuple[float, float]],
    ) -> float:
        child_top = max(self.occupied_bounds(child_id, positions)["top"] for child_id in child_ids)
        parent_bottom = min(self.occupied_bounds(parent_id, positions)["bottom"] for parent_id in parent_ids)
        desired_y = child_top + self.branch_clearance
        highest_open_y = parent_bottom - self.branch_clearance

        if desired_y < highest_open_y:
            return desired_y

        return 0.5 * (child_top + parent_bottom)

    def parent_branch_y(
        self,
        parent_ids: tuple[int, int],
        positions: dict[int, tuple[float, float]],
    ) -> float:
        parent_bottom = min(self.occupied_bounds(parent_id, positions)["bottom"] for parent_id in parent_ids)
        return parent_bottom - self.branch_clearance

    def clear_vertical_route_x(
        self,
        desired_x: float,
        y0: float,
        y1: float,
        positions: dict[int, tuple[float, float]],
        ignore_ids: set[int],
    ) -> float:
        low_y = min(y0, y1)
        high_y = max(y0, y1)

        def intersects_any(candidate_x: float) -> bool:
            for rock_id in self.rocks:
                if rock_id in ignore_ids:
                    continue

                bounds = self.occupied_bounds(rock_id, positions)
                overlaps_y = bounds["bottom"] <= high_y and bounds["top"] >= low_y
                inside_x = bounds["left"] <= candidate_x <= bounds["right"]
                if overlaps_y and inside_x:
                    return True

            return False

        if not intersects_any(desired_x):
            return desired_x

        step = max(self.required_x_gap() * 0.5, self.node_width)
        for distance in range(1, 8):
            for direction in (1, -1):
                candidate_x = desired_x + direction * distance * step
                if not intersects_any(candidate_x):
                    return candidate_x

        return desired_x

    def segment_hits_obstacle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        positions: dict[int, tuple[float, float]],
        ignore_ids: set[int],
    ) -> bool:
        x0, y0 = start
        x1, y1 = end
        low_x, high_x = sorted((x0, x1))
        low_y, high_y = sorted((y0, y1))

        for rock_id in self.rocks:
            if rock_id in ignore_ids:
                continue

            bounds = self.occupied_bounds(rock_id, positions)

            if x0 == x1:
                overlaps_y = bounds["bottom"] <= high_y and bounds["top"] >= low_y
                if bounds["left"] <= x0 <= bounds["right"] and overlaps_y:
                    return True

            elif y0 == y1:
                overlaps_x = bounds["left"] <= high_x and bounds["right"] >= low_x
                if bounds["bottom"] <= y0 <= bounds["top"] and overlaps_x:
                    return True

        return False

    def route_orthogonal(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        positions: dict[int, tuple[float, float]],
        ignore_ids: set[int],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Route an orthogonal connector around inflated rock image bounds.
        """
        if start == end:
            return []

        def clean_points(points):
            cleaned = [points[0]]
            for point in points[1:]:
                if point != cleaned[-1]:
                    cleaned.append(point)
            return cleaned

        def path_is_clear(points) -> bool:
            points = clean_points(points)
            return all(
                not self.segment_hits_obstacle(points[index - 1], points[index], positions, ignore_ids)
                for index in range(1, len(points))
            )

        def path_length(points) -> float:
            points = clean_points(points)
            return sum(
                abs(points[index][0] - points[index - 1][0]) + abs(points[index][1] - points[index - 1][1])
                for index in range(1, len(points))
            )

        candidates = []
        start_x, start_y = start
        end_x, end_y = end

        if start_x == end_x or start_y == end_y:
            candidates.append([start, end])

        candidates.extend(
            [
                [start, (end_x, start_y), end],
                [start, (start_x, end_y), end],
            ]
        )

        candidate_ys = {start_y, end_y}
        candidate_xs = {start_x, end_x}
        for rock_id in self.rocks:
            if rock_id in ignore_ids:
                continue
            bounds = self.occupied_bounds(rock_id, positions)
            candidate_ys.add(bounds["top"] + self.branch_clearance)
            candidate_ys.add(bounds["bottom"] - self.branch_clearance)
            candidate_xs.add(bounds["left"] - self.branch_clearance)
            candidate_xs.add(bounds["right"] + self.branch_clearance)

        for route_y in sorted(candidate_ys, key=lambda value: abs(value - 0.5 * (start_y + end_y))):
            candidates.append([start, (start_x, route_y), (end_x, route_y), end])

        for route_x in sorted(candidate_xs, key=lambda value: abs(value - 0.5 * (start_x + end_x))):
            candidates.append([start, (route_x, start_y), (route_x, end_y), end])

        clear_paths = [clean_points(points) for points in candidates if path_is_clear(points)]
        if clear_paths:
            best = min(clear_paths, key=lambda points: (len(points), path_length(points)))
            return [
                (best[index - 1], best[index])
                for index in range(1, len(best))
                if best[index - 1] != best[index]
            ]

        return [(start, end)]

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
    rock_image_sprite_size: float = 1.4
    rock_image_dpi: int = 400
    adaptive_image_resolution: bool = True
    render_images: bool = True
    canvas_width: int = 1200
    canvas_height: int = 800
    line_style_seed: int | None = None
    use_dash_styles: bool = False
    line_clearance: float = 0.28
    generation_gap_rocks: float = 1.8
    route_fudge: float = 0.25
    debug_connectors: bool = False
    highlighted_rock_ids: tuple[int, ...] = ()
    rock_badges: dict[int, dict[str, str]] = field(default_factory=dict)
    tree_checkbox_ids: tuple[int, ...] = ()
    tree_checked_ids: tuple[int, ...] = ()
    obstacle_routing_threshold: int = 70

    def __post_init__(self):
        if self.helper is None:
            self.helper = TreeHelper.from_game(self.game)
        self.helper.node_width = self.rock_image_size
        self.helper.node_height = self.rock_image_size
        self.helper.branch_clearance = self.line_clearance
        self.helper.generation_gap_rocks = self.generation_gap_rocks
        self.helper.route_fudge = self.route_fudge

    def draw(self, show: bool = False) -> go.Figure:
        positions = self.get_cached_positions()
        fig = go.Figure()

        self.add_family_lines(fig, positions)
        if self.render_images:
            sprite_size, dpi = self.image_render_settings()
            image_by_id = render_game_rock_images(
                self.game,
                sprite_size=sprite_size,
                dpi=dpi,
            )
            self.add_rock_images(fig, positions, image_by_id)
        self.add_node_text(fig, positions)
        self.add_rock_badges(fig, positions)
        self.add_tree_checkboxes(fig, positions)
        self.apply_layout(fig)

        if show:
            fig.show(config={"scrollZoom": True, "displayModeBar": True, "responsive": True})

        return fig

    def image_render_settings(self) -> tuple[float, int]:
        if not self.adaptive_image_resolution:
            return self.rock_image_sprite_size, self.rock_image_dpi

        rock_count = len(self.helper.rocks)
        if rock_count > 150:
            return min(self.rock_image_sprite_size, 1.0), min(self.rock_image_dpi, 100)
        if rock_count > 70:
            return min(self.rock_image_sprite_size, 1.2), min(self.rock_image_dpi, 250)
        return self.rock_image_sprite_size, self.rock_image_dpi

    def geometry_cache_key(self) -> tuple:
        rock_bits = tuple(
            (
                int(rock_id),
                int(getattr(rock, "generation", 0)),
                bool(getattr(rock, "is_market", False)),
                tuple(self.helper.parent_ids(rock)),
            )
            for rock_id, rock in sorted(self.helper.rocks.items())
        )
        return (
            "lineage_geometry_v2",
            int(getattr(self.game, "generation", 0)),
            float(self.rock_image_size),
            float(self.line_clearance),
            float(self.generation_gap_rocks),
            float(self.route_fudge),
            self.line_style_seed,
            bool(self.use_dash_styles),
            int(self.obstacle_routing_threshold),
            rock_bits,
        )

    def get_geometry_cache(self) -> dict:
        if not hasattr(self.game, "lineage_geometry_cache") or self.game.lineage_geometry_cache is None:
            self.game.lineage_geometry_cache = {}
        return self.game.lineage_geometry_cache

    def get_cached_positions(self) -> dict[int, tuple[float, float]]:
        cache = self.get_geometry_cache()
        key = self.geometry_cache_key()
        cached = cache.get(key)
        if cached is not None and "positions" in cached:
            self.helper.positions = dict(cached["positions"])
            return dict(cached["positions"])

        positions = self.helper.compute_positions()
        cache[key] = {"positions": dict(positions)}
        return positions

    def get_cached_family_line_specs(self, positions: dict[int, tuple[float, float]]) -> list[dict[str, Any]]:
        cache = self.get_geometry_cache()
        key = self.geometry_cache_key()
        cached = cache.setdefault(key, {"positions": dict(positions)})
        if "family_line_specs" not in cached:
            cached["family_line_specs"] = self.build_family_line_specs(positions)
        return cached["family_line_specs"]

    def add_family_lines(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        line_specs = self.get_cached_family_line_specs(positions)
        if line_specs:
            fig.add_traces(
                [
                    go.Scatter(
                        x=spec["x"],
                        y=spec["y"],
                        mode="lines",
                        line=spec["line"],
                        hoverinfo="skip",
                        showlegend=False,
                    )
                    for spec in line_specs
                ]
            )

        if self.debug_connectors:
            family_styles = self.helper.family_styles(
                seed=self.line_style_seed,
                use_dash_styles=self.use_dash_styles,
            )
            for family_key in self.helper.family_groups():
                parent_a_id, parent_b_id = family_key
                xa, ya = positions[parent_a_id]
                xb, yb = positions[parent_b_id]
                parent_connector = self.parent_pair_connector_segments(
                    parent_a_pos=(xa, ya),
                    parent_b_pos=(xb, yb),
                    parent_connector_y=self.helper.parent_branch_y(family_key, positions),
                    child_connector_y=0.0,
                )
                self.add_debug_connector_markers(
                    fig=fig,
                    endpoints=parent_connector["endpoints"],
                    color=family_styles[family_key]["color"],
                )

    def build_family_line_specs(self, positions: dict[int, tuple[float, float]]) -> list[dict[str, Any]]:
        family_styles = self.helper.family_styles(
            seed=self.line_style_seed,
            use_dash_styles=self.use_dash_styles,
        )
        grouped_line_specs: dict[tuple[str, str, int], dict[str, Any]] = {}

        for family_key, child_ids in self.helper.family_groups().items():
            parent_a_id, parent_b_id = family_key
            style = family_styles[family_key]
            xa, ya = positions[parent_a_id]
            xb, yb = positions[parent_b_id]
            parent_center_x = 0.5 * (xa + xb)
            parent_bar_y = self.helper.parent_branch_y(family_key, positions)
            child_branch_y = self.helper.child_branch_y(family_key, child_ids, positions)
            route_x = parent_center_x
            child_xs = [positions[child_id][0] for child_id in child_ids]

            ignore_ids = {parent_a_id, parent_b_id, *child_ids}
            parent_connector = self.parent_pair_connector_segments(
                parent_a_pos=(xa, ya),
                parent_b_pos=(xb, yb),
                parent_connector_y=parent_bar_y,
                child_connector_y=child_branch_y,
            )
            parent_segments = [
                ((xa, self.helper.occupied_bounds(parent_a_id, positions)["bottom"]), (xa, parent_bar_y)),
                ((xb, self.helper.occupied_bounds(parent_b_id, positions)["bottom"]), (xb, parent_bar_y)),
                parent_connector["horizontal"],
                parent_connector["vertical_drop"],
            ]
            segment_requests = []
            segment_requests.extend(self.horizontal_spans_to_spine(child_xs, route_x, child_branch_y))

            for child_id in child_ids:
                child_x, _ = positions[child_id]
                child_top = self.helper.occupied_bounds(child_id, positions)["top"]
                segment_requests.append(((child_x, child_branch_y), (child_x, child_top)))

            segments = []
            use_obstacle_routing = self.should_route_around_obstacles()
            for start, end in [*parent_segments, *segment_requests]:
                if use_obstacle_routing:
                    segments.extend(
                        self.helper.route_orthogonal(
                            start=start,
                            end=end,
                            positions=positions,
                            ignore_ids=ignore_ids,
                        )
                    )
                else:
                    segments.append((start, end))

            line_x = []
            line_y = []
            for (x0, y0), (x1, y1) in segments:
                if x0 == x1 and y0 == y1:
                    continue

                line_x.extend([x0, x1, None])
                line_y.extend([y0, y1, None])

            if line_x:
                group_key = (style["color"], style["dash"], 3)
                group = grouped_line_specs.setdefault(
                    group_key,
                    {
                        "x": [],
                        "y": [],
                        "line": {"color": style["color"], "dash": style["dash"], "width": 3},
                    },
                )
                group["x"].extend(line_x)
                group["y"].extend(line_y)

        return list(grouped_line_specs.values())

    def should_route_around_obstacles(self) -> bool:
        return len(self.helper.rocks) <= self.obstacle_routing_threshold

    @staticmethod
    def parent_pair_connector_segments(
        parent_a_pos: tuple[float, float],
        parent_b_pos: tuple[float, float],
        parent_connector_y: float,
        child_connector_y: float,
    ) -> dict[str, object]:
        parent_a_x, _ = parent_a_pos
        parent_b_x, _ = parent_b_pos
        x_start = min(parent_a_x, parent_b_x)
        x_end = max(parent_a_x, parent_b_x)
        couple_mid_x = 0.5 * (parent_a_x + parent_b_x)

        return {
            "horizontal": ((x_start, parent_connector_y), (x_end, parent_connector_y)),
            "vertical_drop": ((couple_mid_x, parent_connector_y), (couple_mid_x, child_connector_y)),
            "midpoint": (couple_mid_x, parent_connector_y),
            "endpoints": [(x_start, parent_connector_y), (x_end, parent_connector_y)],
        }

    @staticmethod
    def add_debug_connector_markers(
        fig: go.Figure,
        endpoints: list[tuple[float, float]],
        color: str,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in endpoints],
                y=[point[1] for point in endpoints],
                mode="markers",
                marker={"size": 7, "color": color, "symbol": "x"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    @staticmethod
    def horizontal_spans_to_spine(
        anchor_xs: list[float],
        spine_x: float,
        y: float,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if not anchor_xs:
            return []

        left = min(anchor_xs)
        right = max(anchor_xs)

        if spine_x <= left:
            return [((spine_x, y), (right, y))]
        if spine_x >= right:
            return [((left, y), (spine_x, y))]

        return [
            ((left, y), (spine_x, y)),
            ((spine_x, y), (right, y)),
        ]

    def add_rock_images(
        self,
        fig: go.Figure,
        positions: dict[int, tuple[float, float]],
        image_by_id: dict[int, str],
    ) -> None:
        layout_images = []
        for rock_id, rock in self.helper.rocks.items():
            x, y = positions[rock_id]
            opacity = 0.45 if rock.status == genetics.RockStatus.SOLD else 1.0

            layout_images.append(
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

        fig.update_layout(images=layout_images)

    def add_node_text(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        highlighted = {int(rock_id) for rock_id in self.highlighted_rock_ids}
        rock_ids = []
        xs = []
        ys = []
        hover_texts = []
        colors = []
        symbols = []
        highlight_xs = []
        highlight_ys = []

        for rock_id, rock in self.helper.rocks.items():
            x, y = positions[int(rock_id)]
            rock_ids.append(int(rock_id))
            xs.append(x)
            ys.append(y)
            hover_texts.append(self.hover_text(rock))
            colors.append(get_gender_color(rock))
            symbols.append("circle" if rock.sex == genetics.Sex.MALE else "diamond")
            if int(rock_id) in highlighted:
                highlight_xs.append(x)
                highlight_ys.append(y)

        if rock_ids:
            if self.render_images:
                marker = {
                    "size": 28,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "rgba(0,0,0,0)", "width": 0},
                }
            else:
                marker = {
                    "size": 15,
                    "color": colors,
                    "symbol": symbols,
                    "opacity": 0.9,
                    "line": {"color": "#4B4038", "width": 1},
                }

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=marker,
                    customdata=rock_ids,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        if highlight_xs:
            fig.add_trace(
                go.Scatter(
                    x=highlight_xs,
                    y=highlight_ys,
                    mode="markers",
                    marker={
                        "size": 36,
                        "color": "rgba(255,255,255,0.01)",
                        "line": {"color": "#8B5A2B", "width": 4},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def add_rock_badges(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        grouped_badges: dict[tuple[str, str], dict[str, list[float] | list[str]]] = {}
        for rock_id, badge in self.rock_badges.items():
            if int(rock_id) not in positions:
                continue

            x, y = positions[int(rock_id)]
            text = badge.get("text", "\u2665")
            color = badge.get("color", "#E15759")
            group = grouped_badges.setdefault((text, color), {"x": [], "y": [], "text": []})
            group["x"].append(x - 0.42 * self.rock_image_size)
            group["y"].append(y + 0.42 * self.rock_image_size)
            group["text"].append(text)

        for (text, color), group in grouped_badges.items():
            fig.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["y"],
                    mode="text",
                    text=group["text"],
                    textfont={
                        "size": 24,
                        "color": color,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def add_tree_checkboxes(self, fig: go.Figure, positions: dict[int, tuple[float, float]]) -> None:
        checkbox_ids = {int(rock_id) for rock_id in self.tree_checkbox_ids}
        checked_ids = {int(rock_id) for rock_id in self.tree_checked_ids}
        if not checkbox_ids:
            return

        unchecked_xs = []
        unchecked_ys = []
        checked_xs = []
        checked_ys = []

        for rock_id in sorted(checkbox_ids):
            if rock_id not in positions:
                continue

            x, y = positions[rock_id]
            target_xs, target_ys = (checked_xs, checked_ys) if rock_id in checked_ids else (unchecked_xs, unchecked_ys)
            target_xs.append(x)
            target_ys.append(y - 0.76 * self.rock_image_size)

        if unchecked_xs:
            fig.add_trace(
                go.Scatter(
                    x=unchecked_xs,
                    y=unchecked_ys,
                    mode="text",
                    text=["\u2610"] * len(unchecked_xs),
                    textfont={
                        "size": 24,
                        "color": "#6F6258",
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if checked_xs:
            fig.add_trace(
                go.Scatter(
                    x=checked_xs,
                    y=checked_ys,
                    mode="text",
                    text=["\u2611"] * len(checked_xs),
                    textfont={
                        "size": 24,
                        "color": "#8B5A2B",
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    @staticmethod
    def hover_text(rock: genetics.Rock) -> str:
        parents = ", ".join(str(parent_id) for parent_id in rock.parent_ids) or "founder"
        phenotype_lines = []
        for gene_name in genetics.GENE_SPECS:
            if rock.genotype.genes[gene_name].phenotype != "n/a": # CHANGE TO TRY AND REMOVE N/A IN PHENOTYPES IN HOVER TEXT
                gene_pair = rock.genotype.genes.get(gene_name)
                phenotype = "n/a" if gene_pair is None or gene_pair.phenotype is None else str(gene_pair.phenotype)
                if gene_name == "eye_color":
                    if rock.genotype.genes["eyes"].phenotype == "n/a":
                        phenotype = "n/a"
                if gene_name in ["hair_color", "hair_texture"]:
                    if rock.genotype.genes["hair"].phenotype == "n/a" and rock.genotype.genes["brows"].phenotype == "n/a" and rock.genotype.genes["facial_hair"].phenotype == "n/a":
                        phenotype = "n/a"
                phenotype_lines.append(f"{html.escape(gene_name)}: {html.escape(phenotype)}")

        return (
            f"id: {int(rock.id)}<br>"
            f"name: {html.escape(rock_display_name(rock))}<br>"
            f"generation: {int(rock.generation)}<br>"
            f"parents: {html.escape(parents)}<br>"
            f"status: {html.escape(rock.status.value)}<br>"
            f"value: ${int(rock.value)}<br>"
            f"sell value: ${int(rock.sell_value)}<br>"
            f"phenotypes:<br>"
            + "<br>".join(phenotype_lines)
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
