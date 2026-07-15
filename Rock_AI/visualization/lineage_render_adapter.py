"""Thin adapter over the existing lineage renderer."""

from __future__ import annotations


def build_lineage_figure(
    game: object,
    *,
    selected_parent_ids=(),
    new_child_ids=(),
    mutation_rock_ids=(),
    rare_rock_ids=(),
    render_images: bool = True,
    canvas_height: int = 600,
):
    from Rock_Drawing.rock_lineage_drawing_helper import TreeDrawer

    badges = {}
    for rock_id in new_child_ids:
        badges[int(rock_id)] = {"symbol": "+", "color": "#2f855a"}
    for rock_id in mutation_rock_ids:
        badges[int(rock_id)] = {"symbol": "M", "color": "#c2415d"}
    for rock_id in rare_rock_ids:
        badges.setdefault(int(rock_id), {"symbol": "R", "color": "#b7791f"})
    return TreeDrawer(
        game,
        canvas_width=1100,
        canvas_height=canvas_height,
        highlighted_rock_ids=tuple(int(value) for value in selected_parent_ids),
        rock_badges=badges,
        render_images=render_images,
    ).draw()
