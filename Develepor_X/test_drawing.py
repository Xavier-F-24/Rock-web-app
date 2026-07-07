import base64

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from Rock_Drawing.rock_draw_machine import DrawMachine, draw_rock
from Rock_Drawing.rock_drawing_helper import render_game_rock_images, rock_to_image_uri
from Rock_Drawing.rock_lineage_drawing_helper import TreeDrawer, TreeHelper, draw_game_tree, show_rocks
from Rock_Drawing.rock_render_context import RockRenderContext
from Rock_GameState.rock_game_state_helper import GameMaster

from conftest import make_rock


def test_render_context_from_rock_creates_body_and_layout_data():
    fig, ax = plt.subplots()
    rock = make_rock(gene_overrides={"eyes": (1, 1), "mouths": (1, 1)})

    try:
        ctx = RockRenderContext.from_rock(rock=rock, ax=ax)

        assert ctx.rock is rock
        assert ctx.body is not None
        assert ctx.body_points.shape[1] == 2
        assert ctx.size_scale > 0
        assert ctx.body_color is not None
        assert ctx.presence.eyes is True
        assert ctx.presence.mouth is True
        assert ctx.face_layout.has("eyes")
        assert ctx.face_layout.has("mouth")
    finally:
        plt.close(fig)


def test_draw_machine_draw_returns_axis_and_adds_artists():
    fig, ax = plt.subplots()
    rock = make_rock(gene_overrides={"eyes": (1, 1), "mouths": (1, 1)})

    try:
        result_ax = DrawMachine(rock=rock, ax=ax).draw()

        assert result_ax is ax
        assert len(ax.patches) > 0
        assert len(ax.texts) >= 2
        assert ax.get_aspect() == 1.0
        assert not ax.axison
    finally:
        plt.close(fig)


def test_draw_rock_returns_axis_and_adds_body_patch():
    fig, ax = plt.subplots()
    rock = make_rock()

    try:
        result_ax = draw_rock(rock, ax=ax)

        assert result_ax is ax
        assert len(ax.patches) >= 1
    finally:
        plt.close(fig)


def test_rock_to_image_uri_returns_png_data_uri():
    rock = make_rock()

    uri = rock_to_image_uri(rock, sprite_size=1.0, dpi=80)

    assert uri.startswith("data:image/png;base64,")
    encoded = uri.split(",", 1)[1]
    png_bytes = base64.b64decode(encoded)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_game_rock_images_returns_id_to_uri_cache():
    game = GameMaster(seed=61)

    images = render_game_rock_images(game, sprite_size=0.8, dpi=60)

    assert set(images) == set(game.rocks)
    assert all(uri.startswith("data:image/png;base64,") for uri in images.values())
    assert set(game.rock_image_cache) == set(game.rocks)


def test_tree_helper_computes_positions_and_family_links():
    game = GameMaster(seed=62)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    helper = TreeHelper.from_game(game)
    positions = helper.compute_positions()

    assert set(positions) == set(game.rocks)
    assert helper.family_links()
    assert helper.bounds()["x"][0] < helper.bounds()["x"][1]


def test_tree_drawer_creates_plotly_figure_with_images_and_labels():
    game = GameMaster(seed=63)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    fig = TreeDrawer(game=game, canvas_width=600, canvas_height=400).draw()

    assert len(fig.layout.images) == len(game.rocks)
    assert len(fig.data) >= len(game.rocks)
    assert fig.layout.dragmode == "pan"


def test_draw_game_tree_wrapper_returns_figure():
    game = GameMaster(seed=64)

    fig = draw_game_tree(game)

    assert len(fig.layout.images) == len(game.rocks)


def test_show_rocks_returns_matplotlib_grid():
    rocks = {1: make_rock(rock_id=1), 2: make_rock(rock_id=2)}

    fig, axes = show_rocks(rocks, cols=2)

    try:
        assert fig is not None
        assert len(axes) == 2
    finally:
        plt.close(fig)
