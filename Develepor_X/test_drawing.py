import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from Rock_Drawing.rock_draw_machine import DrawMachine, IdentityLabelLayout, draw_rock
from Rock_Drawing.rock_drawing_helper import render_game_rock_images, rock_to_image_uri
from Rock_Drawing.rock_lineage_drawing_helper import TreeDrawer, TreeHelper, draw_game_tree, show_rocks
from Rock_Drawing.rock_render_context import RockRenderContext
from Rock_GameState.rock_game_state_helper import GameMaster

from conftest import make_rock
import Rock_Genetics.rock_genetic_helper as genetics


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
        assert ax.texts[-1].get_color() == "royalblue"
        assert ax.get_aspect() == 1.0
        assert not ax.axison
    finally:
        plt.close(fig)


def test_draw_rock_returns_axis_and_adds_body_patch():
    fig, ax = plt.subplots()
    rock = make_rock()
    layout = IdentityLabelLayout(name_below_body_radius=0.75, gender_right_body_radius=0.75)

    try:
        result_ax = draw_rock(rock, ax=ax, identity_layout=layout)

        assert result_ax is ax
        assert len(ax.patches) >= 1
        assert ax.texts[-1].get_position()[1] < 0
    finally:
        plt.close(fig)


def test_identity_layout_formats_structured_names_and_status_symbol():
    fig, ax = plt.subplots()
    rock = make_rock()
    rock.name = genetics.RockName(
        honorific="Lady",
        given="Pebble",
        family="Moonstone",
        epithet="the Bright",
    )
    rock.change_status(genetics.RockStatus.SOLD)

    try:
        DrawMachine(rock=rock, ax=ax).draw()

        text_values = [text.get_text() for text in ax.texts]
        assert "$" in text_values
        assert any("Lady\nPebble Moonstone\nthe Bright" in value for value in text_values)
    finally:
        plt.close(fig)


def test_identity_layout_normalizes_status_symbols():
    cases = [
        (genetics.RockStatus.BRED, {}, "o", "gray"),
        (genetics.RockStatus.DEAD, {}, "\u271d", "black"),
        (genetics.RockStatus.CRAISENED, {}, "x", "crimson"),
        (genetics.RockStatus.SOLD, {}, "$", "green"),
        (genetics.RockStatus.ACTIVE, {"puffed": True}, "p", "royalblue"),
        (genetics.RockStatus.ACTIVE, {"is_market": True}, "I", "darkorange"),
    ]

    for status, attrs, symbol, color in cases:
        rock = make_rock(status=status)
        for attr_name, attr_value in attrs.items():
            setattr(rock, attr_name, attr_value)

        machine = DrawMachine(rock=rock)

        assert machine.get_status_symbol_and_color() == (symbol, color)


def test_identity_layout_places_status_top_left_and_gender_top_right():
    fig, ax = plt.subplots()
    rock = make_rock(status=genetics.RockStatus.SOLD)

    try:
        machine = DrawMachine(rock=rock, ax=ax)
        machine.draw()

        gender_text = next(text for text in ax.texts if text.get_text() in {"\u2642", "\u2640"})
        status_text = next(text for text in ax.texts if text.get_text() == "$")

        assert status_text.get_position()[0] < machine.ctx.xmin
        assert status_text.get_position()[1] > machine.ctx.ymax
        assert gender_text.get_position()[0] > machine.ctx.xmax
        assert gender_text.get_position()[1] > machine.ctx.ymax
    finally:
        plt.close(fig)


def test_rock_to_image_uri_returns_png_data_uri():
    rock = make_rock()

    uri = rock_to_image_uri(rock, sprite_size=1.0, dpi=80)

    assert uri.startswith("data:image/png;base64,")
    encoded = uri.split(",", 1)[1]
    png_bytes = base64.b64decode(encoded)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_rock_to_image_uri_uses_stable_canvas_for_long_names():
    short_name_rock = make_rock(rock_id=1)
    long_name_rock = make_rock(rock_id=2)
    long_name_rock.name = genetics.RockName(
        honorific="Archduchess",
        given="Pebblewithaverylonggivenname",
        family="Moonstonewithaverylongfamilyname",
        epithet="the Particularly Bright",
    )

    short_uri = rock_to_image_uri(short_name_rock, sprite_size=1.0, dpi=80)
    long_uri = rock_to_image_uri(long_name_rock, sprite_size=1.0, dpi=80)

    short_image = Image.open(io.BytesIO(base64.b64decode(short_uri.split(",", 1)[1])))
    long_image = Image.open(io.BytesIO(base64.b64decode(long_uri.split(",", 1)[1])))

    assert short_image.size == long_image.size


def test_render_game_rock_images_returns_id_to_uri_cache():
    game = GameMaster(seed=61)

    images = render_game_rock_images(game, sprite_size=0.8, dpi=60)

    assert set(images) == set(game.rocks)
    assert all(uri.startswith("data:image/png;base64,") for uri in images.values())
    assert set(game.rock_image_cache) == set(game.rocks)


def test_render_game_rock_images_cache_tracks_market_status():
    game = GameMaster(seed=610)
    rock = next(iter(game.rocks.values()))

    render_game_rock_images(game, sprite_size=0.8, dpi=60)
    first_signature = game.rock_image_cache[rock.id]["signature"]

    rock.is_market = True
    render_game_rock_images(game, sprite_size=0.8, dpi=60)

    assert game.rock_image_cache[rock.id]["signature"] != first_signature


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


def test_tree_helper_assigns_distinct_family_styles():
    game = GameMaster(seed=65, starting_money=80)
    males = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE]

    game.add_pair_to_queue(males[0].id, females[0].id)
    game.add_pair_to_queue(males[1].id, females[1].id)
    game.advance_generation()

    helper = TreeHelper.from_game(game)
    styles = helper.family_styles()

    assert len(styles) >= 2
    assert len({style["color"] for style in styles.values()}) >= 2
    assert all("dash" in style for style in styles.values())


def test_tree_drawer_creates_plotly_figure_with_images_without_duplicate_labels():
    game = GameMaster(seed=63)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    fig = TreeDrawer(game=game, canvas_width=600, canvas_height=400).draw()

    assert len(fig.layout.images) == len(game.rocks)
    assert len(fig.data) >= len(game.rocks)
    assert fig.layout.dragmode == "pan"
    visible_texts = [
        text
        for trace in fig.data
        if getattr(trace, "mode", None) == "text"
        for text in (trace.text or [])
    ]
    assert not any("#" in str(text) for text in visible_texts)
    assert not any("\u2642" in str(text) or "\u2640" in str(text) for text in visible_texts)


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
