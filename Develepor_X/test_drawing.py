import base64

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from Rock_Drawing.rock_draw_machine import DrawMachine, draw_rock
from Rock_Drawing.rock_drawing_helper import rock_to_image_uri
from Rock_Drawing.rock_render_context import RockRenderContext

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
