import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from Rock_Drawing.rock_draw_machine import DrawMachine, IdentityLabelLayout, draw_rock
from Rock_Drawing.rock_drawing_helper import get_render_variant_key, render_game_rock_images, rock_to_image_uri
from Rock_Drawing.rock_lineage_drawing_helper import (
    FAMILY_LINE_COLORS,
    TreeDrawer,
    TreeHelper,
    draw_game_tree,
    show_rocks,
    shuffled_family_colors,
)
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


def test_identity_layout_wraps_long_structured_name_lines():
    rock = make_rock()
    rock.name = genetics.RockName(
        honorific="Archduchess",
        given="Pebblewithaverylonggivenname",
        family="Moonstonewithaverylongfamilyname",
        epithet="the Particularly Bright and Surprisingly Wide",
    )
    machine = DrawMachine(rock=rock)

    formatted_name = machine.format_rock_name()

    assert "Archduchess" in formatted_name
    assert len(formatted_name.splitlines()) > 3
    assert all(len(line) <= 24 for line in formatted_name.splitlines() if line != "Archduchess")


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


def test_craisen_status_uses_identity_badge_without_body_overlay():
    fig, ax = plt.subplots()
    rock = make_rock(status=genetics.RockStatus.CRAISENED)

    try:
        DrawMachine(rock=rock, ax=ax).draw()

        text_values = [text.get_text() for text in ax.texts]
        assert "x" in text_values
        assert "CRAISEN" not in text_values
    finally:
        plt.close(fig)


def test_identity_layout_places_status_top_left_and_gender_top_right():
    fig, ax = plt.subplots()
    rock = make_rock(status=genetics.RockStatus.SOLD)

    try:
        machine = DrawMachine(rock=rock, ax=ax)
        machine.draw()

        gender_text = next(text for text in ax.texts if text.get_text() in {"\u2642", "\u2640"})
        status_text = next(text for text in ax.texts if text.get_text() == "$")
        body_radius = 0.5 * machine.ctx.unit

        assert status_text.get_position()[0] < machine.ctx.xmin
        assert status_text.get_position()[1] > machine.ctx.ymax
        assert gender_text.get_position()[0] > machine.ctx.xmax
        assert gender_text.get_position()[1] > machine.ctx.ymax
        assert status_text.get_position()[0] <= machine.ctx.xmin - 0.95 * body_radius
        assert gender_text.get_position()[0] >= machine.ctx.xmax + 0.95 * body_radius
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
    assert all(get_render_variant_key(sprite_size=0.8, dpi=60) in variants for variants in game.rock_image_cache.values())


def test_render_game_rock_images_cache_tracks_market_status():
    game = GameMaster(seed=610)
    rock = next(iter(game.rocks.values()))

    render_game_rock_images(game, sprite_size=0.8, dpi=60)
    variant_key = get_render_variant_key(sprite_size=0.8, dpi=60)
    first_signature = game.rock_image_cache[rock.id][variant_key]["signature"]

    rock.is_market = True
    render_game_rock_images(game, sprite_size=0.8, dpi=60)

    assert game.rock_image_cache[rock.id][variant_key]["signature"] != first_signature


def test_render_game_rock_images_caches_render_variants_separately():
    game = GameMaster(seed=611)

    small_images = render_game_rock_images(game, sprite_size=0.8, dpi=40)
    large_images = render_game_rock_images(game, sprite_size=1.2, dpi=90)

    small_key = get_render_variant_key(sprite_size=0.8, dpi=40)
    large_key = get_render_variant_key(sprite_size=1.2, dpi=90)

    assert small_key != large_key
    assert all(small_key in variants and large_key in variants for variants in game.rock_image_cache.values())
    assert any(small_images[rock_id] != large_images[rock_id] for rock_id in game.rocks)


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


def test_tree_helper_accounts_for_rock_occupied_bounds():
    game = GameMaster(seed=621)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    children = game.advance_generation()

    helper = TreeHelper.from_game(
        game,
        node_width=2.0,
        node_height=2.0,
        node_margin_x=0.6,
        branch_clearance=0.4,
        route_fudge=0.5,
    )
    positions = helper.compute_positions()
    parent_ids = tuple(sorted(children[0].parent_ids))
    child_ids = helper.family_groups()[parent_ids]
    branch_y = helper.child_branch_y(parent_ids, child_ids, positions)
    child_top = max(helper.occupied_bounds(child_id, positions)["top"] for child_id in child_ids)
    parent_bottom = min(helper.occupied_bounds(parent_id, positions)["bottom"] for parent_id in parent_ids)
    first_generation_ids = helper.generation_groups()[0]
    first_gap = abs(positions[first_generation_ids[1]][0] - positions[first_generation_ids[0]][0])
    first_bounds = helper.occupied_bounds(first_generation_ids[0], positions)

    assert first_gap >= helper.required_x_gap()
    assert child_top < branch_y < parent_bottom
    assert first_bounds["right"] - first_bounds["left"] == 3.0


def test_tree_helper_routes_orthogonal_segments_around_obstacles():
    obstacle = make_rock(rock_id=1)
    helper = TreeHelper(
        rocks={1: obstacle},
        node_width=2.0,
        node_height=2.0,
        branch_clearance=0.25,
        route_fudge=0.25,
    )
    positions = {1: (0.0, 0.0)}

    segments = helper.route_orthogonal(
        start=(-3.0, 0.0),
        end=(3.0, 0.0),
        positions=positions,
        ignore_ids=set(),
    )

    assert len(segments) > 1
    assert all(
        not helper.segment_hits_obstacle(start, end, positions, ignore_ids=set())
        for start, end in segments
    )


def test_shuffled_family_colors_picks_three_distinct_non_red_colors():
    colors = shuffled_family_colors(3, seed=5)

    assert len(colors) == 3
    assert len(set(colors)) == 3
    assert not any(color.lower() in {"#e15759", "#ff9da7", "#d45087", "#f95d6a"} for color in colors)


def test_tree_helper_assigns_distinct_family_styles():
    game = GameMaster(seed=65, starting_money=80)
    males = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE]

    game.add_pair_to_queue(males[0].id, females[0].id)
    game.add_pair_to_queue(males[1].id, females[1].id)
    game.advance_generation()

    helper = TreeHelper.from_game(game)
    styles = helper.family_styles(seed=8)

    assert len(styles) >= 2
    assert len({style["color"] for style in styles.values()}) >= 2
    assert all("color" in style and "dash" in style for style in styles.values())
    assert all(style["dash"] == "solid" for style in styles.values())


def test_tree_helper_uses_import_lane_for_market_founders_without_changing_game_generation():
    game = GameMaster(seed=69, starting_money=80)
    imported = game.buy_random_rock()

    helper = TreeHelper.from_game(game)
    graph_generations = helper.graph_generations()
    groups = helper.generation_groups()

    assert imported.generation == game.generation
    assert graph_generations[imported.id] == -1
    assert imported.id in groups[-1]


def test_tree_helper_uses_current_generation_minus_one_for_later_imports():
    game = GameMaster(seed=71, starting_money=80)
    males = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE]
    game.add_pair_to_queue(males[0].id, females[0].id)
    game.advance_generation()

    imported = game.buy_random_rock()
    helper = TreeHelper.from_game(game)
    graph_generations = helper.graph_generations()

    assert game.generation == 1
    assert imported.generation == game.generation
    assert graph_generations[imported.id] == game.generation - 1


def test_tree_helper_visually_places_market_pod_parents_above_kept_child():
    game = GameMaster(seed=70, starting_money=80)
    offer_id = game.market_pods[0].offer_id

    pending = game.market_manager.buy_market_pod(game, offer_id)
    child = game.market_manager.choose_market_pod_child(game, 0)
    helper = TreeHelper.from_game(game)
    graph_generations = helper.graph_generations()

    assert child.generation == game.generation
    assert all(game.get_rock(parent_id).generation == game.generation for parent_id in child.parent_ids)
    assert all(graph_generations[parent_id] == -1 for parent_id in child.parent_ids)
    assert graph_generations[child.id] == 0
    assert len(pending.children) > 0


def test_tree_helper_can_opt_into_dash_styles():
    game = GameMaster(seed=66, starting_money=80)
    males = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.MALE]
    females = [rock for rock in game.rocks.values() if rock.sex == genetics.Sex.FEMALE]

    game.add_pair_to_queue(males[0].id, females[0].id)
    game.add_pair_to_queue(males[1].id, females[1].id)
    game.advance_generation()

    helper = TreeHelper.from_game(game)
    styles = helper.family_styles(seed=8, use_dash_styles=True)

    assert any(style["dash"] != "solid" for style in styles.values())


def test_tree_drawer_horizontal_spans_meet_at_route_spine():
    inside_segments = TreeDrawer.horizontal_spans_to_spine([0.0, 4.0], 2.0, -1.0)
    outside_segments = TreeDrawer.horizontal_spans_to_spine([0.0, 4.0], 6.0, -1.0)

    assert inside_segments == [((0.0, -1.0), (2.0, -1.0)), ((2.0, -1.0), (4.0, -1.0))]
    assert outside_segments == [((0.0, -1.0), (6.0, -1.0))]


def test_tree_drawer_parent_pair_connector_uses_only_parent_anchors():
    connector = TreeDrawer.parent_pair_connector_segments(
        parent_a_pos=(6.0, 1.0),
        parent_b_pos=(2.0, 1.0),
        parent_connector_y=-0.5,
        child_connector_y=-2.0,
    )

    assert connector["horizontal"] == ((2.0, -0.5), (6.0, -0.5))
    assert connector["vertical_drop"] == ((4.0, -0.5), (4.0, -2.0))
    assert connector["midpoint"] == (4.0, -0.5)
    assert connector["endpoints"] == [(2.0, -0.5), (6.0, -0.5)]


def test_parent_pair_connector_can_route_c_shape_around_blocking_rock():
    blocking_rock = make_rock(rock_id=1)
    helper = TreeHelper(
        rocks={1: blocking_rock},
        node_width=2.0,
        node_height=2.0,
        branch_clearance=0.25,
        route_fudge=0.25,
    )
    positions = {1: (0.0, -0.5)}
    connector = TreeDrawer.parent_pair_connector_segments(
        parent_a_pos=(-3.0, 1.0),
        parent_b_pos=(3.0, 1.0),
        parent_connector_y=-0.5,
        child_connector_y=-2.0,
    )

    routed = helper.route_orthogonal(
        start=connector["horizontal"][0],
        end=connector["horizontal"][1],
        positions=positions,
        ignore_ids=set(),
    )

    assert len(routed) > 1
    assert routed[0][0] == (-3.0, -0.5)
    assert routed[-1][1] == (3.0, -0.5)
    assert all(
        not helper.segment_hits_obstacle(start, end, positions, ignore_ids=set())
        for start, end in routed
    )


def test_tree_drawer_debug_connector_markers_are_optional():
    game = GameMaster(seed=68)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    plain_fig = TreeDrawer(game=game, canvas_width=600, canvas_height=400).draw()
    debug_fig = TreeDrawer(game=game, canvas_width=600, canvas_height=400, debug_connectors=True).draw()

    assert len(debug_fig.data) > len(plain_fig.data)
    assert any(getattr(trace, "mode", None) == "markers" for trace in debug_fig.data)


def test_tree_drawer_creates_plotly_figure_with_images_without_duplicate_labels():
    game = GameMaster(seed=63)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    fig = TreeDrawer(game=game, canvas_width=600, canvas_height=400).draw()

    assert len(fig.layout.images) == len(game.rocks)
    line_traces = [trace for trace in fig.data if getattr(trace, "mode", None) == "lines"]
    assert line_traces
    assert all(trace.line.dash == "solid" for trace in line_traces)
    assert fig.layout.dragmode == "pan"
    visible_texts = [
        text
        for trace in fig.data
        if getattr(trace, "mode", None) == "text"
        for text in (trace.text or [])
    ]
    assert not any("#" in str(text) for text in visible_texts)
    assert not any("\u2642" in str(text) or "\u2640" in str(text) for text in visible_texts)


def test_tree_drawer_caches_routed_geometry_on_game_runtime_state():
    game = GameMaster(seed=630)
    ids = list(game.rocks)
    game.add_pair_to_queue(ids[0], ids[1])
    game.advance_generation()

    TreeDrawer(game=game, canvas_width=600, canvas_height=400).draw()
    cache = getattr(game, "lineage_geometry_cache")
    cache_keys = set(cache)

    assert cache
    assert all("positions" in entry and "family_line_specs" in entry for entry in cache.values())

    TreeDrawer(
        game=game,
        canvas_width=600,
        canvas_height=400,
        highlighted_rock_ids=(ids[0],),
        tree_checkbox_ids=tuple(game.rocks),
    ).draw()

    assert set(game.lineage_geometry_cache) == cache_keys


def test_tree_drawer_large_tree_fast_route_skips_obstacle_router(monkeypatch):
    game = GameMaster(seed=634)
    game.rocks.clear()
    game.next_rock_id = 1

    for index in range(75):
        rock = make_rock(rock_id=game.reserve_rock_id())
        rock.generation = 0 if rock.id <= 2 else 1
        rock.parent_ids = [] if rock.id <= 2 else [1, 2]
        game.rocks[rock.id] = rock

    def fail_route(*args, **kwargs):
        raise AssertionError("large-tree fast route should not call obstacle router")

    monkeypatch.setattr(TreeHelper, "route_orthogonal", fail_route)

    fig = TreeDrawer(game=game, obstacle_routing_threshold=70).draw()

    assert len(fig.layout.images) == len(game.rocks)
    assert any(getattr(trace, "mode", None) == "lines" for trace in fig.data)


def test_tree_drawer_uses_adaptive_image_resolution_for_large_trees():
    game = GameMaster(seed=635)
    game.rocks.clear()
    game.next_rock_id = 1

    for _ in range(160):
        rock = make_rock(rock_id=game.reserve_rock_id())
        game.rocks[rock.id] = rock

    drawer = TreeDrawer(game=game)

    assert drawer.image_render_settings() == (1.0, 70)
    assert TreeDrawer(game=game, adaptive_image_resolution=False).image_render_settings() == (1.4, 100)


def test_tree_drawer_fast_overview_skips_png_image_rendering(monkeypatch):
    game = GameMaster(seed=636)

    def fail_render_images(*args, **kwargs):
        raise AssertionError("fast overview should not render PNG rock images")

    monkeypatch.setattr(
        "Rock_Drawing.rock_lineage_drawing_helper.render_game_rock_images",
        fail_render_images,
    )

    fig = TreeDrawer(game=game, render_images=False).draw()
    marker_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "mode", None) == "markers" and getattr(trace, "customdata", None) is not None
    ]

    assert len(fig.layout.images) == 0
    assert marker_traces
    assert marker_traces[0].marker.color


def test_tree_drawer_groups_family_line_traces_by_style():
    game = GameMaster(seed=637)
    game.rocks.clear()
    game.next_rock_id = 1

    for _ in range(90):
        rock = make_rock(rock_id=game.reserve_rock_id())
        rock.generation = 0 if rock.id <= 20 else 1
        rock.parent_ids = [] if rock.id <= 20 else [1 + (rock.id % 20), 1 + ((rock.id + 7) % 20)]
        game.rocks[rock.id] = rock

    fig = TreeDrawer(game=game, render_images=False, obstacle_routing_threshold=10).draw()
    line_traces = [trace for trace in fig.data if getattr(trace, "mode", None) == "lines"]

    assert len(TreeHelper.from_game(game).family_groups()) > len(FAMILY_LINE_COLORS)
    assert line_traces
    assert len(line_traces) <= len(FAMILY_LINE_COLORS)


def test_tree_hover_text_includes_requested_fields_and_phenotypes():
    rock = make_rock(rock_id=77)
    rock.parent_ids = [1, 2]
    rock.generation = 3
    rock.value = 12
    rock.sell_value = 7

    hover = TreeDrawer.hover_text(rock)

    expected_order = [
        "id: 77",
        "name:",
        "generation: 3",
        "parents: 1, 2",
        "status:",
        "value: $12",
        "sell value: $7",
        "phenotypes:",
        "eyes:",
    ]
    positions = [hover.index(text) for text in expected_order]

    assert positions == sorted(positions)


def test_tree_drawer_node_markers_expose_clickable_rock_ids_and_highlights():
    game = GameMaster(seed=631)
    highlighted_id = next(iter(game.rocks))

    fig = TreeDrawer(
        game=game,
        canvas_width=600,
        canvas_height=400,
        highlighted_rock_ids=(highlighted_id,),
    ).draw()

    marker_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "mode", None) == "markers" and getattr(trace, "customdata", None) is not None
    ]
    highlight_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "mode", None) == "markers"
        and getattr(trace, "customdata", None) is None
        and getattr(trace.marker.line, "color", None) == "#8B5A2B"
    ]
    customdata_ids = {
        int(rock_id)
        for trace in marker_traces
        for rock_id in trace.customdata
    }

    assert customdata_ids == set(game.rocks)
    assert highlight_traces
    assert highlight_traces[0].marker.line.color == "#8B5A2B"
    assert highlight_traces[0].marker.line.width == 4


def test_tree_drawer_renders_queued_parent_badges():
    game = GameMaster(seed=632)
    ids = list(game.rocks)
    badge_id = ids[0]

    fig = TreeDrawer(
        game=game,
        canvas_width=600,
        canvas_height=400,
        rock_badges={badge_id: {"text": "\u2665", "color": "#E15759"}},
    ).draw()

    badge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "mode", None) == "text" and "\u2665" in list(trace.text or [])
    ]

    assert badge_traces
    assert badge_traces[0].textfont.color == "#E15759"


def test_tree_drawer_renders_tree_checkbox_marks():
    game = GameMaster(seed=633)
    ids = list(game.rocks)
    unchecked_id, checked_id = ids[:2]

    fig = TreeDrawer(
        game=game,
        canvas_width=600,
        canvas_height=400,
        tree_checkbox_ids=(unchecked_id, checked_id),
        tree_checked_ids=(checked_id,),
    ).draw()

    checkbox_texts = [
        text
        for trace in fig.data
        if getattr(trace, "mode", None) == "text"
        for text in list(trace.text or [])
        if text in {"\u2610", "\u2611"}
    ]

    assert "\u2610" in checkbox_texts
    assert "\u2611" in checkbox_texts


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
