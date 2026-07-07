

# -----------------------------
# SHOWING SOME ROCK
# -----------------------------

def show_rocks(
    rock_items,
    rock_source=None,
    cols=6,
    figsize_per_rock=3.2,
    show_genes=False,
    show_traits=False,
    title=None,
    sort_by_generation=False,
    normalize_size=True
):
    """
    Display a grid of rocks.

    Accepts:
    - a dictionary of rocks: show_rocks(rocks)
    - a list of rock IDs: show_rocks([1, 2, 3], rock_source=rocks)
    - a list of Rock objects: show_rocks([rock1, rock2])
    - a dictionary of test rocks: show_rocks(test_rocks)

    Parameters
    ----------
    rock_items:
        Dict[int, Rock], list[int], tuple[int], list[Rock], or tuple[Rock]

    rock_source:
        Optional dictionary used when rock_items is a list of IDs.
        If None, the function tries to use the global `rocks`.

    cols:
        Number of columns in the display grid.

    figsize_per_rock:
        Size multiplier for each rock subplot.

    show_genes:
        Passes show_genes=True into draw_rock.

    show_traits:
        Adds a compact trait label under each rock.

    title:
        Optional figure title.

    sort_by_generation:
        If True, sorts rocks by generation, then ID.
    """

    # -----------------------------
    # Resolve input into Rock objects
    # -----------------------------

    if isinstance(rock_items, dict):
        rock_list = list(rock_items.values())

    else:
        rock_list = []

        for item in list(rock_items):
            if isinstance(item, Rock):
                rock_list.append(item)

            elif isinstance(item, int):
                source = rock_source

                if source is None:
                    try:
                        source = rocks
                    except NameError:
                        raise ValueError(
                            "You passed rock IDs, but no rock_source was provided "
                            "and no global `rocks` dictionary exists."
                        )

                if item not in source:
                    raise KeyError(f"Rock ID {item} was not found in the provided rock source.")

                rock_list.append(source[item])

            else:
                raise TypeError(
                    "show_rocks expects a dict of rocks, a list of Rock objects, "
                    "or a list of integer rock IDs."
                )

    if sort_by_generation:
        rock_list = sorted(rock_list, key=lambda r: (r.generation, r.id))

    n = len(rock_list)

    if n == 0:
        print("No rocks to show.")
        return None, None

    # -----------------------------
    # Create grid
    # -----------------------------

    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_rock, rows * figsize_per_rock)
    )

    axes = np.array(axes).reshape(-1)

    # Turn all axes off first.
    for ax in axes:
        ax.axis("off")

    # -----------------------------
    # Draw rocks
    # -----------------------------

    for ax, rock in zip(axes, rock_list):
        draw_rock(rock, ax=ax, show_genes=show_genes, normalize_size=normalize_size)

        pad_rock_axis(ax, pad_frac=PAD_FRAC)

        if show_traits:
            v = get_visual_phenotype(rock)

            trait_text = (
                f"{v.get('shape', 'n/a')} | {v.get('size', 'n/a')} | {v.get('color', 'n/a')}\n"
                f"eyes: {v.get('eyes', 'n/a')} | hair: {v.get('hair', 'n/a')} | {v.get('hair_color', 'n/a')}"
            )

            if v.get("is_craisen", False):
                trait_text += "\nCRAISEN"

            ax.text(
                0.5,
                -0.08,
                trait_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8
            )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

    return fig, axes

# -----------------------------
# COMPUTING LAYOUT FOR ROCK TREE
# -----------------------------

def compute_lineage_positions_christmas(
    rocks,
    x_gap=5,
    gen_gap=5,
    parent_pair_gap=3,
    min_node_gap=3,
    anti_overlap=True,
    layout_passes=4
):
    """
    Robust Christmas-tree lineage layout.

    Goals:
    - all rocks get positions
    - parents are pulled toward their children
    - missing/orphan parent links do not crash the tree
    - sold/dead/puffed/spore clones can remain visible
    - works even when weird generation relationships appear

    Returns:
        pos = {rock_id: (x, y)}
    """

    if rocks is None or len(rocks) == 0:
        return {}

    # Make sure IDs are normal ints when possible.
    rock_ids = list(rocks.keys())

    # --------------------------------------------------
    # Group rocks by generation
    # --------------------------------------------------
    by_generation = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        by_generation.setdefault(gen, []).append(rid)

    generations = sorted(by_generation.keys())

    # --------------------------------------------------
    # Initial centered layout within each generation
    # --------------------------------------------------
    pos_x = {}

    for gen in generations:
        ids = sorted(by_generation[gen])
        n = len(ids)

        if n == 1:
            pos_x[ids[0]] = 0.0
        else:
            start_x = -0.5 * (n - 1) * x_gap

            for i, rid in enumerate(ids):
                pos_x[rid] = start_x + i * x_gap

    # Ensure every rock has an x position.
    for rid in rock_ids:
        if rid not in pos_x:
            pos_x[rid] = 0.0

    # --------------------------------------------------
    # Pull parents toward their children, bottom-up
    # --------------------------------------------------
    for _ in range(layout_passes):
        desired_x = {rid: [] for rid in rock_ids}

        # For each child, request parent positions around child center.
        for child_id, child in rocks.items():
            parents = getattr(child, "parents", None)

            if parents is None:
                continue

            if len(parents) != 2:
                continue

            p1, p2 = parents

            # Defensive skip: parent might not exist in current tree dictionary.
            if p1 not in rocks or p2 not in rocks:
                continue

            if child_id not in pos_x:
                continue

            child_center_x = pos_x[child_id]

            # Make sure keys exist even if the old data is odd.
            desired_x.setdefault(p1, [])
            desired_x.setdefault(p2, [])

            desired_x[p1].append(child_center_x - parent_pair_gap / 2)
            desired_x[p2].append(child_center_x + parent_pair_gap / 2)

        # Update positions from desired child-centered positions.
        # Work from older generations first to keep the tree stable.
        for gen in generations:
            for rid in by_generation[gen]:
                if rid in desired_x and len(desired_x[rid]) > 0:
                    old_x = pos_x.get(rid, 0.0)
                    target_x = sum(desired_x[rid]) / len(desired_x[rid])

                    # Blend instead of snapping to reduce wild oscillations.
                    pos_x[rid] = 0.45 * old_x + 0.55 * target_x

        # --------------------------------------------------
        # Anti-overlap pass within each generation
        # --------------------------------------------------
        if anti_overlap:
            for gen in generations:
                ids = sorted(by_generation[gen], key=lambda rid: pos_x.get(rid, 0.0))

                if len(ids) <= 1:
                    continue

                # Left-to-right push
                for i in range(1, len(ids)):
                    prev_id = ids[i - 1]
                    curr_id = ids[i]

                    if pos_x[curr_id] - pos_x[prev_id] < min_node_gap:
                        pos_x[curr_id] = pos_x[prev_id] + min_node_gap

                # Recenter the generation around zero-ish
                mean_x = sum(pos_x[rid] for rid in ids) / len(ids)

                for rid in ids:
                    pos_x[rid] -= mean_x

    # --------------------------------------------------
    # Final positions
    # --------------------------------------------------
    pos = {}

    for rid, rock in rocks.items():
        gen = getattr(rock, "generation", 0)

        if gen is None:
            gen = 0

        x = pos_x.get(rid, 0.0)
        y = -gen * gen_gap

        pos[rid] = (x, y)

    return pos

if px is not None:
    FAMILY_PALETTE = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Bold
    )
else:
    FAMILY_PALETTE = (
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#9C755F",
        "#FF9DA7",
        "#BAB0AC",
    )

def family_color(parent_pair):
    """
    Deterministic color for a parent pair.
    Same parent pair -> same color every time.
    """
    key = f"{min(parent_pair)}-{max(parent_pair)}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return FAMILY_PALETTE[h % len(FAMILY_PALETTE)]

FAMILY_COLORS = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#9C755F",  # brown
    "#FF9DA7",  # pink
    "#BAB0AC",  # gray
]

FAMILY_DASHES = [
    "solid",
    "dash",
    "dot",
    "longdash",
    "dashdot",
]

def get_family_styles(rocks):
    """
    Assign a stable color/dash style to each parent pair.
    """
    families = []

    for child_id, rock in rocks.items():
        if rock.parents is not None:
            families.append(tuple(sorted(rock.parents)))

    families = sorted(set(families))

    style_map = {}

    for i, fam in enumerate(families):
        style_map[fam] = {
            "color": FAMILY_COLORS[i % len(FAMILY_COLORS)],
            "dash": FAMILY_DASHES[(i // len(FAMILY_COLORS)) % len(FAMILY_DASHES)]
        }

    return style_map

def build_family_segments(pos, parent_pair, child_ids):
    """
    Builds pedigree line segments for one parent pair and its displayed children.
    """
    p1, p2 = parent_pair

    if p1 not in pos or p2 not in pos:
        return [], []

    child_ids = [cid for cid in child_ids if cid in pos]

    if len(child_ids) == 0:
        return [], []

    child_ids = sorted(child_ids, key=lambda cid: pos[cid][0])

    x1, y1 = pos[p1]
    x2, y2 = pos[p2]

    child_xs = [pos[cid][0] for cid in child_ids]
    child_ys = [pos[cid][1] for cid in child_ids]

    child_y = child_ys[0]
    parent_y = min(y1, y2)

    parent_bar_y = parent_y - 0.48
    sibling_bar_y = child_y + 0.68

    parent_center_x = (x1 + x2) / 2
    child_center_x = sum(child_xs) / len(child_xs)

    line_segments_x = []
    line_segments_y = []

    def add_segment(xa, ya, xb, yb):
        line_segments_x.extend([xa, xb, None])
        line_segments_y.extend([ya, yb, None])

    # Parent drops.
    add_segment(x1, y1 - 0.45, x1, parent_bar_y)
    add_segment(x2, y2 - 0.45, x2, parent_bar_y)

    # Parent pair bar.
    add_segment(x1, parent_bar_y, x2, parent_bar_y)

    # Descent toward children.
    add_segment(parent_center_x, parent_bar_y, child_center_x, sibling_bar_y)

    # Sibling bar and child drops.
    if len(child_ids) > 1:
        add_segment(min(child_xs), sibling_bar_y, max(child_xs), sibling_bar_y)

        for cx in child_xs:
            add_segment(cx, sibling_bar_y, cx, child_y + 0.45)
    else:
        cx = child_xs[0]
        add_segment(child_center_x, sibling_bar_y, cx, child_y + 0.45)

    return line_segments_x, line_segments_y

SIZE_SCALE_MAP = {
    "medium": 1.00,
    "large": 1.22,
    "small": 0.78,
    "giant": 1.55,
    "missized": 1.10,
}

def get_rock_size_scale(rock):
    """
    Returns the expressed visual size scale for a rock.
    """
    v = get_visual_phenotype(rock)
    return SIZE_SCALE_MAP.get(v.get("size", "medium"), 1.0)

def get_gender_symbol(rock):
    """
    Return display symbol for rock gender.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "♂"

    return "♀"

def get_gender_color(rock):
    """
    Display color for gender symbol.
    """
    gender = get_rock_gender_value(rock)

    if gender == 1:
        return "royalblue"

    return "deeppink"

# -----------------------------
# GETTING HOVER VALUES FOR ROCK TREE
# -----------------------------

def clean_hover_value(value):
    """
    Make phenotype values readable in Plotly hover text.
    """
    if isinstance(value, float):
        return f"{value:.3g}"

    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)

    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())

    return str(value)

def format_full_phenotype_hover(rock):
    """
    Build a full phenotype printout for hover boxes.
    """
    v = get_visual_phenotype(rock)

    lines = []

    for key in sorted(v.keys()):
        value = clean_hover_value(v[key])
        lines.append(f"{key}: {value}")

    return "<br>".join(lines)

def format_selected_phenotype_hover(rock):
    v = get_visual_phenotype(rock)

    preferred_keys = [
        "gender",
        "shape",
        "size",
        "color",
        "eyes",
        "eye_color",
        "mouths",
        "noses",
        "arms",
        "wings",
        "horns",
        "halos",
        "ears",
        "hair",
        "hair_color",
        "facial_hair",
        "wrinkles",
        "fuzz",
        "freckles",
        "stones",
        "tails",
        "splitting",
    ]

    lines = []

    for key in preferred_keys:
        if key in v:
            lines.append(f"{key}: {clean_hover_value(v[key])}")

    return "<br>".join(lines)

# -----------------------------
# DRAWING DA ROCK TREE
# -----------------------------

def normalize_parent_pair_for_tree(parent_pair):
    """
    Convert a parent-pair-like value into a clean (p1, p2) tuple.

    Returns None if the value is empty, malformed, or not exactly two ids.

    Handles:
        None
        ()
        []
        (1, 2)
        ["1", "2"]
        {"a": 1, "b": 2}
        Rock objects with .id
    """
    if parent_pair is None:
        return None

    if isinstance(parent_pair, dict):
        raw_values = list(parent_pair.values())
    elif isinstance(parent_pair, (list, tuple, set)):
        raw_values = list(parent_pair)
    else:
        raw_values = [parent_pair]

    cleaned = []

    for value in raw_values:
        if value is None:
            continue

        if hasattr(value, "id"):
            value = value.id

        try:
            cleaned.append(int(value))
        except Exception:
            continue

    if len(cleaned) != 2:
        return None

    p1, p2 = cleaned

    if p1 == p2:
        return None

    return (p1, p2)

def draw_game_tree(
    game,
    selected_ids=None,
    x_gap=3.2,
    gen_gap=3.2,
    parent_pair_gap=1.7,
    rock_image_size=1.15,
    canvas_width=1800,
    canvas_height=1100,
    show_labels=True,
    show_sold=True,
    inactive_sold_opacity=0.55,
    show = False,
    highlight_breeding_queue=False
):
    """
    Draw the full game lineage tree.

    Features:
    - all rocks shown
    - sold rocks marked with green $
    - craisen rocks marked with red X
    - bred parents marked with gray circle if not sold/craisen
    - selected rocks highlighted
    """
    evaluate_all_rocks(game)

    selected_ids = set(selected_ids or [])

    rocks_dict = game.rocks

    if len(rocks_dict) == 0:
        print("No rocks to draw.")
        return None

    pos = compute_lineage_positions_christmas(
        rocks_dict,
        x_gap=x_gap,
        gen_gap=gen_gap,
        parent_pair_gap=parent_pair_gap,
        anti_overlap=True
    )

    fig = go.Figure()

    # Group children by parent pair.
    families = {}

    for child_id, rock in rocks_dict.items():
        if rock.parents is not None:
            key = tuple(sorted(rock.parents))
            families.setdefault(key, []).append(child_id)

    family_styles = get_family_styles(rocks_dict)

    # Draw family lines.
    for parent_pair, child_ids in sorted(families.items()):
        parent_pair = normalize_parent_pair_for_tree(parent_pair)

        if parent_pair is None:
            continue

        p1, p2 = parent_pair

        if p1 not in pos or p2 not in pos:
            continue

        x_line, y_line = build_family_segments(pos, parent_pair, child_ids)

        if len(x_line) == 0:
            continue

        style = family_styles.get(parent_pair, {"color": "#4E79A7", "dash": "solid"})

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=7,
                    color="rgba(255,255,255,0.95)",
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    width=3,
                    color=style["color"],
                    dash=style["dash"]
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Add rock images.
    image_cache = {}

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        if rid not in image_cache:
            image_cache[rid] = rock_to_image_uri_cached(game, rock)

        size_scale = get_rock_size_scale(rock)

        opacity = 1.0
        if getattr(rock, "sold", False):
            opacity = inactive_sold_opacity

        fig.add_layout_image(
            dict(
                source=image_cache[rid],
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=rock_image_size * size_scale,
                sizey=rock_image_size * size_scale,
                xanchor="center",
                yanchor="middle",
                layer="above",
                opacity=opacity
            )
        )

    # Selected rings.
    for rid in selected_ids:
        if rid not in pos:
            continue

        x, y = pos[rid]

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    size=rock_image_size * 85,
                    color="rgba(255,255,255,0)",
                    line=dict(
                        color="gold",
                        width=6
                    )
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Hover points, labels, and status symbols.
    hover_x = []
    hover_y = []
    hover_text = []
    labels = []

    status_x = []
    status_y = []
    status_text = []
    status_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]
        v = get_visual_phenotype(rock)

        parent_text = "Founder/import"
        if rock.parents is not None:
            parent_text = f"Parents: #{rock.parents[0]} and #{rock.parents[1]}"

        flags = []
        if getattr(rock, "sold", False):
            flags.append("SOLD")
        if getattr(rock, "used_as_parent", False):
            flags.append("BRED PARENT")
        if getattr(rock, "is_craisen", 0) == 1:
            flags.append("CRAISEN")
        if getattr(rock, "imported", False):
            flags.append("IMPORTED")

        flag_text = ", ".join(flags) if flags else "OK"

        full_phenotype_text = format_selected_phenotype_hover(rock)

        text = (
            f"<b>{rock.name} #{rock.id}</b><br>"
            f"Generation: {rock.generation}<br>"
            f"{parent_text}<br>"
            f"Gender: {v.get('gender', 'n/a')} {get_gender_symbol(rock)}<br>"
            f"Base value: ${rock.base_value}<br>"
            f"Sell value: ${rock.sell_value}<br>"
            f"Score value: ${rock.score_value}<br>"
            f"Status: {flag_text}<br>"
            f"<br>"
            f"<b>Full phenotype</b><br>"
            f"{full_phenotype_text}"
        )

        hover_x.append(x)
        hover_y.append(y)
        hover_text.append(text)

        if show_labels:
            labels.append(f"{rock.name}<br>#{rock.id}")
        else:
            labels.append("")

        symbol = get_rock_status_symbol(rock)

        if symbol != "":
            status_x.append(x)
            status_y.append(y - 0.72 * rock_image_size)
            status_text.append(symbol)
            status_colors.append(get_rock_status_color(rock))

            # Queued breeding pair markers.
        if highlight_breeding_queue and len(game.breeding_queue) > 0:
            queue_labels_by_rock = get_queue_labels_by_rock(game)

            # Draw dashed red lines between currently queued future parents.
            for i, entry in enumerate(game.breeding_queue, start=1):
                a, b = get_queue_entry_pair(entry)

                if a not in pos or b not in pos:
                    continue

                xa, ya = pos[a]
                xb, yb = pos[b]

                # Slightly above rocks so it reads as a planned pair, not lineage.
                ya2 = ya + 0.78 * rock_image_size
                yb2 = yb + 0.78 * rock_image_size

                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[ya2, yb2],
                        mode="lines",
                        line=dict(
                            color="crimson",
                            width=3,
                            dash="dot"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

                mid_x = 0.5 * (xa + xb)
                mid_y = 0.5 * (ya2 + yb2)

                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y + 0.12 * rock_image_size],
                        mode="text",
                        text=[f"❤{i}"],
                        textfont=dict(
                            size=28,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            # Draw heart labels on each queued rock.
            for rid, labels in queue_labels_by_rock.items():
                if rid not in pos:
                    continue

                x, y = pos[rid]
                rock = rocks_dict[rid]
                size_scale = get_rock_size_scale(rock)

                fig.add_trace(
                    go.Scatter(
                        x=[x - 0.42 * rock_image_size * size_scale],
                        y=[y + 0.42 * rock_image_size * size_scale],
                        mode="text",
                        text=[" ".join(labels)],
                        textfont=dict(
                            size=24,
                            color="crimson",
                            family="Arial Black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

    # Invisible hover/labels.
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=rock_image_size * 48,
                color="rgba(0,0,0,0)"
            ),
            text=labels,
            textposition="bottom center",
            textfont=dict(size=10, color="black"),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        )
    )

    # Status symbols.
    for sx, sy, st, sc in zip(status_x, status_y, status_text, status_colors):
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="text",
                text=[st],
                textfont=dict(
                    size=30,
                    color=sc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

        # Gender symbols near top-right of each rock.
    gender_x = []
    gender_y = []
    gender_text = []
    gender_colors = []

    for rid, rock in rocks_dict.items():
        if rid not in pos:
            continue

        x, y = pos[rid]

        size_scale = get_rock_size_scale(rock)

        gender_x.append(x + 0.42 * rock_image_size * size_scale)
        gender_y.append(y + 0.42 * rock_image_size * size_scale)
        gender_text.append(get_gender_symbol(rock))
        gender_colors.append(get_gender_color(rock))

    for gx, gy, gt, gc in zip(gender_x, gender_y, gender_text, gender_colors):
        fig.add_trace(
            go.Scatter(
                x=[gx],
                y=[gy],
                mode="text",
                text=[gt],
                textfont=dict(
                    size=26,
                    color=gc,
                    family="Arial Black"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    margin_x = 3.0
    margin_y = 3.0

    fig.update_layout(
        title=(
            f"Rock Game Tree — Generation {game.generation}/{game.max_generation} "
            f"| Cash ${game.money} | Score Estimate ${get_final_score_estimate(game)}"
        ),
        width=canvas_width,
        height=canvas_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            visible=False,
            range=[min(xs) - margin_x, max(xs) + margin_x]
        ),
        yaxis=dict(
            visible=False,
            range=[min(ys) - margin_y, max(ys) + margin_y],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=20, r=20, t=70, b=20),
        dragmode="pan"
    )

    if show:
        fig.show(config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        })

    return fig

# -----------------------------
# ROCK FLAGS AND SYMBOLS
# -----------------------------

def is_rock_sold_flag(rock):
    return bool(getattr(rock, "sold", False))

def get_rock_status_symbol(rock):
    """
    Symbol shown near rocks in game views.
    """
    if getattr(rock, "puffed", False):
        return "☁"

    if getattr(rock, "dead", False):
        return "†"

    if is_rock_sold_flag(rock):
        return "$"

    if getattr(rock, "is_craisen", 0) == 1:
        return "X"

    if getattr(rock, "used_as_parent", False):
        return "○"

    if getattr(rock, "market_guest", False):
        return(("NPC", "darkviolet"))

    return ""

def get_rock_status_color(rock):
    """
    Color for status symbols.
    """
    if getattr(rock, "puffed", False):
        return "dimgray"

    if getattr(rock, "dead", False):
        return "black"

    if is_rock_sold_flag(rock):
        return "green"

    if getattr(rock, "is_craisen", 0) == 1:
        return "crimson"

    if getattr(rock, "used_as_parent", False):
        return "gray"

    return "black"









