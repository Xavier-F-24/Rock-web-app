import io
import contextlib

import streamlit as st

from rockgame_core import *


st.set_page_config(
    page_title="Rock Game",
    page_icon="🪨",
    layout="wide"
)


#streamlit run app.py


# -----------------------------
# Utility: capture printed output
# -----------------------------

def capture_print(fn, *args, **kwargs):
    """
    Run a print-heavy core function and return printed text.
    This lets us use existing notebook-style functions in Streamlit.
    """
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)

    return result, buf.getvalue()

def load_game_from_uploaded_file(uploaded_file):
    """
    Load a GameState from a Streamlit uploaded JSON file.
    """
    if uploaded_file is None:
        return None

    json_string = uploaded_file.getvalue().decode("utf-8")
    return game_from_json_string(json_string)

def get_catalog_option_maps(gene_name):
    """
    Return values and label map for REQUEST_TRAIT_CATALOG selectboxes.
    """
    options_dict = REQUEST_TRAIT_CATALOG[gene_name]["options"]

    values = list(options_dict.values())
    label_by_value = {v: k for k, v in options_dict.items()}

    return values, label_by_value

def requested_dependency_ok_streamlit(gene_name):
    """
    Streamlit-side dependency logic.

    Eye Color requires Eyes.
    Hair Color and Hair Texture require Hair.
    """
    if gene_name == "eye_color":
        eyes_checked = st.session_state.get("req_eyes_check", False)
        eyes_value = st.session_state.get("req_eyes_value", "00")
        return eyes_checked and eyes_value != "00"

    if gene_name in ["hair_color", "hair_texture"]:
        hair_checked = st.session_state.get("req_hair_check", False)
        hair_value = st.session_state.get("req_hair_value", "00")
        return hair_checked and hair_value != "00"

    return True

def collect_requested_values_from_streamlit():
    """
    Build requested_values dict from Streamlit checkbox/selectbox state.
    """
    requested_values = {}

    for gene_name in REQUEST_TRAIT_CATALOG.keys():
        check_key = f"req_{gene_name}_check"
        value_key = f"req_{gene_name}_value"

        checked = st.session_state.get(check_key, False)

        if not checked:
            continue

        if not requested_dependency_ok_streamlit(gene_name):
            continue

        requested_values[gene_name] = str(st.session_state.get(value_key, "00"))

    return requested_values

def clear_requested_import_state():
    """
    Clear requested import checkboxes and pending quote.
    """
    for gene_name in REQUEST_TRAIT_CATALOG.keys():
        check_key = f"req_{gene_name}_check"

        if check_key in st.session_state:
            st.session_state[check_key] = False

    if hasattr(st.session_state.game, "pending_requested_import"):
        st.session_state.game.pending_requested_import = None

    st.session_state.requested_import_message = ""

def show_requested_quote_preview(game):
    """
    Display pending requested import quote if one exists.
    """
    quote = getattr(game, "pending_requested_import", None)

    if quote is None:
        return

    rock = quote["rock"]

    st.write("### Current Requested Rock Quote")
    st.write(f"**Rock:** #{rock.id} {rock.name}")
    st.write(f"**Base value:** ${rock.base_value}")
    st.write(f"**Purchase cost:** ${quote['cost']}")
    st.write(f"**Craisen:** {bool(rock.is_craisen)}")
    st.write(f"**Attempts:** {quote['attempts']}")

    uri = rock_to_image_uri(rock)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="{uri}" style="width:280px; max-width:100%;">
        </div>
        """,
        unsafe_allow_html=True
    )

    actual_forced = getattr(rock, "requested_actual_forced_values", {})

    if len(quote["requested_values"]) > 0:
        st.write("**Requested traits:**")
        for gene_name, requested_gene_value in quote["requested_values"].items():
            label = REQUEST_TRAIT_CATALOG.get(gene_name, {}).get("label", gene_name)
            actual_gene_value = actual_forced.get(gene_name, requested_gene_value)

            if actual_gene_value != requested_gene_value:
                st.write(f"- {label}: requested `{requested_gene_value}`, generated `{actual_gene_value}`")
            else:
                st.write(f"- {label}: `{actual_gene_value}`")
    else:
        st.write("**Requested traits:** None")

def get_game():
    if "game" not in st.session_state:
        st.session_state.game = create_new_game(seed=None)

    return st.session_state.game

def reset_game(seed=None):
    st.session_state.game = create_new_game(seed=seed)
    st.session_state.selected_parent_a = None
    st.session_state.selected_parent_b = None
    st.session_state.selected_sell_rock = None
    st.session_state.requested_import_message = ""

    if "uploaded_save" in st.session_state:
        del st.session_state["uploaded_save"]

def option_labels_and_values(options):
    """
    Core dropdown options are usually:
        [(label, value), ...]
    Streamlit wants values, but can format them.
    """
    labels = {}
    values = []

    for label, value in options:
        labels[value] = label
        values.append(value)

    return values, labels


# -----------------------------
# Header
# -----------------------------

st.title("🪨 Rock Game")
st.caption("Seven generations. Breed wisely. Sell well. Avoid craisen chaos.")

game = get_game()

top_left, top_mid, top_right = st.columns([1, 1, 1])

with top_left:
    if st.button("New Game", type="primary"):
        reset_game()
        st.rerun()

with top_mid:
    st.metric("Generation", f"{game.generation} / {game.max_generation}")

with top_right:
    score = get_final_score_estimate(game)
    st.metric("Score Estimate", f"${score}")


# -----------------------------
# Sidebar: compact status
# -----------------------------

with st.sidebar:
    st.header("Game Status")

    evaluate_all_rocks(game)

    st.write(f"**Cash:** ${game.money}")
    st.write(f"**Generation:** {game.generation} / {game.max_generation}")
    st.write(f"**Breeding Queue:** {len(game.breeding_queue)} / {game.max_pairs_per_generation}")
    st.write(f"**Score Estimate:** ${get_final_score_estimate(game)}")

    st.divider()

    # -----------------------------
    # Pages
    # -----------------------------

    page = st.sidebar.radio(
        "Page",
        [
            "🌳 Game Board",
            "🧬 Breeding",
            "💰 Market",
            "💾 Save / Load",
            "📋 Tables",
        ]
    )

    st.divider()
    st.subheader("Quick Save")

    st.download_button(
        label="Download Save",
        data=game_to_json_string(game),
        file_name=make_save_filename(game),
        mime="application/json"
    )

    st.divider()
    st.subheader("Recent Events")
    if len(game.events) == 0:
        st.write("No events yet.")
    else:
        for event in game.events[-8:][::-1]:
            st.write(f"- {event}")

    
# -----------------------------
# Game Board
# -----------------------------

if page == "🌳 Game Board":
    st.subheader("Family Tree")

    fig = draw_game_tree(
        game,
        selected_ids=[],
        show_labels=True,
        show=False
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        }
    )


# -----------------------------
# Breeding
# -----------------------------

elif page == "🧬 Breeding":
    st.subheader("Breeding")

    parent_options = get_breeding_dropdown_options(game)
    parent_values, parent_labels = option_labels_and_values(parent_options)

    col_a, col_b = st.columns(2)

    with col_a:
        parent_a = st.selectbox(
            "Parent A",
            options=parent_values,
            format_func=lambda x: parent_labels.get(x, "None"),
            key="parent_a_select"
        )

    with col_b:
        parent_b = st.selectbox(
            "Parent B",
            options=parent_values,
            format_func=lambda x: parent_labels.get(x, "None"),
            key="parent_b_select"
        )

    potion_options = get_owned_potion_options(game)
    potion_values, potion_labels = option_labels_and_values(potion_options)

    potion_key = st.selectbox(
        "Potion for this pair",
        options=potion_values,
        format_func=lambda x: potion_labels.get(x, "No potion")
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Preview Pair"):
            _, text = capture_print(preview_breeding_pair, game, parent_a, parent_b)
            st.text(text)

    with b2:
        if st.button("Add Pair", type="primary"):
            _, text = capture_print(
                add_pair_to_breeding_queue,
                game,
                parent_a,
                parent_b,
                potion_key
            )
            st.text(text)
            st.rerun()

    with b3:
        if st.button("Clear Queue"):
            _, text = capture_print(clear_breeding_queue, game)
            st.text(text)
            st.rerun()

    st.divider()

    st.subheader("Current Breeding Queue")
    _, queue_text = capture_print(show_breeding_queue, game)
    st.text(queue_text)

    if st.button("Breed Generation", type="primary"):
        _, text = capture_print(run_generation_from_ui, game)
        st.text(text)
        st.rerun()


# -----------------------------
# Market
# -----------------------------

elif page == "💰 Market":
    st.subheader("Sell Rocks")

    sell_options = get_sell_dropdown_options(game)
    sell_values, sell_labels = option_labels_and_values(sell_options)

    sell_id = st.selectbox(
        "Rock to sell",
        options=sell_values,
        format_func=lambda x: sell_labels.get(x, "None")
    )

    if st.button("Sell Selected Rock"):
        _, text = capture_print(sell_rock, game, sell_id, False)
        st.text(text)
        st.rerun()

    st.divider()

    st.subheader("Random Import")

    import_gender = st.selectbox(
        "Import gender",
        options=[None, "male", "female"],
        format_func=lambda x: "Random" if x is None else x.title()
    )

    if st.button("Import Random Rock ($8)"):
        _, text = capture_print(import_random_rock, game, RANDOM_IMPORT_COST, import_gender)
        st.text(text)
        st.rerun()


    st.divider()

    with st.expander("🧾 Requested Rock Import", expanded=False):
        st.write(
            "Check traits to force them. Unchecked traits are suppressed, "
            "but many can still carry hidden recessive alleles."
        )

        st.caption(
            "Eye Color requires Eyes. Hair Color and Hair Texture require Hair."
        )

        req_cols = st.columns(2)

        gene_names = list(REQUEST_TRAIT_CATALOG.keys())

        for i, gene_name in enumerate(gene_names):
            info = REQUEST_TRAIT_CATALOG[gene_name]
            label = info["label"]

            values, label_by_value = get_catalog_option_maps(gene_name)

            dep_ok = requested_dependency_ok_streamlit(gene_name)

            with req_cols[i % 2]:
                check_key = f"req_{gene_name}_check"
                value_key = f"req_{gene_name}_value"

                disabled = not dep_ok

                if disabled:
                    st.checkbox(
                        f"{label} — locked",
                        key=check_key,
                        disabled=True,
                        value=False
                    )
                else:
                    st.checkbox(
                        label,
                        key=check_key
                    )

                st.selectbox(
                    f"{label} level",
                    options=values,
                    format_func=lambda x, m=label_by_value: m.get(x, str(x)),
                    key=value_key,
                    disabled=disabled or not st.session_state.get(check_key, False),
                    label_visibility="collapsed"
                )

                if disabled:
                    if gene_name == "eye_color":
                        st.caption("Requires Eyes.")
                    elif gene_name in ["hair_color", "hair_texture"]:
                        st.caption("Requires Hair.")

        reroll_if_craisen = st.checkbox(
            "Reroll requested import if craisen",
            value=True,
            key="req_reroll_craisen"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Preview Requested Rock"):
                requested_values = collect_requested_values_from_streamlit()

                _, text = capture_print(
                    preview_requested_import_rock,
                    game,
                    requested_values,
                    reroll_if_craisen
                )

                st.session_state.requested_import_message = text

        with c2:
            if st.button("Buy Requested Rock", type="primary"):
                _, text = capture_print(
                    buy_requested_import_quote,
                    game
                )

                st.session_state.requested_import_message = text
                st.rerun()

        with c3:
            if st.button("Clear Request"):
                clear_requested_import_state()
                st.rerun()

        if st.session_state.get("requested_import_message", ""):
            st.text(st.session_state.requested_import_message)

        show_requested_quote_preview(game)

    st.divider()

    st.subheader("Potion Shop")

    potion_shop_options = [
        (f"{info['name']} (${info['cost']})", key)
        for key, info in POTION_SHOP.items()
    ]

    potion_shop_values, potion_shop_labels = option_labels_and_values(potion_shop_options)

    buy_key = st.selectbox(
        "Potion to buy",
        options=potion_shop_values,
        format_func=lambda x: potion_shop_labels.get(x, x)
    )

    if st.button("Buy Potion"):
        _, text = capture_print(buy_potion, game, buy_key)
        st.text(text)
        st.rerun()

    st.divider()

    st.subheader("Inventory")
    _, inventory_text = capture_print(show_inventory, game)
    st.text(inventory_text)


# -----------------------------
# Save / Load
# -----------------------------

elif page == "💾 Save / Load":
    st.subheader("Save Game")

    save_json = game_to_json_string(game)
    save_filename = make_save_filename(game)

    st.download_button(
        label="Download Save JSON",
        data=save_json,
        file_name=save_filename,
        mime="application/json",
        type="primary"
    )

    st.caption("This saves your full game state: rocks, genes, lineage, money, potions, queue, and events.")

    st.divider()

    st.subheader("Load Game")

    uploaded_save = st.file_uploader(
        "Upload a Rock Game JSON save",
        type=["json"]
    )

    if uploaded_save is not None:
        if st.button("Load Uploaded Save", type="primary"):
            try:
                loaded_game = load_game_from_uploaded_file(uploaded_save)

                st.session_state.game = loaded_game
                st.session_state.requested_import_message = ""
                st.session_state.selected_parent_a = None
                st.session_state.selected_parent_b = None
                st.session_state.selected_sell_rock = None

                st.success("Save loaded successfully.")
                st.rerun()

            except Exception as e:
                st.error(f"Could not load save file: {e}")

    st.warning(
        "Loading a save replaces the current game in this browser session. "
        "Download your current save first if you want to keep it."
    )

# -----------------------------
# Tables
# -----------------------------

elif page == "📋 Tables":
    st.subheader("Money Table")

    _, money_text = capture_print(show_rock_money_table, game)
    st.text(money_text)

    st.subheader("Raw Event Log")
    for event in game.events:
        st.write(f"- {event}")