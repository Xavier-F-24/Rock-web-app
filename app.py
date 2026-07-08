import io
import contextlib

import streamlit as st

from rockgame_core import *
from Rock_GameState.rock_game_state_helper import GameMaster as SplitGameMaster
from Rock_Serialization.rock_serialization_helper import (
    game_from_json_string as split_game_from_json_string,
    game_to_json_string as split_game_to_json_string,
)

st.set_page_config(
    page_title="Rock Game",
    page_icon="🪨",
    layout="wide"
)


#streamlit run app.py


def render_split_module_prototype_app():
    """
    Minimal playable loop backed by the split GameMaster modules.
    """
    st.title("Rock Game Prototype")

    if "split_game" not in st.session_state:
        st.session_state.split_game = SplitGameMaster(seed=None)

    game = st.session_state.split_game

    with st.sidebar:
        st.header("Prototype")
        if st.button("New Prototype Game"):
            st.session_state.split_game = SplitGameMaster(seed=None)
            st.rerun()

        save_json = split_game_to_json_string(game)
        st.download_button(
            "Download Prototype Save",
            save_json,
            file_name=f"rock_game_prototype_gen{game.generation}.json",
            mime="application/json",
        )

        uploaded = st.file_uploader("Load Prototype Save", type=["json"], key="split_save_upload")
        if uploaded is not None and st.button("Load Prototype Save"):
            st.session_state.split_game = split_game_from_json_string(
                uploaded.getvalue().decode("utf-8")
            )
            st.rerun()

    status = game.update_display()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Generation", status["generation"])
    col_b.metric("Money", status["money"])
    col_c.metric("Rocks", status["rock_count"])
    col_d.metric("Market Pods", status["market_pods"])

    st.subheader("Rocks")
    st.dataframe(
        [
            {
                "id": rock.id,
                "name": rock.name.full_name if hasattr(rock.name, "full_name") else str(rock.name),
                "sex": rock.sex.value,
                "generation": rock.generation,
                "status": rock.status.value,
                "sell_value": rock.sell_value,
            }
            for rock in game.rocks.values()
        ],
        width="stretch",
    )

    active_rocks = [
        rock
        for rock in game.rocks.values()
        if getattr(rock.status, "value", rock.status) == "active"
    ]

    st.subheader("Breed")
    if len(active_rocks) >= 2:
        labels = {
            rock.id: f"#{rock.id} {rock.name.full_name if hasattr(rock.name, 'full_name') else rock.name} ({rock.sex.value})"
            for rock in active_rocks
        }
        values = list(labels)

        breed_col_a, breed_col_b, breed_col_c = st.columns(3)
        parent_a_id = breed_col_a.selectbox("Parent A", values, format_func=lambda rid: labels[rid])
        parent_b_id = breed_col_b.selectbox("Parent B", values, format_func=lambda rid: labels[rid])
        potion_key = breed_col_c.selectbox("Potion", [None] + sorted(game.potions), format_func=lambda key: "None" if key is None else key)

        if st.button("Queue Pair"):
            try:
                game.add_pair_to_queue(parent_a_id, parent_b_id, potion_key=potion_key)
                st.success("Pair queued.")
            except Exception as exc:
                st.error(str(exc))

    st.write(f"Queued pairs: {len(game.breeding_queue)}")
    if st.button("Advance Generation"):
        try:
            children = game.advance_generation()
            st.success(f"Advanced generation with {len(children)} child rock(s).")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Market")
    market_col_a, market_col_b, market_col_c = st.columns(3)
    with market_col_a:
        potion_to_buy = st.selectbox("Buy Potion", ["fertility", "reroll", "mutation", "anti_craisen"])
        if st.button("Buy Potion"):
            try:
                game.buy_potion(potion_to_buy)
                st.success(f"Bought {potion_to_buy}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with market_col_b:
        if st.button("Buy Random Rock"):
            try:
                rock = game.buy_random_rock()
                st.success(f"Bought rock #{rock.id}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with market_col_c:
        if st.button("Buy Orange Double-Eye Rock"):
            try:
                rock = game.buy_defined_trait_rock(
                    {"gender": "female", "color": "34", "eyes": "11"},
                    random_fill=False,
                )
                st.success(f"Bought rock #{rock.id}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if game.market_pods:
        offer_labels = {
            offer.offer_id: f"{offer.name} (${offer.price})"
            for offer in game.market_pods
            if not offer.used
        }
        if offer_labels:
            offer_id = st.selectbox("Market Pod", list(offer_labels), format_func=lambda oid: offer_labels[oid])
            if st.button("Buy Selected Pod"):
                try:
                    pending = game.market_manager.buy_market_pod(game, offer_id)
                    st.success(f"Pod bought with {len(pending.children)} child candidate(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if game.pending_market_pod is not None:
        pending = game.pending_market_pod
        child_options = list(range(len(pending.children)))
        child_index = st.selectbox("Keep Market Child", child_options)
        if st.button("Keep Child"):
            child = game.market_manager.choose_market_pod_child(game, child_index)
            st.success(f"Kept child #{child.id}.")
            st.rerun()

    st.subheader("Recent Events")
    for event in game.events[-10:]:
        st.write(f"- {event}")


if st.sidebar.checkbox("Use split-module prototype", value=False):
    render_split_module_prototype_app()
    st.stop()


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

def get_breeding_candidate_rocks(game, include_queued=True):
    """
    Rocks that are biologically/actionably breedable.

    include_queued=True means queued rocks still show in the gallery,
    but we label them as queued.
    """
    evaluate_all_rocks(game)

    candidates = []

    for rid, rock in sorted(game.rocks.items()):
        if not is_parent_dropdown_eligible(rock):
            continue

        if not include_queued and is_rock_queued_for_breeding(game, rid):
            continue

        candidates.append(rock)

    return candidates

def get_streamlit_rock_image_uri(game, rock):
    """
    Use cached renderer if available.
    """
    try:
        return rock_to_image_uri_cached(game, rock)
    except Exception:
        return rock_to_image_uri(rock)

def render_breeder_card(game, rock, selected=False):
    """
    Render one breeder card.
    """
    uri = get_streamlit_rock_image_uri(game, rock)

    border = "4px solid gold" if selected else "1px solid #dddddd"

    #print("why")

    queued = is_rock_queued_for_breeding(game, rock.id)

    queue_text = ""
    if queued:
        labels = get_queue_labels_by_rock(game).get(rock.id, [])
        queue_text = f"<div style='color:crimson; font-weight:bold;'>{' '.join(labels)} QUEUED</div>"

    gender_symbol = get_gender_symbol(rock)
    gender_color = get_gender_color(rock)

    st.markdown(
        f"""
        <div style="
            border:{border};
            border-radius:12px;
            padding:8px;
            text-align:center;
            background-color:#ffffff;
            min-height:300px;
        ">
            <div style="text-align:right; font-size:24px; color:{gender_color}; font-weight:bold;">
                {gender_symbol}
            </div>
            <img src="{uri}" style="width:150px; max-width:100%;">
            <div style="font-weight:bold;">#{rock.id} {rock.name}</div>
            <div>Gen {rock.generation} | ${rock.sell_value}</div>
            {queue_text}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_available_breeders_gallery(game, selected_ids=None, max_cols=4):
    """
    Show breeding candidates at the top of the Breeding page.
    """
    selected_ids = set(selected_ids or [])

    candidates = get_breeding_candidate_rocks(game, include_queued=True)

    if len(candidates) == 0:
        st.info("No breeding candidates available.")
        return

    st.write(f"**Breeding candidates:** {len(candidates)}")

    for start in range(0, len(candidates), max_cols):
        row = candidates[start:start + max_cols]
        cols = st.columns(max_cols)

        for col, rock in zip(cols, row):
            with col:
                render_breeder_card(
                    game,
                    rock,
                    selected=(rock.id in selected_ids)
                )

def render_queue_removal_controls(game):
    """
    Streamlit UI for removing selected queued breeding pairs.
    """
    if len(game.breeding_queue) == 0:
        st.info("No breeding pairs currently queued.")
        return

    with st.expander("Remove queued breeding pair(s)", expanded=False):
        st.write("Select the pair(s) you want to remove from the queue.")

        selected_indices = []

        for i, entry in enumerate(game.breeding_queue):
            a, b = get_queue_entry_pair(entry)
            potion_key = get_queue_entry_potion(entry)

            rock_a = get_rock(game, a)
            rock_b = get_rock(game, b)

            name_a = rock_a.name if rock_a is not None else "missing"
            name_b = rock_b.name if rock_b is not None else "missing"

            potion_text = get_potion_name(potion_key)

            label = (
                f"Remove Pair {i + 1}: "
                f"#{a} {name_a} × #{b} {name_b} "
                f"| Potion: {potion_text}"
            )

            checked = st.checkbox(
                label,
                key=f"remove_queue_pair_{game.generation}_{i}_{a}_{b}_{potion_key}"
            )

            if checked:
                selected_indices.append(i)

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "Remove Selected Pair(s)",
                disabled=len(selected_indices) == 0
            ):
                removed_count, text = capture_print(
                    remove_selected_pairs_from_breeding_queue,
                    game,
                    selected_indices
                )

                st.text(text)
                st.success(f"Removed {removed_count} queued pair(s).")
                st.rerun()

        with c2:
            if st.button("Clear All Queued Pair(s)"):
                _, text = capture_print(clear_breeding_queue, game)
                st.text(text)
                st.rerun()

def requested_dependency_ok_streamlit(gene_name):
    """
    Streamlit-side dependency logic.

    Eye Color requires Eyes.
    Hair Color requires Hair OR Facial Hair OR Brows.
    Hair Texture requires Hair OR Facial Hair.
    """
    if gene_name == "eye_color":
        eyes_checked = st.session_state.get("req_eyes_check", False)
        eyes_value = st.session_state.get("req_eyes_value", "00")
        return eyes_checked and eyes_value != "00"

    if gene_name == "hair_color":
        hair_checked = st.session_state.get("req_hair_check", False)
        hair_value = st.session_state.get("req_hair_value", "00")

        facial_checked = st.session_state.get("req_facial_hair_check", False)
        facial_value = st.session_state.get("req_facial_hair_value", "00")

        brows_checked = st.session_state.get("req_brows_check", False)
        brows_value = st.session_state.get("req_brows_value", "00")

        return (
            (hair_checked and hair_value != "00")
            or (facial_checked and facial_value != "00")
            or (brows_checked and brows_value != "00")
        )

    if gene_name == "hair_texture":
        hair_checked = st.session_state.get("req_hair_check", False)
        hair_value = st.session_state.get("req_hair_value", "00")

        facial_checked = st.session_state.get("req_facial_hair_check", False)
        facial_value = st.session_state.get("req_facial_hair_value", "00")

        brows_checked = st.session_state.get("req_brows_check", False)
        brows_value = st.session_state.get("req_brows_value", "00")

        return (
            (hair_checked and hair_value != "00")
            or (facial_checked and facial_value != "00")
            or (brows_checked and brows_value != "00")
        )

    return True

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

    ensure_player_profile_state(game)

    st.markdown("### Player Profile")

    player_name_input = st.text_input(
        "Player name",
        value=getattr(game, "player_name", ""),
        placeholder="Type your name..."
    )

    if player_name_input != getattr(game, "player_name", ""):
        set_player_name(game, player_name_input)

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
        file_name=get_game_save_filename(game),
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

    if parent_a is not None and parent_b is not None and parent_a != parent_b:
        st.info(format_relatedness_report(game, parent_a, parent_b))

    st.divider()

    b1, b2 = st.columns(2)

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

    st.divider()

    st.subheader("Current Breeding Queue")
    _, queue_text = capture_print(show_breeding_queue, game)
    st.text(queue_text)

    render_queue_removal_controls(game)

    show_breeding_tree = st.checkbox(
        "Show queue tree on breeding page",
        value=True
    )

    st.divider()

    if st.button("Breed Generation", type="primary"):
        _, text = capture_print(run_generation_from_ui, game)
        st.text(text)
        st.rerun()

    selected_breeding_ids = []

    if parent_a is not None:
        selected_breeding_ids.append(parent_a)

    if parent_b is not None:
        selected_breeding_ids.append(parent_b)

    

    with st.expander("Available Breeding Rocks", expanded=True):
        render_available_breeders_gallery(
            game,
            selected_ids=selected_breeding_ids,
            max_cols=4
        )

    st.divider()

    if show_breeding_tree:
        fig = draw_game_tree(
            game,
            selected_ids=selected_breeding_ids,
            show_labels=True,
            show=False,
            highlight_breeding_queue=True
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

    st.subheader("Breeding Pod Market")

    generate_market_pods_for_generation(game)

    if game.pending_market_pod is not None:
        pending = game.pending_market_pod

        st.warning(
            f"Pending pod: {pending['name']}. Choose one child to keep."
        )

        st.write("### Guest Parents")

        parent_a = game.rocks[pending["parent_a_id"]]
        parent_b = game.rocks[pending["parent_b_id"]]

        cols = st.columns(2)

        with cols[0]:
            st.image(rock_to_image_uri_cached(game, parent_a), caption=f"Guest Parent A #{parent_a.id}")

        with cols[1]:
            st.image(rock_to_image_uri_cached(game, parent_b), caption=f"Guest Parent B #{parent_b.id}")

        st.write("### Choose One Child")

        children = pending["children"]

        child_cols = st.columns(min(4, max(1, len(children))))

        for i, child in enumerate(children):
            with child_cols[i % len(child_cols)]:
                st.image(rock_to_image_uri(child), caption=f"Candidate {i + 1} | ${child.sell_value}")

                if st.button(f"Keep Candidate {i + 1}", key=f"keep_market_child_{i}"):
                    _, text = capture_print(choose_market_pod_child, game, i)
                    st.text(text)
                    st.rerun()

    else:
        st.caption("Outside breeders refresh each generation. You buy the pod, they keep the leftovers. Brutal.")

        for offer in game.market_pods:
            if offer.get("used", False):
                continue

            tier = offer["tier"]

            with st.container(border=True):
                st.write(f"### {offer['name']}")
                st.caption(offer["tagline"])
                st.write(f"**Price:** ${offer['price']}")
                st.write("Parents hidden until purchase.")
                st.write("You may keep exactly one child from the clutch.")

                if st.button(
                    f"Buy {offer['name']} Pod - ${offer['price']}",
                    key=f"buy_pod_{offer['offer_id']}"
                ):
                    _, text = capture_print(buy_market_pod, game, offer["offer_id"])
                    st.text(text)
                    st.rerun()

    st.divider()

    with st.expander("🧾 Requested Rock Import - NOW FOR LIL SHITTERS", expanded=False):
        st.write(
            "Check traits to force them. Unchecked traits are suppressed, "
            "but many can still carry hidden recessive alleles."
        )

        st.caption(
            "Eye Color requires Eyes. Hair Color requires Hair, Facial Hair, or Brows. "
            "Hair Texture requires Hair or Facial Hair."
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
                    elif gene_name == "hair_color":
                        st.caption("Requires Hair, Facial Hair, or Brows.")
                    elif gene_name == "hair_texture":
                        st.caption("Requires Hair or Facial Hair, or Brows.")
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
    save_filename = get_game_save_filename(game)

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
