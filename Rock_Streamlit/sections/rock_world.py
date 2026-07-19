"""Public persistent NPC farm galleries, lineages, and direct trades."""

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.components.farm_visualizer import render_farm
from Rock_Streamlit.components.world_turn_panel import render_world_turn_panel
from Rock_Streamlit.ui_components import page_header, section
from rockgame_ui import game_controller


def _trade_builder(game, farm_id, rocks):
    world = game_controller.get_world(game)
    requested = [rock for rock in rocks if rock.is_active and rock.id not in world.reserved_rock_ids]
    offered = [rock for rock in game_controller.get_active_rocks(game) if rock.id not in world.reserved_rock_ids]
    if not requested:
        st.info("This farm has no public transferable rocks right now.")
        return
    requested_rock = st.selectbox(
        "Rock requested", requested, format_func=game_controller.rock_label,
        key=f"world_requested_{farm_id}",
    )
    offered_options = [None] + offered
    offered_rock = st.selectbox(
        "Your rock offered", offered_options,
        format_func=lambda rock: "No rock" if rock is None else game_controller.rock_label(rock),
        key=f"world_offered_{farm_id}",
    )
    money_columns = st.columns(2)
    offered_money = money_columns[0].number_input("Money you offer", min_value=0, value=0, key=f"world_money_offer_{farm_id}")
    requested_money = money_columns[1].number_input("Money you request", min_value=0, value=0, key=f"world_money_request_{farm_id}")
    if st.button("Send Trade Offer", key=f"world_send_trade_{farm_id}", type="primary"):
        result = game_controller.create_direct_trade_offer(
            game, farm_id, offered_rock_id=None if offered_rock is None else offered_rock.id,
            requested_rock_id=requested_rock.id, offered_money=int(offered_money), requested_money=int(requested_money),
        )
        (st.success if result.ok else st.error)(result.message)


def render():
    game = get_game_state()
    page_header("Rock World", "Visit public farms, inspect their lineages, and propose real trades.")
    render_world_turn_panel(game, key_prefix="rock_world")
    farms = game_controller.get_public_farms(game)
    if not farms:
        st.info("No NPC farms are attached to this game.")
        return
    tabs = st.tabs([farm["name"] for farm in farms])
    for tab, farm in zip(tabs, farms):
        with tab:
            st.caption(f"Generation {farm['generation']} | {farm['active_count']} active rocks | {farm['rock_count']} lineage records")
            npc_game = game_controller.get_world(game).farm(farm["farm_id"]).game
            render_farm(npc_game, show_hidden_truth=False)
            with section("Offer A Trade", "Request any public active rock; the farmer may accept or reject next turn."):
                _trade_builder(game, farm["farm_id"], game_controller.get_public_farm_rocks(game, farm["farm_id"]))
