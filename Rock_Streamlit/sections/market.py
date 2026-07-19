"""Persistent farmer-owned market, direct offers, inbox, and potion shop."""

from __future__ import annotations

import streamlit as st

from Rock_Streamlit.app_state import get_game_state
from Rock_Streamlit.components.world_turn_panel import render_world_turn_panel
from Rock_Streamlit.ui_components import metric_strip, page_header, section
from rockgame_ui import game_controller


MARKET_MESSAGE_KEY = "market_last_action"
MARKET_SECTION_KEY = "market_active_section"
MARKET_SECTIONS = ("Listings", "Family Pods", "Sell / List", "Direct Offers", "Inbox", "Potions", "Owned Rocks")


def _finish(result):
    st.session_state[MARKET_MESSAGE_KEY] = result
    st.rerun()


def _show_message():
    result = st.session_state.pop(MARKET_MESSAGE_KEY, None)
    if result is not None:
        (st.success if result.ok else st.error)(result.message)


def _rock_image(rock, caption=None):
    st.image(game_controller.render_rock(rock), width=180)
    st.caption(caption or game_controller.rock_label(rock))


def _listings(game):
    rows = [row for row in game_controller.get_market_listings(game) if not row["own_listing"]]
    if not rows:
        st.info("No farmer listings are active. End a World Turn to let farmers act.")
        return
    for row in rows:
        with st.container(border=True):
            left, middle, right = st.columns([1, 2, 1])
            with left:
                _rock_image(row["rock"])
            with middle:
                st.markdown(f"**{row['seller_name']}**")
                st.write(f"Asking ${row['asking_price']} | expires turn {row['expires_turn']}")
                st.caption(f"Visible value ${row['rock'].value} | generation {row['rock'].generation} | {row['rock'].status.value}")
            options = game_controller.get_bid_price_options(game, row["listing_id"])
            with right:
                if options:
                    amount = st.selectbox("Bid", options, key=f"listing_bid_amount_{row['listing_id']}")
                    if st.button("Place Bid", key=f"listing_bid_{row['listing_id']}", type="primary"):
                        _finish(game_controller.place_listing_bid(game, row["listing_id"], amount))
                else:
                    st.caption("Not currently affordable.")


def _family_pods(game):
    pods = game_controller.get_family_pod_rows(game)
    if not pods:
        st.info("No real family pods are active. New sibling pods may appear after farmers breed.")
        return
    for pod in pods:
        with st.container(border=True):
            st.markdown(f"**{pod['seller_name']} family pod**")
            st.caption(f"Parents #{pod['parent_ids'][0]} and #{pod['parent_ids'][1]} | ${pod['price']} | expires turn {pod['expires_turn']}")
            columns = st.columns(min(4, len(pod["children"])))
            for index, child in enumerate(pod["children"]):
                with columns[index % len(columns)]:
                    _rock_image(child)
                    if st.button("Purchase", key=f"pod_buy_{pod['pod_id']}_{child.id}"):
                        _finish(game_controller.purchase_family_pod_child(game, pod["pod_id"], child.id))


def _sell_list(game):
    world = game_controller.get_world(game)
    own_listings = [row for row in game_controller.get_market_listings(game) if row["own_listing"]]
    if own_listings:
        st.markdown("**Your active listings**")
        for row in own_listings:
            left, right = st.columns([4, 1])
            left.write(f"#{row['rock'].id} {game_controller.rock_name(row['rock'])} for ${row['asking_price']}")
            if right.button("Cancel", key=f"cancel_listing_{row['listing_id']}"):
                _finish(game_controller.cancel_player_listing(game, row["listing_id"]))
    queued = {rock_id for pair in game.breeding_queue for rock_id in (pair.parent_a_id, pair.parent_b_id)}
    rocks = [rock for rock in game_controller.get_active_rocks(game) if rock.id not in world.reserved_rock_ids and rock.id not in queued]
    if not rocks:
        st.info("No unreserved active rocks are available to list.")
        return
    rock = st.selectbox("Rock to list", rocks, format_func=game_controller.rock_label, key="market_list_rock")
    price = st.selectbox("Asking price", game_controller.get_listing_price_options(game, rock.id), key="market_list_price")
    _rock_image(rock)
    if st.button("Create Listing", key="market_create_listing", type="primary"):
        _finish(game_controller.create_player_listing(game, rock.id, price))


def _direct_offers(game):
    rows = game_controller.get_direct_trade_rows(game, incoming=False)
    if not rows:
        st.info("No outgoing offers. Visit Rock World to propose a trade for a farmer's rock.")
        return
    for row in rows:
        with st.container(border=True):
            st.write(f"{row['other_name']} | {row['status']} | expires turn {row['expires_turn']}")
            st.caption(f"Offered rocks {list(row['offered_rock_ids']) or 'none'} + ${row['offered_money']} | requested rocks {list(row['requested_rock_ids']) or 'none'} + ${row['requested_money']}")
            if row["status"] == "open" and st.button("Cancel Offer", key=f"cancel_offer_{row['offer_id']}"):
                _finish(game_controller.cancel_direct_trade_offer(game, row["offer_id"]))


def _inbox(game):
    messages = game_controller.get_player_messages(game)
    if not messages:
        st.info("No farmer messages yet.")
        return
    for message in messages:
        with st.container(border=True):
            marker = "New" if not message["read"] else "Read"
            st.markdown(f"**{message['sender_name']}** | {marker} | turn {message['turn']}")
            st.write(message["text"])
            if message["requires_response"]:
                accept, reject = st.columns(2)
                if accept.button("Accept", key=f"message_accept_{message['message_id']}", type="primary"):
                    _finish(game_controller.respond_to_message(game, message["message_id"], True))
                if reject.button("Reject", key=f"message_reject_{message['message_id']}"):
                    _finish(game_controller.respond_to_message(game, message["message_id"], False))
            elif not message["read"] and st.button("Mark Read", key=f"message_read_{message['message_id']}"):
                _finish(game_controller.mark_message_read(game, message["message_id"]))


def _potions(game):
    rows = game_controller.get_potion_rows(game)
    quantities = {}
    columns = st.columns(len(rows))
    for column, row in zip(columns, rows):
        with column:
            st.markdown(f"**{row['name']}**")
            st.caption(row["description"])
            st.write(f"${row['cost']} | owned {row['owned']}")
            quantities[row["key"]] = st.number_input("Qty", 0, 99, 0, key=f"market_potion_{row['key']}")
    total = sum(row["cost"] * quantities[row["key"]] for row in rows)
    if st.button(f"Purchase Potions (${total})", disabled=total <= 0, key="market_buy_potions", type="primary"):
        _finish(game_controller.buy_potions(game, quantities))


def render():
    game = get_game_state()
    summary = game_controller.get_game_summary(game)
    page_header("Market", "Every rock here belongs to a real farm in your persistent world.")
    _show_message()
    metric_strip((("Money", f"${summary['money']}"),))
    render_world_turn_panel(game, key_prefix="market")
    selected = st.radio("Market section", MARKET_SECTIONS, horizontal=True, key=MARKET_SECTION_KEY)
    with section(selected):
        if selected == "Listings": _listings(game)
        elif selected == "Family Pods": _family_pods(game)
        elif selected == "Sell / List": _sell_list(game)
        elif selected == "Direct Offers": _direct_offers(game)
        elif selected == "Inbox": _inbox(game)
        elif selected == "Potions": _potions(game)
        else:
            rows = game_controller.get_active_rock_rows(game)
            st.dataframe(rows, width="stretch", hide_index=True) if rows else st.info("No active rocks.")
