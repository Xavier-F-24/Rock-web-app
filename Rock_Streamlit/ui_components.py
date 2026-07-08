"""Reusable Streamlit UI helpers for the rock game."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


def apply_cozy_lab_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #fbfaf7;
            color: #241f1a;
        }
        div[data-testid="stMetric"] {
            background: #fffdf8;
            border: 1px solid #e6dccb;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e6dccb;
            border-radius: 8px;
        }
        .rock-card {
            border: 1px solid #e6dccb;
            border-radius: 8px;
            padding: 0.75rem;
            background: #fffdf8;
            min-height: 100%;
        }
        .rock-card-selected {
            border: 2px solid #8b5a2b;
            background: #fff8ec;
        }
        .section-note {
            color: #6c6258;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


@contextmanager
def section(title: str, caption: str | None = None):
    with st.container(border=True):
        st.subheader(title)
        if caption:
            st.caption(caption)
        yield


def metric_strip(metrics: list[tuple[str, object]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def action_message(result) -> None:
    if result is None:
        return
    if result.ok:
        st.success(result.message)
    else:
        st.error(result.message)


def rock_card(card: dict, key_prefix: str, selected: bool = False) -> bool:
    css_class = "rock-card rock-card-selected" if selected else "rock-card"
    st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    st.image(card["image_uri"], width=150)
    st.markdown(f"**#{card['id']} {card['name']}**")
    st.caption(
        f"{card['sex']} | gen {card['generation']} | {card['status']} | value ${card['value']}"
    )
    clicked = st.button(
        "Selected" if selected else "Select",
        key=f"{key_prefix}_{card['id']}",
        disabled=selected,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked
