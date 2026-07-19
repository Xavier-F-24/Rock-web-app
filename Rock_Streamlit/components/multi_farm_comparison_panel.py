import streamlit as st

from Rock_AI.visualization.farmer_comparison_visualizer import farmer_comparison_rows


def render_multi_farm_comparison(world, active_farm_id=None):
    rows = farmer_comparison_rows(world)
    columns = st.columns(len(rows))
    for column, row in zip(columns, rows):
        with column, st.container(border=True):
            prefix = "Active: " if row["farm_id"] == active_farm_id else ""
            st.subheader(prefix + row["name"])
            st.caption(f"{row['farm_id']} | {row['objective']} | generation {row['generation']}")
            st.metric("Cash", f"${row['cash']}", help=f"${row['committed_cash']} committed")
            st.metric("Active rock value", f"${row['active_rock_value']}")
            st.metric("Objective utility", f"{row['objective_utility']:.1f}")
