"""Ranked candidate table and selected-parent comparison."""

from __future__ import annotations

import streamlit as st

from Rock_AI.visualization.decision_plot_helper import candidate_table_rows
from Rock_AI.visualization.farm_render_adapter import build_farm_rock_views


def render_parent_comparison(game, explanation) -> None:
    if explanation is None or not explanation.selected_parent_ids:
        return
    views = {str(row.rock_id): row for row in build_farm_rock_views(game, selected_parent_ids=explanation.selected_parent_ids)}
    columns = st.columns(2)
    for column, rock_id in zip(columns, explanation.selected_parent_ids):
        rock = views.get(str(rock_id))
        if rock is None:
            continue
        with column, st.container(border=True):
            if rock.image_uri:
                st.image(rock.image_uri, width="stretch")
            st.markdown(f"**{rock.name}**")
            st.caption(f"#{rock.rock_id} | ${rock.value:.2f} | Gen {rock.generation}")
            st.write(", ".join(f"{name}: {value}" for name, value in rock.phenotype_traits[:8]))


def render_candidate_pairs(explanation, *, limit: int = 20) -> None:
    st.subheader("Ranked Candidate Pairs")
    if explanation is None or not explanation.top_candidates:
        st.caption("Candidate scores appear after an agent decision.")
        return
    rows = candidate_table_rows(
        explanation.top_candidates,
        selected_parent_ids=explanation.selected_parent_ids or (),
    )[:limit]
    st.dataframe(rows, width="stretch", hide_index=True, key="ai_obs_candidate_table")
    inspect_options = {
        f"#{row['rank']} {row['parent_a']} + {row['parent_b']}": row for row in rows
    }
    selected_label = st.selectbox("Inspect candidate", tuple(inspect_options), key="ai_obs_inspect_candidate")
    st.json(inspect_options[selected_label])
