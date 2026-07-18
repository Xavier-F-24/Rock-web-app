"""Gallery and lineage farm views."""

from __future__ import annotations

import streamlit as st

from Rock_AI.visualization.farm_render_adapter import build_farm_rock_views
from Rock_AI.visualization.lineage_render_adapter import build_lineage_figure


def render_farm(
    game, *, selected_ids=(), child_ids=(), mutation_ids=(),
    show_hidden_truth: bool = False,
) -> None:
    gallery_tab, lineage_tab = st.tabs(("Gallery", "Lineage"))
    with gallery_tab:
        views = build_farm_rock_views(
            game,
            selected_parent_ids=selected_ids,
            new_child_ids=child_ids,
            mutation_rock_ids=mutation_ids,
            include_oracle_truth=show_hidden_truth,
        )
        columns_per_row = 4
        for start in range(0, len(views), columns_per_row):
            columns = st.columns(columns_per_row)
            for column, rock in zip(columns, views[start : start + columns_per_row]):
                with column:
                    labels = []
                    if rock.selected_parent:
                        labels.append("Selected")
                    if rock.newly_created:
                        labels.append("New child")
                    if rock.mutated:
                        labels.append("Mutation")
                    if rock.rare_trait:
                        labels.append("Rare")
                    if rock.high_value:
                        labels.append("High value")
                    with st.container(border=True):
                        if rock.image_uri:
                            st.image(rock.image_uri, width="stretch")
                        else:
                            st.warning("Rock image unavailable")
                        st.markdown(f"**{rock.name}**")
                        st.caption(
                            f"#{rock.rock_id} | {rock.sex} | Gen {rock.generation} | "
                            f"${rock.value:.2f} | {rock.status}"
                        )
                        if labels:
                            st.caption(" | ".join(labels))
                        trait_text = ", ".join(f"{name}: {value}" for name, value in rock.phenotype_traits[:6])
                        st.caption(trait_text)
                        if show_hidden_truth:
                            with st.expander("Privileged genotype truth"):
                                st.warning("Developer-only ORACLE_TRUTH. The agent cannot observe this.")
                                st.write("\n".join(rock.genotype_summary))
    with lineage_tab:
        try:
            rare_ids = tuple(
                view.rock_id for view in build_farm_rock_views(
                    game, include_images=False, include_oracle_truth=show_hidden_truth
                ) if view.rare_trait
            )
            figure = build_lineage_figure(
                game,
                selected_parent_ids=selected_ids,
                new_child_ids=child_ids,
                mutation_rock_ids=mutation_ids,
                rare_rock_ids=rare_ids,
            )
            st.plotly_chart(figure, width="stretch", config={"scrollZoom": True})
        except Exception as error:
            st.warning(f"Lineage rendering is unavailable: {error}")
