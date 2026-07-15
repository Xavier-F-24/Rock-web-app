"""Compact model and runtime compatibility metadata."""

from __future__ import annotations

import streamlit as st


def render_model_info(session) -> None:
    with st.expander("Model and runtime information"):
        configuration = session.agent.configuration() if session.agent is not None else {"agent_type": "replay"}
        checkpoint = session.checkpoint_metadata or {}
        st.json(
            {
                "agent": configuration,
                "checkpoint_path": checkpoint.get("ranker_checkpoint_path"),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "validation_metrics": checkpoint.get("validation_metrics"),
                "encoding_schema_version": checkpoint.get("encoding_schema_version"),
                "dataset_schema_version": checkpoint.get("dataset_schema_version"),
                "model_architecture": checkpoint.get("model_architecture_config"),
                "objective_profile": session.objective_profile.to_dict(),
                "inference_device": checkpoint.get("device", "agent default"),
            }
        )
