"""Shared CSS helpers for the Streamlit UI."""

import streamlit as st


def apply_global_styles() -> None:
    """Apply shared styles used by the UI screens."""
    st.markdown(
        """
        <style>
            .main > div {
                padding-top: 0;
                padding-bottom: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
