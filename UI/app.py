"""Streamlit entry point for the model optimization UI."""

import streamlit as st

from config import APP_TITLE
from state import init_state
from styles import apply_global_styles
from windows import landing, workspace


def main() -> None:
    """Initialize and render the active UI window."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    apply_global_styles()

    if st.session_state.current_screen == "workspace":
        workspace.render()
    else:
        landing.render()


if __name__ == "__main__":
    main()
