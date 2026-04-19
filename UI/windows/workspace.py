import streamlit as st

from config import APP_TITLE
from state import go_to_landing
from tabs import prune


def render_project_status() -> None:
    """Show the current project source at the top of the workspace."""
    if st.session_state.selected_file_path:
        st.success(f"Selected config: {st.session_state.selected_file_path}")
    else:
        st.info("New project created. No config file uploaded yet.")


def render() -> None:
    """Render the main workspace window."""
    st.title(APP_TITLE)
    render_project_status()

    train_tab, benchmark_tab, prune_tab, quantize_tab = st.tabs(
        ["train", "benchmark", "prune", "quantize"]
    )

    with train_tab:
        st.subheader("Train")
        st.write("Training settings and actions will appear here.")

    with benchmark_tab:
        st.subheader("Benchmark")
        st.write("Benchmark settings and results will appear here.")

    with prune_tab:
        prune.render()

    with quantize_tab:
        st.subheader("Quantize")
        st.write("Quantization settings and actions will appear here.")

    if st.button("Back to Start", key="back_to_start"):
        go_to_landing()
        st.rerun()
