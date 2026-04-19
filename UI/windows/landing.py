import tempfile
from pathlib import Path

import streamlit as st

from config import APP_TITLE
from state import go_to_workspace, set_uploaded_file, show_upload_prompt


def render_selected_file() -> None:
    """Show the selected file or upload validation error."""
    if st.session_state.selected_file_path:
        st.success(f"Selected config: {st.session_state.selected_file_path}")

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)


def render_upload_box() -> None:
    """Render the YAML uploader on the landing screen."""
    if not st.session_state.show_uploader:
        return

    st.write("Select a YAML config file to continue.")
    uploaded_file = st.file_uploader(
        "Select a YAML config file",
        type=["yml", "yaml"],
        accept_multiple_files=False,
        key="config_uploader",
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".yml"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        set_uploaded_file(uploaded_file.name, temp_path)
        st.rerun()


def render() -> None:
    """Render the centered landing window."""
    st.markdown(f'<div class="hero-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="center-shell">', unsafe_allow_html=True)

    _, col1, _, col2, _ = st.columns([1.2, 2.2, 0.4, 2.2, 1.2])
    with col1:
        if st.button("New Project", key="new_project"):
            go_to_workspace("new")
            st.rerun()
    with col2:
        if st.button("Upload Project", key="upload_project"):
            show_upload_prompt()

    st.markdown("</div>", unsafe_allow_html=True)
    render_upload_box()
    render_selected_file()
