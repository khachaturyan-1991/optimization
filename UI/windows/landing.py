"""Landing window for creating or uploading a project."""

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


def render_home() -> None:
    """Render the centered home screen."""
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top, rgba(84, 160, 255, 0.18), transparent 34%),
                    linear-gradient(180deg, #0b0f14 0%, #080b10 100%);
            }
            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }
            .main > div {
                max-width: 860px;
                min-height: 100vh;
                margin: 0 auto;
                padding: clamp(2rem, 5vw, 3rem);
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            h1.home-title {
                margin: 0 0 1.75rem;
                text-align: center;
                font-size: clamp(2.4rem, 5vw, 3.35rem);
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: #f4f8fc;
            }
            [data-testid="stHorizontalBlock"] {
                align-items: stretch;
            }
            [data-testid="column"] > div {
                height: 100%;
            }
            div[data-testid="stButton"] {
                height: 100%;
            }
            div[data-testid="stButton"] > button {
                width: 100%;
                min-height: 5rem;
                padding: 1rem 1.4rem;
                border-radius: 18px;
                border: 1px solid rgba(104, 167, 255, 0.22);
                background: linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(18, 25, 37, 0.98));
                color: #f4f8fc;
                font-size: 1.15rem;
                font-weight: 700;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
                transition:
                    transform 0.18s ease,
                    border-color 0.18s ease,
                    box-shadow 0.18s ease,
                    background 0.18s ease;
            }
            div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                border-color: rgba(132, 188, 255, 0.56);
                background: linear-gradient(180deg, rgba(36, 52, 75, 0.98), rgba(21, 31, 47, 1));
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.36);
            }
            div[data-testid="stButton"] > button:focus:not(:active) {
                border-color: rgba(132, 188, 255, 0.72);
                box-shadow:
                    0 0 0 0.18rem rgba(84, 160, 255, 0.2),
                    0 18px 40px rgba(0, 0, 0, 0.36);
            }
            p,
            label {
                color: #d2d9e2;
            }
            [data-testid="stFileUploaderDropzone"] {
                border-radius: 18px;
                border: 1px dashed rgba(122, 149, 186, 0.34);
                background: rgba(11, 16, 24, 0.78);
            }
            [data-testid="stAlert"] {
                border-radius: 16px;
            }
            @media (max-width: 900px) {
                .main > div {
                    padding: 1.5rem 1rem 2rem;
                }
                h1.home-title {
                    margin-bottom: 1.25rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<h1 class="home-title">{APP_TITLE}</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button("New Project", key="new_project"):
            go_to_workspace("new")
            st.rerun()
    with col2:
        if st.button("Upload Project", key="upload_project"):
            show_upload_prompt()

    if (
        st.session_state.show_uploader
        or st.session_state.selected_file_path
        or st.session_state.upload_error
    ):
        render_upload_box()
        render_selected_file()


def render() -> None:
    """Render the landing window."""
    render_home()
