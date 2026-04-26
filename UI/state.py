"""Session-state helpers for the Streamlit UI."""

from pathlib import Path

import streamlit as st

from config import ALLOWED_SUFFIXES


def init_state() -> None:
    """Initialize persistent UI state."""
    st.session_state.setdefault("selected_file_path", None)
    st.session_state.setdefault("selected_file_name", None)
    st.session_state.setdefault("selected_config_runtime_path", None)
    st.session_state.setdefault("upload_error", None)
    st.session_state.setdefault("show_uploader", False)
    st.session_state.setdefault("project_action", None)
    st.session_state.setdefault("current_screen", "landing")
    st.session_state.setdefault("prune_sparsity", "0.3")
    st.session_state.setdefault("prune_analysis_threshold", "0.1")
    st.session_state.setdefault("prune_output_model", "")
    st.session_state.setdefault("prune_model_file", None)
    st.session_state.setdefault("prune_model_runtime_path", None)
    st.session_state.setdefault("show_prune_model_uploader", False)
    st.session_state.setdefault("prune_model_layers", [])
    st.session_state.setdefault("prune_model_summary", None)
    st.session_state.setdefault("prune_model_error", None)
    st.session_state.setdefault("prune_run_error", None)
    st.session_state.setdefault("prune_run_output", None)
    st.session_state.setdefault("prune_logs", [])
    st.session_state.setdefault("prune_analysis_data", None)
    st.session_state.setdefault("prune_analysis_json", None)
    st.session_state.setdefault("prune_analysis_path", None)
    st.session_state.setdefault("prune_analysis_import_error", None)
    st.session_state.setdefault("prune_protected_layers", [])
    st.session_state.setdefault("prune_protected_layers_input", "")
    st.session_state.setdefault("prune_results", None)
    st.session_state.setdefault("show_prune_analysis_loader", False)


def set_uploaded_file(file_name: str, file_path: str | None = None) -> None:
    """Persist uploaded file metadata after validation."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        st.session_state.upload_error = "Please select a valid YAML file (.yml or .yaml)."
        return

    st.session_state.selected_file_name = file_name
    st.session_state.selected_file_path = file_path or file_name
    st.session_state.selected_config_runtime_path = file_path
    st.session_state.upload_error = None
    st.session_state.current_screen = "workspace"
    st.session_state.show_uploader = False


def go_to_workspace(project_action: str) -> None:
    """Switch to the workspace screen."""
    st.session_state.project_action = project_action
    st.session_state.show_uploader = False
    st.session_state.current_screen = "workspace"


def show_upload_prompt() -> None:
    """Reveal the YAML uploader on the landing screen."""
    st.session_state.project_action = "upload"
    st.session_state.show_uploader = True


def go_to_landing() -> None:
    """Return to the landing screen."""
    st.session_state.current_screen = "landing"
    st.session_state.show_uploader = False
