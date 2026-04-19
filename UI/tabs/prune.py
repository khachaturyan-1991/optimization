import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from API.engine.upload_jit import JITModelInspector

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "API" / "data" / "configs" / "config.yml"


def inspect_uploaded_model(uploaded_model) -> None:
    """Persist the uploaded TorchScript file and inspect its modules."""
    suffix = Path(uploaded_model.name).suffix or ".pt"

    previous_path = st.session_state.prune_model_runtime_path
    if previous_path and os.path.exists(previous_path):
        os.unlink(previous_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_model.getbuffer())
        temp_path = temp_file.name

    try:
        inspector = JITModelInspector()
        inspector.load_model(temp_path)
        st.session_state.prune_model_layers = inspector.get_layer_names()
        st.session_state.prune_model_error = None
        st.session_state.prune_model_runtime_path = temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        st.session_state.prune_model_runtime_path = None
        raise


def load_base_config() -> dict:
    """Load the base config used to construct prune runs."""
    config_path = st.session_state.selected_config_runtime_path or str(DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prune_config() -> dict:
    """Map UI fields to the pruning config structure."""
    if not st.session_state.prune_model_runtime_path:
        raise ValueError("Please upload a model file before pruning.")

    try:
        sparsity = float(st.session_state.prune_sparsity)
    except ValueError as exc:
        raise ValueError("Sparsity must be a valid number.") from exc

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in the range [0.0, 1.0).")

    output_model = st.session_state.prune_output_model.strip()
    if not output_model:
        raise ValueError("Please enter an output model path.")

    config = load_base_config()
    pruning_config = config.setdefault("pruning", {})
    pruning_config["ch_sparsity"] = sparsity
    pruning_config["checkpoint_path"] = st.session_state.prune_model_runtime_path
    pruning_config["output_path"] = output_model
    pruning_config["ignore_layers"] = list(st.session_state.prune_ignore_layers)
    return config


def run_pruning() -> None:
    """Run the existing pruning CLI with a generated config file."""
    config = build_prune_config()
    pruning_config = config["pruning"]

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yml", encoding="utf-8"
    ) as temp_config:
        yaml.safe_dump(config, temp_config, sort_keys=False)
        temp_config_path = temp_config.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "API" / "engine" / "main.py"),
                "--prune",
                "--config",
                temp_config_path,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

    combined_output = "\n".join(
        part for part in [result.stdout.strip(), result.stderr.strip()] if part
    )

    if result.returncode != 0:
        raise RuntimeError(combined_output or "Pruning failed.")

    st.session_state.prune_run_error = None
    summary = [
        f"Using pruning.ch_sparsity: {pruning_config['ch_sparsity']}",
        f"Using pruning.checkpoint_path: {pruning_config['checkpoint_path']}",
        f"Using pruning.output_path: {pruning_config['output_path']}",
        f"Using pruning.ignore_layers: {pruning_config['ignore_layers']}",
    ]
    if combined_output:
        summary.append(combined_output)
    else:
        summary.append("Pruning finished successfully.")
    st.session_state.prune_run_output = "\n".join(summary)


def render_architecture() -> None:
    """Render the extracted model architecture as a tree-like list."""
    if st.session_state.prune_model_error:
        st.error(st.session_state.prune_model_error)

    if not st.session_state.prune_model_layers:
        return

    with st.expander("Model architecture", expanded=True):
        for layer in st.session_state.prune_model_layers:
            depth = layer["name"].count(".")
            indent = "&nbsp;" * 4 * depth
            st.markdown(
                f"{indent}- `{layer['name']}` ({layer['type']})",
                unsafe_allow_html=True,
            )


def add_ignore_layer() -> None:
    """Add a layer name to pruning.ignore_layers."""
    layer_name = st.session_state.prune_ignore_layer_input.strip()
    if not layer_name:
        return

    if layer_name not in st.session_state.prune_ignore_layers:
        st.session_state.prune_ignore_layers.append(layer_name)

    st.session_state.prune_ignore_layer_input = ""


def remove_ignore_layer(layer_name: str) -> None:
    """Remove a layer name from pruning.ignore_layers."""
    st.session_state.prune_ignore_layers = [
        name for name in st.session_state.prune_ignore_layers if name != layer_name
    ]


def render_ignore_layers_editor() -> None:
    """Render the manual ignore_layers editor."""
    st.write("ignore_layers")

    input_col, button_col = st.columns([5, 1])
    with input_col:
        st.text_input("Layer name", key="prune_ignore_layer_input", placeholder="features.0")
    with button_col:
        st.write("")
        st.write("")
        st.button("Add", key="add_ignore_layer", on_click=add_ignore_layer)

    if not st.session_state.prune_ignore_layers:
        return

    for layer_name in st.session_state.prune_ignore_layers:
        name_col, remove_col = st.columns([5, 1])
        with name_col:
            st.code(layer_name)
        with remove_col:
            st.button(
                "Remove",
                key=f"remove_ignore_layer_{layer_name}",
                on_click=remove_ignore_layer,
                args=(layer_name,),
            )


def render_model_selector() -> None:
    """Render the model file picker for prune inputs."""
    if st.button("Model", key="prune_model_button"):
        st.session_state.show_prune_model_uploader = True

    if not st.session_state.show_prune_model_uploader:
        return

    uploaded_model = st.file_uploader(
        "Select a model file",
        type=["pt", "pth"],
        accept_multiple_files=False,
        key="prune_model_uploader",
    )

    if uploaded_model is not None:
        st.session_state.prune_model_file = uploaded_model.name
        try:
            inspect_uploaded_model(uploaded_model)
        except Exception as exc:
            st.session_state.prune_model_layers = []
            st.session_state.prune_model_runtime_path = None
            st.session_state.prune_model_error = f"Failed to load model: {exc}"

    if st.session_state.prune_model_file:
        st.caption(f"Selected model: {st.session_state.prune_model_file}")

    render_architecture()


def render() -> None:
    """Render the prune tab content."""
    st.subheader("Prune")
    st.text_input("sparsity", key="prune_sparsity")
    render_model_selector()
    render_ignore_layers_editor()
    st.text_input("output_model", key="prune_output_model")

    if st.button("Prune", key="prune_run_button"):
        try:
            run_pruning()
        except Exception as exc:
            st.session_state.prune_run_output = None
            st.session_state.prune_run_error = str(exc)

    if st.session_state.prune_run_error:
        st.error(st.session_state.prune_run_error)

    if st.session_state.prune_run_output:
        with st.expander("Prune output", expanded=True):
            st.code(st.session_state.prune_run_output)
