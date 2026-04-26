"""Guided pruning workflow for the Streamlit UI."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import streamlit as st
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "API" / "data" / "configs" / "config.yml"


def _get_loader_torch_jit() -> type[Any]:
    """Return the TorchScript loader after making the API package importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    module = import_module("API.engine._model_loader")
    return cast(type[Any], getattr(module, "LoaderTorchJit"))


def _get_prune_api() -> Any:
    """Return the pruning module used by the UI workflow."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return import_module("API.engine.prune")


def _format_shape(shape: Any) -> str | None:
    """Render a layer shape tuple in a compact display form."""
    if not shape:
        return None
    return "x".join("?" if dim is None else str(dim) for dim in shape)


def _append_log(message: str) -> None:
    """Append a short message to the read-only prune log."""
    logs = list(st.session_state.prune_logs)
    logs.append(message)
    st.session_state.prune_logs = logs[-12:]


def _uploaded_file_signature(uploaded_file: Any) -> tuple[str, int]:
    """Return a stable signature for one uploaded file selection."""
    return (str(getattr(uploaded_file, "name", "")), len(uploaded_file.getbuffer()))


def _reset_prune_workflow() -> None:
    """Clear analysis and result state after model changes."""
    st.session_state.prune_analysis_data = None
    st.session_state.prune_analysis_json = None
    st.session_state.prune_analysis_path = None
    st.session_state.prune_analysis_import_error = None
    st.session_state.prune_results = None
    st.session_state.prune_run_error = None
    st.session_state.prune_run_output = None
    st.session_state.prune_protected_layers = []
    st.session_state.prune_protected_layers_input = ""


def _layers_from_loader(loader: Any) -> list[dict[str, str | None]]:
    """Normalize loader graph details for the prune UI."""
    layers: list[dict[str, str | None]] = []
    for layer in loader.get_details().graph:
        if is_dataclass(layer):
            layer_data = asdict(layer)
        elif isinstance(layer, dict):
            layer_data = layer
        else:
            layer_data = {
                "name": getattr(layer, "name", ""),
                "op_type": getattr(layer, "op_type", None),
                "input_shape": getattr(layer, "input_shape", None),
                "output_shape": getattr(layer, "output_shape", None),
            }

        name = str(layer_data.get("name") or "").strip()
        if not name:
            continue

        layers.append(
            {
                "name": name,
                "type": str(layer_data.get("op_type") or "Unknown"),
                "input_shape": _format_shape(layer_data.get("input_shape")),
                "output_shape": _format_shape(layer_data.get("output_shape")),
            }
        )
    return layers


def _inspect_model(uploaded_model: Any) -> None:
    """Persist the uploaded TorchScript file and inspect its modules."""
    suffix = Path(uploaded_model.name).suffix or ".pt"
    previous_path = st.session_state.prune_model_runtime_path
    if previous_path and os.path.exists(previous_path):
        os.unlink(previous_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_model.getbuffer())
        temp_path = temp_file.name

    try:
        loader_class = _get_loader_torch_jit()
        loader = loader_class(temp_path)
        layers = _layers_from_loader(loader)

        try:
            loaded_model = torch.jit.load(temp_path, map_location="cpu")
            param_count = sum(parameter.numel() for parameter in loaded_model.parameters())
        except Exception:
            param_count = None

        st.session_state.prune_model_runtime_path = temp_path
        st.session_state.prune_model_file = uploaded_model.name
        st.session_state.prune_model_layers = layers
        st.session_state.prune_model_summary = {
            "model_path": uploaded_model.name,
            "layer_count": len(layers),
            "param_count": param_count,
        }
        st.session_state.prune_model_error = None
        _reset_prune_workflow()
        _append_log(f"Loaded weights: {uploaded_model.name}")
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        st.session_state.prune_model_runtime_path = None
        st.session_state.prune_model_layers = []
        st.session_state.prune_model_summary = None
        raise


def _load_base_config() -> dict[str, Any]:
    """Load the base config used to construct prune runs."""
    config_path = st.session_state.selected_config_runtime_path or str(DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as file:
        return cast(dict[str, Any], yaml.safe_load(file) or {})


def _build_runtime_config(
    *,
    require_output_path: bool,
    sparsity_key: str = "prune_analysis_sparsity",
) -> dict[str, Any]:
    """Map current UI values to the pruning config structure."""
    if not st.session_state.prune_model_runtime_path:
        raise ValueError("Load model weights before running analysis or pruning.")

    try:
        sparsity = float(st.session_state[sparsity_key])
    except ValueError as exc:
        raise ValueError("Sparsity must be a valid number.") from exc

    try:
        analysis_threshold = float(st.session_state.prune_analysis_threshold)
    except ValueError as exc:
        raise ValueError("Analysis threshold must be a valid number.") from exc

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in the range [0.0, 1.0).")
    if analysis_threshold < 0.0:
        raise ValueError("Analysis threshold must be greater than or equal to 0.0.")

    config = _load_base_config()
    pruning_config = config.setdefault("pruning", {})
    pruning_config["ch_sparsity"] = sparsity
    pruning_config["final_sparsity"] = sparsity
    pruning_config["max_accuracy_drop"] = analysis_threshold
    pruning_config["checkpoint_path"] = st.session_state.prune_model_runtime_path
    output_model = st.session_state.prune_output_model.strip()
    if require_output_path:
        if not output_model:
            raise ValueError("Enter an output model path.")
        pruning_config["output_path"] = output_model
    elif output_model:
        pruning_config["output_path"] = output_model
    return config


def _format_param_count(value: int | None) -> str:
    """Render a parameter count for display."""
    if value is None:
        return "Unknown"
    return f"{value:,}"


def _format_ratio(value: float | None) -> str:
    """Render a ratio as a percentage string."""
    if value is None:
        return "Unknown"
    return f"{value * 100:.2f}%"


def _parse_protected_layers(text: str) -> list[str]:
    """Parse comma or newline separated protected-layer names."""
    layers: list[str] = []
    for token in text.replace("\n", ",").split(","):
        layer_name = token.strip()
        if layer_name and layer_name not in layers:
            layers.append(layer_name)
    return layers


def _sync_protected_layers_input() -> None:
    """Update the visible protected-layer field from canonical state."""
    st.session_state.prune_protected_layers_input = ", ".join(
        st.session_state.prune_protected_layers
    )


def _set_protected_layers(layers: list[str]) -> None:
    """Replace the protected-layer selection and sync the input field."""
    st.session_state.prune_protected_layers = list(dict.fromkeys(layers))
    _sync_protected_layers_input()


def _analysis_layers() -> list[dict[str, Any]]:
    """Return analysis rows from session state."""
    data = st.session_state.prune_analysis_data or {}
    layers = data.get("layers", [])
    return layers if isinstance(layers, list) else []


def _protected_layers_from_analysis(data: dict[str, Any]) -> list[str]:
    """Extract the suggested protected layers from an analysis payload."""
    configured = data.get("protected_layers")
    if isinstance(configured, list):
        return [str(layer) for layer in configured if str(layer).strip()]

    layers: list[str] = []
    for entry in data.get("layers", []):
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("suggested_protected")):
            layer_name = str(entry.get("layer", "")).strip()
            if layer_name and layer_name not in layers:
                layers.append(layer_name)
    return layers


def _set_analysis_data(data: dict[str, Any], *, source_path: str | None = None) -> None:
    """Persist an analysis payload and sync the protected-layer selection."""
    st.session_state.prune_analysis_data = data
    st.session_state.prune_analysis_json = json.dumps(data, indent=2)
    st.session_state.prune_analysis_path = source_path
    st.session_state.prune_analysis_import_error = None
    _set_protected_layers(_protected_layers_from_analysis(data))


def _checkbox_key(layer_name: str) -> str:
    """Return the checkbox key for one layer row."""
    sanitized = layer_name.replace(".", "_").replace("[", "_").replace("]", "_")
    return f"prune_protected_{sanitized}"


def _apply_text_field_selection() -> None:
    """Sync manual protected-layer edits into the row selection state."""
    layers = _parse_protected_layers(st.session_state.prune_protected_layers_input)
    _set_protected_layers(layers)


def _apply_checkbox_selection() -> None:
    """Rebuild protected layers from row checkboxes and sync the text field."""
    layers: list[str] = []
    for entry in _analysis_layers():
        layer_name = str(entry.get("layer", "")).strip()
        if not layer_name:
            continue
        if bool(st.session_state.get(_checkbox_key(layer_name), False)):
            layers.append(layer_name)
    _set_protected_layers(layers)


def _set_selection_mode(mode: str) -> None:
    """Apply one of the quick selection actions to the current analysis rows."""
    current = set(st.session_state.prune_protected_layers)
    suggested = {
        str(entry.get("layer", "")).strip()
        for entry in _analysis_layers()
        if bool(entry.get("suggested_protected"))
    }
    all_layers = {
        str(entry.get("layer", "")).strip()
        for entry in _analysis_layers()
        if str(entry.get("layer", "")).strip()
    }

    if mode == "suggested":
        next_layers = [layer for layer in suggested if layer]
    elif mode == "clear":
        next_layers = []
    elif mode == "invert":
        next_layers = [layer for layer in all_layers if layer and layer not in current]
    else:
        next_layers = list(current)

    _set_protected_layers(sorted(next_layers))


def _load_previous_analysis(uploaded_file: Any) -> None:
    """Read and validate a structured analysis JSON file."""
    payload = cast(dict[str, Any], json.loads(uploaded_file.getvalue().decode("utf-8")))
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Analysis JSON must contain a top-level 'layers' list.")
    _set_analysis_data(payload, source_path=uploaded_file.name)
    _append_log(f"Loaded previous analysis: {uploaded_file.name}")


def _run_analysis() -> None:
    """Execute sensitivity analysis and persist the structured JSON result."""
    st.session_state.prune_run_error = None
    prune_api = _get_prune_api()
    config = _build_runtime_config(
        require_output_path=False,
        sparsity_key="prune_analysis_sparsity",
    )

    with st.spinner("Running sensitivity analysis..."):
        result = prune_api.analyze_with_config(config)

    analysis_data = cast(dict[str, Any], result["result"])
    _set_analysis_data(analysis_data, source_path=str(result["analysis_path"]))
    _append_log(f"Analysis saved: {result['analysis_path']}")


def _run_pruning() -> None:
    """Execute structured pruning with the current protected-layer list."""
    protected_layers = list(st.session_state.prune_protected_layers)
    if not protected_layers:
        raise ValueError("Define at least one protected layer before pruning.")

    st.session_state.prune_run_error = None
    prune_api = _get_prune_api()
    config = _build_runtime_config(
        require_output_path=True,
        sparsity_key="prune_analysis_sparsity",
    )

    with st.spinner("Running structured pruning..."):
        result = prune_api.prune_with_protected_layers(config, protected_layers)

    st.session_state.prune_results = cast(dict[str, Any], result)
    st.session_state.prune_run_error = None
    _append_log(f"Pruned model saved: {result['output_path']}")


def _render_model_section() -> None:
    """Render section 1: model loading and summary."""
    st.markdown("### 1. Model")
    load_clicked = st.button("Load Weights", key="prune_load_weights")
    if load_clicked:
        st.session_state.show_prune_model_uploader = True
        st.session_state.prune_model_upload_signature = None
        st.session_state.prune_model_error = None
        st.session_state.pop("prune_model_uploader", None)

    if st.session_state.show_prune_model_uploader:
        uploaded_model = st.file_uploader(
            "Select a TorchScript weights file",
            type=["pt", "pth"],
            accept_multiple_files=False,
            key="prune_model_uploader",
        )
        if uploaded_model is not None:
            signature = _uploaded_file_signature(uploaded_model)
            if signature != st.session_state.get("prune_model_upload_signature"):
                st.session_state.prune_model_upload_signature = signature
                try:
                    _inspect_model(uploaded_model)
                    st.session_state.show_prune_model_uploader = False
                except Exception as exc:
                    st.session_state.prune_model_error = f"Failed to load model: {exc}"

    if st.session_state.prune_model_error:
        st.error(st.session_state.prune_model_error)

    summary = st.session_state.prune_model_summary
    if not summary:
        st.caption("Load weights to enable analysis and pruning.")
        return

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Model", str(summary["model_path"]))
    info_col2.metric("Layers", str(summary["layer_count"]))
    info_col3.metric("Params", _format_param_count(summary["param_count"]))

    with st.expander("Model architecture", expanded=False):
        for layer in st.session_state.prune_model_layers:
            st.markdown(f"`{layer['name']}`")


def _render_analysis_section() -> None:
    """Render section 2: optional analysis and review."""
    st.markdown("### 2. Sensetivity analyses")
    st.caption("Estimate sensitivity and suggest layers to protect.")

    disabled = not bool(st.session_state.prune_model_runtime_path)
    action_col1, action_col2, action_col3, action_col4 = st.columns([1.8, 1.8, 1.4, 2.0])
    with action_col1:
        st.text_input(
            "Threshold",
            key="prune_analysis_threshold",
            help="Equivalent to pruning.max_accuracy_drop for the analysis step.",
        )
    with action_col2:
        st.text_input(
            "Sparsity",
            key="prune_analysis_sparsity",
            help="Equivalent to pruning.ch_sparsity for the analysis step.",
        )
    with action_col3:
        if st.button("Run Analysis", key="prune_run_analysis", disabled=disabled):
            try:
                _run_analysis()
            except Exception as exc:
                st.session_state.prune_run_error = str(exc)
    with action_col4:
        if st.button(
            "Load Previous Analysis",
            key="prune_load_analysis",
            disabled=disabled,
        ):
            st.session_state.show_prune_analysis_loader = True
            st.session_state.prune_analysis_upload_signature = None
            st.session_state.prune_analysis_import_error = None
            st.session_state.pop("prune_analysis_uploader", None)

    if st.session_state.show_prune_analysis_loader:
        uploaded_analysis = st.file_uploader(
            "Select analysis_result.json",
            type=["json"],
            accept_multiple_files=False,
            key="prune_analysis_uploader",
        )
        if uploaded_analysis is not None:
            signature = _uploaded_file_signature(uploaded_analysis)
            if signature != st.session_state.get("prune_analysis_upload_signature"):
                st.session_state.prune_analysis_upload_signature = signature
                try:
                    _load_previous_analysis(uploaded_analysis)
                    st.session_state.show_prune_analysis_loader = False
                except Exception as exc:
                    st.session_state.prune_analysis_import_error = str(exc)

    if st.session_state.prune_analysis_import_error:
        st.error(st.session_state.prune_analysis_import_error)

    analysis_data = st.session_state.prune_analysis_data
    if not analysis_data:
        st.caption("Analysis is optional, but it helps prefill the protected-layer selection.")
        return

    export_col1, export_col2 = st.columns([3, 2])
    with export_col1:
        source = st.session_state.prune_analysis_path or "Current session"
        st.caption(f"Analysis source: {source}")
    with export_col2:
        st.download_button(
            "Export Analysis JSON",
            data=st.session_state.prune_analysis_json or "{}",
            file_name="analysis_result.json",
            mime="application/json",
            key="prune_export_analysis",
        )

    quick1, quick2, quick3 = st.columns(3)
    quick1.button(
        "Select suggested",
        key="prune_select_suggested",
        on_click=_set_selection_mode,
        args=("suggested",),
    )
    quick2.button(
        "Clear all",
        key="prune_clear_selection",
        on_click=_set_selection_mode,
        args=("clear",),
    )
    quick3.button(
        "Invert selection",
        key="prune_invert_selection",
        on_click=_set_selection_mode,
        args=("invert",),
    )

    header = st.columns([4, 2, 2, 3])
    header[0].markdown("**Layer name**")
    header[1].markdown("**Accuracy drop**")
    header[2].markdown("**Suggested**")
    header[3].markdown("**Reason**")

    for entry in _analysis_layers():
        layer_name = str(entry.get("layer", "")).strip()
        if not layer_name:
            continue

        checkbox_key = _checkbox_key(layer_name)
        st.session_state[checkbox_key] = layer_name in set(
            st.session_state.prune_protected_layers
        )

        row = st.columns([4, 2, 2, 3])
        row[0].code(layer_name)
        row[1].write(f"{float(entry.get('accuracy_drop', 0.0)):.6f}")
        row[2].checkbox(
            "Protect",
            key=checkbox_key,
            label_visibility="collapsed",
            on_change=_apply_checkbox_selection,
        )
        row[3].caption(str(entry.get("reason", "")))


def _render_protected_layers_section() -> None:
    """Render section 3: final source-of-truth protected layers."""
    st.markdown("### 3. Protected Layers")
    st.text_area(
        "Protected layers",
        key="prune_protected_layers_input",
        placeholder="layer_a, layer_b",
        help="Comma-separated or one layer per line. This field is the final source of truth.",
        on_change=_apply_text_field_selection,
        height=110,
    )
    count = len(st.session_state.prune_protected_layers)
    st.caption(f"{count} protected layer(s) will be applied during pruning.")


def _render_pruning_section() -> None:
    """Render section 4: explicit pruning action."""
    st.markdown("### 4. Pruning")
    st.text_input("Output model path", key="prune_output_model")
    st.caption("Pruning uses the sparsity value from section 2.")

    disabled = not bool(st.session_state.prune_model_runtime_path) or not bool(
        st.session_state.prune_protected_layers
    )
    if st.button("Run Pruning", key="prune_run_button", disabled=disabled):
        try:
            _run_pruning()
        except Exception as exc:
            st.session_state.prune_run_error = str(exc)

    if not st.session_state.prune_protected_layers:
        st.warning("Define at least one protected layer to enable pruning.")


def _render_results_section() -> None:
    """Render section 5: pruning results and small log panel."""
    st.markdown("### 5. Results")

    results = st.session_state.prune_results
    if results:
        summary = cast(dict[str, Any], results.get("summary", {}))
        metrics = st.columns(4)
        metrics[0].metric("Params before", _format_param_count(summary.get("params_before")))
        metrics[1].metric("Params after", _format_param_count(summary.get("params_after")))
        metrics[2].metric(
            "Reduction",
            _format_ratio(summary.get("params_reduction_ratio")),
        )
        metrics[3].metric("Output model", str(results.get("output_path", "")))

        st.caption(f"Summary JSON: {results.get('summary_path', '')}")
        st.caption(f"Layer decisions JSONL: {results.get('layer_decisions_path', '')}")
    else:
        st.caption("No pruning results yet.")

    if st.session_state.prune_run_error:
        st.error(st.session_state.prune_run_error)

    with st.expander("Log", expanded=False):
        if st.session_state.prune_logs:
            st.code("\n".join(st.session_state.prune_logs))
        else:
            st.caption("No log entries yet.")


def render() -> None:
    """Render the prune tab content."""
    st.subheader("Prune")
    st.caption("Load → Analyse (optional) → Review/Edit → Prune")

    _render_model_section()
    st.divider()
    _render_analysis_section()
    st.divider()
    _render_protected_layers_section()
    st.divider()
    _render_pruning_section()
    st.divider()
    _render_results_section()
