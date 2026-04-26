from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

try:
    import torch_pruning as tp
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    tp = None

try:
    from API.engine._optimization_base import (
        EVALUATION_FAILED,
        EXCEEDS_MAX_ACCURACY_DROP,
        WITHIN_MAX_ACCURACY_DROP,
        LayerWiseOptimizer,
        SensitivityReport,
    )
    from API.engine.data_loader import DataLoder
    from API.engine.model import get_model
    from API.engine.structured_logging import (
        append_jsonl,
        build_event_record,
        configure_json_logging,
        ensure_json_logging,
        log_event,
        write_json,
    )
except ModuleNotFoundError:
    from _optimization_base import (
        EVALUATION_FAILED,
        EXCEEDS_MAX_ACCURACY_DROP,
        WITHIN_MAX_ACCURACY_DROP,
        LayerWiseOptimizer,
        SensitivityReport,
    )
    from data_loader import DataLoder
    from model import get_model
    from structured_logging import (
        append_jsonl,
        build_event_record,
        configure_json_logging,
        ensure_json_logging,
        log_event,
        write_json,
    )


LAYER_DECISIONS_FILE = "layer_decisions.jsonl"
PRUNING_SUMMARY_FILE = "pruning_summary.json"
ANALYSIS_RESULT_FILE = "analysis_result.json"


class PruningOptimizer(LayerWiseOptimizer):
    """
    Layer-wise optimizer that uses dependency-aware structured pruning.

    Channels/features are physically removed through ``torch-pruning``. The
    implementation never uses unstructured pruning or zero-weight masking.
    """

    def _apply_optimization(
        self,
        model: nn.Module,
        layer_names: list[str],
        strength: float,
    ) -> nn.Module:
        """
        Apply structured pruning to the requested layers on a model copy.

        Args:
            model: Model copy to prune.
            layer_names: Names of ``nn.Conv2d`` or ``nn.Linear`` layers to prune.
            strength: Fraction of output channels/features to remove per layer.

        Returns:
            The structurally pruned model copy.
        """
        if tp is None:
            raise ImportError("torch-pruning is required for structured pruning.")

        if not 0.0 <= strength < 1.0:
            raise ValueError("pruning strength must be in [0.0, 1.0).")
        if not layer_names or strength == 0.0:
            return model

        model = model.to(self.device)
        model.eval()
        example_inputs = self._example_inputs()

        for layer_name in dict.fromkeys(layer_names):
            module = self._get_prunable_module(model, layer_name)
            pruning_idxs = self._select_pruning_indices(module, strength)
            if not pruning_idxs:
                raise ValueError(
                    f"Layer '{layer_name}' cannot be structurally pruned."
                )
            self._prune_module(model, module, pruning_idxs, example_inputs)

        return model

    def _example_inputs(self) -> Any:
        """Return one evaluation input batch for dependency graph tracing."""
        try:
            batch = next(iter(self.dataloader))
        except StopIteration as exc:
            raise ValueError("Dataloader must provide at least one batch.") from exc

        inputs, _ = self._unpack_batch(batch)
        return self._move_to_device(inputs, self.device)

    def _get_prunable_module(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Resolve and validate a layer selected for structured pruning."""
        module = self._get_named_module(model, layer_name)
        if not isinstance(module, self.supported_layer_types):
            raise TypeError(f"Layer '{layer_name}' is not a supported pruning target.")
        if not self._has_weights(module):
            raise ValueError(
                f"Layer '{layer_name}' does not expose a prunable weight tensor."
            )
        return module

    def _select_pruning_indices(self, module: nn.Module, strength: float) -> list[int]:
        """
        Select low-magnitude output channels/features for structured pruning.

        At least one channel is removed when possible, and the method always
        keeps at least one output channel/feature so the layer remains valid.
        """
        output_dim = self._output_dimension(module)
        if output_dim <= 1:
            return []

        prune_count = min(max(1, int(output_dim * strength)), output_dim - 1)
        scores = self._channel_scores(module)
        return torch.argsort(scores)[:prune_count].tolist()

    def _channel_scores(self, module: nn.Module) -> torch.Tensor:
        """Return L2 magnitude scores for each output channel/feature."""
        weight = module.weight.detach()
        return weight.pow(2).flatten(1).sum(dim=1)

    def _output_dimension(self, module: nn.Module) -> int:
        """Return the output channel/feature dimension for a supported layer."""
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)
        if isinstance(module, nn.Linear):
            return int(module.out_features)
        raise TypeError(f"Unsupported pruning target type: {type(module).__name__}")

    def _prune_module(
        self,
        model: nn.Module,
        module: nn.Module,
        pruning_idxs: list[int],
        example_inputs: Any,
    ) -> None:
        """Prune a layer through a dependency graph so dependent layers stay valid."""
        dependency_graph = tp.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
        )
        pruning_fn = self._pruning_function(module)

        if hasattr(dependency_graph, "get_pruning_group"):
            pruning_group = dependency_graph.get_pruning_group(
                module,
                pruning_fn,
                idxs=pruning_idxs,
            )
            if not dependency_graph.check_pruning_group(pruning_group):
                raise RuntimeError("Structured pruning group is not valid.")
            pruning_group.prune()
            return

        pruning_plan = dependency_graph.get_pruning_plan(
            module,
            pruning_fn,
            idxs=pruning_idxs,
        )
        if not dependency_graph.check_pruning_plan(pruning_plan):
            raise RuntimeError("Structured pruning plan is not valid.")
        pruning_plan.exec()

    def _pruning_function(self, module: nn.Module) -> Any:
        """Return the torch-pruning channel pruning function for a layer."""
        if isinstance(module, nn.Conv2d):
            return tp.prune_conv_out_channels
        if isinstance(module, nn.Linear):
            return tp.prune_linear_out_channels
        raise TypeError(f"Unsupported pruning target type: {type(module).__name__}")


def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the newest epoch checkpoint in a training checkpoint directory."""
    candidates = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _checkpoint_path(cfg: dict[str, Any]) -> str:
    """Resolve the checkpoint path used as the pruning input."""
    pruning_cfg = cfg.get("pruning", {}) or {}
    ckpt_path = pruning_cfg.get("checkpoint_path")
    if not ckpt_path:
        ckpt_dir = cfg.get("train", {}).get("ckpt_dir", "checkpoints")
        ckpt_path = _find_latest_checkpoint(str(ckpt_dir))

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No checkpoint found. Set pruning.checkpoint_path or ensure checkpoints exist."
        )
    return str(ckpt_path)


def _load_model_for_pruning(cfg: dict[str, Any], ckpt_path: str) -> nn.Module:
    """Build the configured Python model and load weights from the JIT checkpoint."""
    model_cfg = copy.deepcopy(cfg.get("model", {}) or {})
    model_cfg["checkpoint_path"] = ckpt_path
    model = get_model(model_cfg)
    model.eval()
    return model


def _classification_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 classification accuracy for logits and integer labels."""
    if predictions.ndim == 1:
        predicted_labels = predictions
    else:
        predicted_labels = predictions.argmax(dim=1)
    targets = targets.reshape(-1)
    predicted_labels = predicted_labels.reshape(-1)
    if targets.numel() == 0:
        return 0.0
    return float((predicted_labels == targets).float().mean().item())


def _output_layer_names(model: nn.Module) -> set[str]:
    """Find final classification layers that should not have output classes pruned."""
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        return set()
    return {linear_layers[-1][0]}


def _config_with_default_ignored_layers(
    cfg: dict[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Return a config copy with final classifier layers ignored by default."""
    runtime_cfg = copy.deepcopy(cfg)
    pruning_cfg = runtime_cfg.setdefault("pruning", {})
    configured_ignored = [
        str(layer_name) for layer_name in pruning_cfg.get("ignore_layers", []) or []
    ]
    ignored_layers = list(
        dict.fromkeys(configured_ignored + sorted(_output_layer_names(model)))
    )
    pruning_cfg["ignore_layers"] = ignored_layers
    return runtime_cfg


def _count_params(model: nn.Module) -> int:
    """Return the number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def _save_pruned_model(
    model: nn.Module,
    output_path: str,
    dataloader: Any,
) -> None:
    """Trace and save the pruned model as TorchScript."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("Dataloader must provide at least one batch.") from exc

    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        example_inputs = batch[0]
    else:
        raise ValueError("Dataloader must yield input tensors.")

    model = model.to("cpu")
    model.eval()
    example_inputs = example_inputs.to("cpu")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs)
    torch.jit.save(traced_model, output_path)


def _decision_reason(layer_report: dict[str, Any]) -> str:
    """Return the stable reason enum required by layer decision logs."""
    reason = str(layer_report.get("reason", ""))
    if reason in {WITHIN_MAX_ACCURACY_DROP, EXCEEDS_MAX_ACCURACY_DROP}:
        return reason
    if reason == EVALUATION_FAILED:
        return EXCEEDS_MAX_ACCURACY_DROP
    if bool(layer_report.get("recommended")):
        return WITHIN_MAX_ACCURACY_DROP
    return EXCEEDS_MAX_ACCURACY_DROP


def _reset_jsonl(path: Path) -> None:
    """Create or truncate a JSONL artifact for the current run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _log_report(report: SensitivityReport, layer_decisions_path: Path) -> None:
    """Write layer decisions as structured JSONL events."""
    for layer_name, layer_report in report.items():
        status = "selected" if layer_report["recommended"] else "skipped"
        drop = float(layer_report["accuracy_drop"])
        fields: dict[str, Any] = {
            "layer": layer_name,
            "status": status,
            "accuracy_drop": drop,
            "reason": _decision_reason(layer_report),
        }
        if layer_report.get("reason") == EVALUATION_FAILED:
            fields["error"] = str(layer_report.get("error", ""))
        record = build_event_record("layer_evaluated", **fields)
        append_jsonl(layer_decisions_path, record)
        log_event("layer_evaluated", **fields)


def _write_pruning_summary(
    summary_path: Path,
    before_params: int,
    after_params: int,
) -> None:
    """Write the pruning summary as a structured JSON object."""
    params_removed = before_params - after_params
    reduction_ratio = params_removed / before_params if before_params else 0.0
    fields = {
        "params_before": int(before_params),
        "params_after": int(after_params),
        "params_removed": int(params_removed),
        "params_reduction_ratio": float(reduction_ratio),
    }
    record = build_event_record("pruning_summary", **fields)
    write_json(summary_path, record)
    log_event("pruning_summary", **fields)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _artifact_paths(run_dir: Path) -> tuple[Path, Path]:
    """Return pruning artifact paths for the current run."""
    return (
        run_dir / LAYER_DECISIONS_FILE,
        run_dir / PRUNING_SUMMARY_FILE,
    )


def _analysis_result_path(run_dir: Path) -> Path:
    """Return the structured analysis artifact path for the current run."""
    return run_dir / ANALYSIS_RESULT_FILE


def _build_analysis_result(
    report: SensitivityReport,
    protected_layers: list[str],
) -> dict[str, Any]:
    """Build a structured sensitivity-analysis payload for UI consumption."""
    configured_protected_set = {str(layer_name) for layer_name in protected_layers}
    suggested_protected_layers = list(configured_protected_set)

    for layer_name, layer_report in report.items():
        if not bool(layer_report.get("recommended", False)):
            if layer_name not in configured_protected_set:
                suggested_protected_layers.append(layer_name)

    suggested_protected_set = set(suggested_protected_layers)
    layers: list[dict[str, Any]] = []
    for layer_name, layer_report in report.items():
        layers.append(
            {
                "layer": layer_name,
                "accuracy_drop": float(layer_report.get("accuracy_drop", 0.0)),
                "reason": str(layer_report.get("reason", "")),
                "allowed_for_pruning": bool(layer_report.get("recommended", False)),
                "suggested_protected": layer_name in suggested_protected_set,
                "baseline_accuracy": float(layer_report.get("baseline_accuracy", 0.0)),
                "optimized_accuracy": float(
                    layer_report.get("optimized_accuracy", 0.0)
                ),
                "error": str(layer_report.get("error", "")) if layer_report.get("error") else None,
            }
        )

    return build_event_record(
        "analysis_result",
        protected_layers=suggested_protected_layers,
        layers=layers,
    )


def _write_analysis_result(
    analysis_path: Path,
    report: SensitivityReport,
    protected_layers: list[str],
) -> dict[str, Any]:
    """Write the structured analysis payload to JSON and return it."""
    record = _build_analysis_result(report, protected_layers)
    write_json(analysis_path, record)
    return record


def _explicit_selection_report(
    optimizer: PruningOptimizer,
    protected_layers: list[str],
) -> SensitivityReport:
    """Build a transparent report for explicit UI-driven pruning choices."""
    protected_set = {str(layer_name) for layer_name in protected_layers}
    report: SensitivityReport = {}
    for layer_name, module in optimizer.model.named_modules():
        if not layer_name:
            continue
        if not isinstance(module, optimizer.supported_layer_types):
            continue
        if not optimizer._has_weights(module):
            continue
        recommended = layer_name not in protected_set
        report[layer_name] = {
            "recommended": recommended,
            "baseline_accuracy": 0.0,
            "optimized_accuracy": 0.0,
            "accuracy_drop": 0.0,
            "reason": (
                "explicit_user_selection"
                if recommended
                else "explicit_user_protected_layer"
            ),
        }
    return report


def _create_runtime_pruning_context(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], nn.Module, Any, str]:
    """Load the configured model, runtime config, dataloader, and checkpoint path."""
    ckpt_path = _checkpoint_path(cfg)
    model = _load_model_for_pruning(cfg, ckpt_path)
    runtime_cfg = _config_with_default_ignored_layers(cfg, model)
    _, test_dataloader = DataLoder(runtime_cfg["data"]).get_dataloaders()
    return runtime_cfg, model, test_dataloader, ckpt_path


def analyze_with_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run sensitivity analysis only and persist a structured JSON report."""
    run_dir = ensure_json_logging(cfg, workflow="prune-analysis")
    analysis_path = _analysis_result_path(run_dir)

    runtime_cfg, model, test_dataloader, _ = _create_runtime_pruning_context(cfg)
    protected_layers = [
        str(layer_name)
        for layer_name in runtime_cfg.get("pruning", {}).get("ignore_layers", []) or []
    ]

    optimizer = PruningOptimizer(
        model=model,
        dataloader=test_dataloader,
        metric_fn=_classification_accuracy,
        config=runtime_cfg,
    )
    report = optimizer.evaluate_sensitivity()
    result = _write_analysis_result(analysis_path, report, protected_layers)
    log_event("analysis_saved", path=analysis_path)
    return {"run_dir": str(run_dir), "analysis_path": str(analysis_path), "result": result}


def prune_with_protected_layers(
    cfg: dict[str, Any],
    protected_layers: list[str],
) -> dict[str, Any]:
    """Run pruning using exactly the provided protected-layer list."""
    runtime_cfg, model, test_dataloader, ckpt_path = _create_runtime_pruning_context(cfg)
    pruning_cfg = runtime_cfg.setdefault("pruning", {})
    pruning_cfg["checkpoint_path"] = ckpt_path
    pruning_cfg["ignore_layers"] = [str(layer_name) for layer_name in protected_layers]

    run_dir = ensure_json_logging(runtime_cfg, workflow="prune")
    layer_decisions_path, summary_path = _artifact_paths(run_dir)
    _reset_jsonl(layer_decisions_path)

    output_path = str(pruning_cfg.get("output_path", "pruned_model.pt"))
    if protected_layers:
        log_event("layers_protected", layers=[str(layer) for layer in protected_layers])

    before_params = _count_params(model)
    optimizer = PruningOptimizer(
        model=model,
        dataloader=test_dataloader,
        metric_fn=_classification_accuracy,
        config=runtime_cfg,
    )
    selected_layers = optimizer._get_candidate_layers()
    pruned_model = optimizer._clone_model().to(optimizer.device)
    if selected_layers:
        pruned_model = optimizer._apply_optimization(
            pruned_model,
            selected_layers,
            optimizer.final_sparsity,
        )
    report = _explicit_selection_report(optimizer, protected_layers)
    after_params = _count_params(pruned_model)

    _log_report(report, layer_decisions_path)
    _write_pruning_summary(summary_path, before_params, after_params)
    _save_pruned_model(pruned_model, output_path, test_dataloader)
    log_event("model_saved", path=output_path)

    summary_record = _read_json(summary_path)
    return {
        "run_dir": str(run_dir),
        "output_path": output_path,
        "summary_path": str(summary_path),
        "layer_decisions_path": str(layer_decisions_path),
        "summary": summary_record,
        "report": report,
    }


def prune_with_config(cfg: dict[str, Any]) -> None:
    """Run layer-wise pruning from a loaded application config."""
    protected_layers = [
        str(layer_name)
        for layer_name in (cfg.get("pruning", {}) or {}).get("ignore_layers", []) or []
    ]
    prune_with_protected_layers(cfg, protected_layers)


def main() -> None:
    """Run pruning directly from this module."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    configure_json_logging(cfg, workflow="prune")
    prune_with_config(cfg)


if __name__ == "__main__":
    main()
