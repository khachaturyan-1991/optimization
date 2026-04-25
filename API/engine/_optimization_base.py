from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


LayerReport = dict[str, bool | float | str]
SensitivityReport = dict[str, LayerReport]
WITHIN_MAX_ACCURACY_DROP = "within_max_accuracy_drop"
EXCEEDS_MAX_ACCURACY_DROP = "exceeds_max_accuracy_drop"
EVALUATION_FAILED = "evaluation_failed"


class LayerWiseOptimizer(ABC):
    """
    Base class for layer-wise sensitivity analysis and final optimization.

    The original model is never modified. Sensitivity analysis and final
    optimization always operate on fresh deep copies of the source model.
    """

    config_section = "pruning"

    def __init__(
        self,
        model: nn.Module,
        dataloader: Any,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        config: dict[str, Any],
    ) -> None:
        """
        Initialize a layer-wise optimizer.

        Args:
            model: Source model to evaluate and optimize by copy.
            dataloader: Evaluation dataloader yielding ``(inputs, targets)``.
            metric_fn: Callable returning an accuracy or score.
            config: Application configuration dictionary.
        """
        self.model = model
        self.dataloader = dataloader
        self.metric_fn = metric_fn
        self.config = config
        self.optimizer_config = config.get(self.config_section, {}) or {}
        self.device = self._resolve_device(
            str(config.get("device", config.get("train", {}).get("device", "cpu")))
        )
        self.max_accuracy_drop = float(self._config_value("max_accuracy_drop", 0.02))
        self.analysis_sparsity = self._resolve_strength(
            "analysis_sparsity",
            self._config_value("analysis_sparsity", 0.3),
        )
        self.final_sparsity = self._resolve_strength(
            "final_sparsity",
            self._config_value(
                "final_sparsity",
                self._config_value("ch_sparsity", 0.3),
            ),
        )
        self.ignore_layers = {
            str(name) for name in self.optimizer_config.get("ignore_layers", []) or []
        }
        self.supported_layer_types: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
        self._last_report: SensitivityReport | None = None

    def evaluate_sensitivity(self) -> SensitivityReport:
        """
        Evaluate each candidate layer with the configured temporary optimization.

        Returns:
            A report keyed by layer name with recommendation and accuracy details.
        """
        baseline_accuracy = self._compute_accuracy(self._clone_model())
        report: SensitivityReport = {}

        for layer_name in self._get_candidate_layers():
            error_message: str | None = None
            try:
                test_model = self._clone_model()
                test_model = self._apply_optimization(
                    test_model,
                    [layer_name],
                    self.analysis_sparsity,
                )
                optimized_accuracy = self._compute_accuracy(test_model)
                recommended, reason = self._make_decision(
                    baseline_accuracy,
                    optimized_accuracy,
                )
            except Exception as exc:
                optimized_accuracy = baseline_accuracy
                recommended = False
                reason = EVALUATION_FAILED
                error_message = str(exc)

            report[layer_name] = {
                "recommended": recommended,
                "baseline_accuracy": baseline_accuracy,
                "optimized_accuracy": optimized_accuracy,
                "accuracy_drop": baseline_accuracy - optimized_accuracy,
                "reason": reason,
            }
            if error_message is not None:
                report[layer_name]["error"] = error_message

        self._last_report = report
        return report

    def select_layers(self, report: SensitivityReport | None = None) -> list[str]:
        """
        Return layer names recommended by a sensitivity report.

        Args:
            report: Optional report from ``evaluate_sensitivity``.
        """
        active_report = report if report is not None else self.evaluate_sensitivity()
        return [
            layer_name
            for layer_name, layer_report in active_report.items()
            if bool(layer_report.get("recommended", False))
        ]

    def optimize(
        self,
        report: SensitivityReport | None = None,
    ) -> tuple[nn.Module, SensitivityReport]:
        """
        Apply final optimization to all selected layers on a fresh model copy.

        Args:
            report: Optional sensitivity report. If omitted, one is computed.

        Returns:
            The optimized model copy and the sensitivity report used to select layers.
        """
        active_report = report
        if active_report is None:
            active_report = self._load_sensitivity_report_from_recipes()
        if active_report is None:
            active_report = self.evaluate_sensitivity()

        selected_layers = self.select_layers(active_report)
        optimized_model = self._clone_model().to(self.device)

        if selected_layers:
            optimized_model = self._apply_optimization(
                optimized_model,
                selected_layers,
                self.final_sparsity,
            )

        return optimized_model, active_report

    def read_allowed_pruning_layers(
        self,
        path: str | Path,
    ) -> list[str]:
        """
        Read a sensitivity recipe file and return prune-eligible layer names.

        Supported inputs are either a JSON object keyed by layer name or JSON/JSONL
        event records that expose layer decisions through fields such as
        ``layer``/``status`` or ``layer``/``recommended``.
        """
        report = self._read_sensitivity_report(path)
        return self.select_layers(report)

    def _compute_accuracy(self, model: nn.Module) -> float:
        """
        Evaluate a model on the configured device using ``metric_fn``.

        The metric function is called once with concatenated predictions and
        targets. If that fails, it is called per batch and averaged by batch size.
        """
        model = model.to(self.device)
        model.eval()

        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, batch_targets = self._unpack_batch(batch)
                moved_inputs = self._move_to_device(inputs, self.device)
                moved_targets = self._move_to_device(batch_targets, self.device)
                batch_predictions = model(moved_inputs)

                if not torch.is_tensor(batch_predictions):
                    raise TypeError("metric_fn requires tensor predictions.")
                if not torch.is_tensor(moved_targets):
                    raise TypeError("metric_fn requires tensor targets.")

                predictions.append(batch_predictions.detach().cpu())
                targets.append(moved_targets.detach().cpu())

        return self._compute_metric(predictions, targets)

    def _get_candidate_layers(self) -> list[str]:
        """
        Return layer names eligible for layer-wise optimization.

        Initially, only ``nn.Conv2d`` and ``nn.Linear`` layers with weights are
        candidates. Layers listed in ``config['pruning']['ignore_layers']`` are
        skipped.
        """
        candidates: list[str] = []
        for layer_name, module in self.model.named_modules():
            if not layer_name:
                continue
            if layer_name in self.ignore_layers:
                continue
            if not isinstance(module, self.supported_layer_types):
                continue
            if not self._has_weights(module):
                continue
            candidates.append(layer_name)
        return candidates

    def _make_decision(
        self,
        baseline_accuracy: float,
        optimized_accuracy: float,
    ) -> tuple[bool, str]:
        """
        Decide whether the accuracy drop is within the configured threshold.
        """
        accuracy_drop = baseline_accuracy - optimized_accuracy
        if accuracy_drop <= self.max_accuracy_drop:
            return True, WITHIN_MAX_ACCURACY_DROP
        return False, EXCEEDS_MAX_ACCURACY_DROP

    @abstractmethod
    def _apply_optimization(
        self,
        model: nn.Module,
        layer_names: list[str],
        strength: float,
    ) -> nn.Module:
        """
        Apply structured optimization to the provided model copy.

        Args:
            model: Fresh model copy to modify.
            layer_names: Layer names to optimize.
            strength: Optimization strength in ``[0.0, 1.0)``.

        Returns:
            The modified model copy.
        """
        raise NotImplementedError

    def _clone_model(self) -> nn.Module:
        """Return a deep copy of the source model."""
        return copy.deepcopy(self.model)

    def _compute_metric(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> float:
        """Compute the configured metric over collected predictions and targets."""
        if not predictions or not targets:
            raise ValueError("No evaluation batches were produced.")

        try:
            return float(
                self.metric_fn(
                    torch.cat(predictions, dim=0),
                    torch.cat(targets, dim=0),
                )
            )
        except Exception:
            weighted_score = 0.0
            total_examples = 0
            for batch_predictions, batch_targets in zip(predictions, targets):
                if batch_targets.ndim > 0:
                    batch_size = int(batch_targets.shape[0])
                else:
                    batch_size = int(batch_targets.numel())
                batch_size = max(batch_size, 1)
                weighted_score += (
                    float(self.metric_fn(batch_predictions, batch_targets)) * batch_size
                )
                total_examples += batch_size

            if total_examples == 0:
                raise ValueError("Metric function did not receive any examples.")
            return weighted_score / total_examples

    def _get_named_module(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Resolve a named module from the provided model."""
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model copy.")
        return module

    def _has_weights(self, module: nn.Module) -> bool:
        """Return whether a module exposes a non-empty weight tensor."""
        weight = getattr(module, "weight", None)
        return torch.is_tensor(weight) and weight.numel() > 0

    def _move_to_device(self, value: Any, device: torch.device) -> Any:
        """Move tensors or tensor containers onto a device."""
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {
                key: self._move_to_device(item, device)
                for key, item in value.items()
            }
        return value

    def _resolve_device(self, device_name: str) -> torch.device:
        """Resolve the configured device to an available PyTorch device."""
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        if (
            device_name == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_strength(self, name: str, value: Any) -> float:
        """Validate an optimization strength value."""
        strength = float(value)
        if not 0.0 <= strength < 1.0:
            raise ValueError(f"pruning.{name} must be in [0.0, 1.0).")
        return strength

    def _config_value(self, name: str, default: Any) -> Any:
        """Return a config value, treating ``None`` as missing."""
        value = self.optimizer_config.get(name)
        return default if value is None else value

    def _recipes_path(self) -> str | None:
        """Return the configured sensitivity recipe path, if any."""
        recipes_path = self._config_value("recipes", None)
        if recipes_path is None:
            recipes_path = self.config.get("recipes")
        if recipes_path is None:
            return None
        return str(recipes_path)

    def _load_sensitivity_report_from_recipes(self) -> SensitivityReport | None:
        """Load a sensitivity report from the configured recipes file."""
        recipes_path = self._recipes_path()
        if not recipes_path:
            return None
        report = self._read_sensitivity_report(recipes_path)
        self._last_report = report
        return report

    def _read_sensitivity_report(self, path: str | Path) -> SensitivityReport:
        """Read a sensitivity report from JSON or JSONL."""
        report_path = Path(path)
        if not report_path.exists():
            raise FileNotFoundError(
                f"Sensitivity analysis recipes file not found: {report_path}"
            )

        raw_data = self._load_json_or_jsonl(report_path)
        return self._normalize_sensitivity_report(raw_data)

    def _load_json_or_jsonl(self, path: Path) -> Any:
        """Read a JSON object/array or a JSONL stream from disk."""
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Sensitivity analysis recipes file is empty: {path}")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in recipes file {path} at line {line_number}."
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Recipes JSONL file {path} must contain JSON objects only."
                    )
                records.append(record)

            if not records:
                raise ValueError(f"Sensitivity analysis recipes file is empty: {path}")
            return records

    def _normalize_sensitivity_report(self, raw_data: Any) -> SensitivityReport:
        """Convert supported recipe payloads into a sensitivity report."""
        if isinstance(raw_data, dict):
            return self._report_from_mapping(raw_data)
        if isinstance(raw_data, list):
            return self._report_from_records(raw_data)
        raise ValueError("Unsupported recipes format for sensitivity analysis.")

    def _report_from_mapping(self, raw_data: dict[str, Any]) -> SensitivityReport:
        """Build a sensitivity report from a mapping payload."""
        if "report" in raw_data and isinstance(raw_data["report"], dict):
            raw_data = raw_data["report"]

        if "layers" in raw_data and isinstance(raw_data["layers"], (dict, list)):
            layers_payload = raw_data["layers"]
            if isinstance(layers_payload, dict):
                return self._report_from_mapping(layers_payload)
            return self._report_from_records(layers_payload)

        report: SensitivityReport = {}
        candidate_layers = set(self._get_candidate_layers())
        for layer_name, layer_report in raw_data.items():
            if not isinstance(layer_name, str):
                continue
            if candidate_layers and layer_name not in candidate_layers:
                continue
            if not isinstance(layer_report, dict):
                continue
            report[layer_name] = self._normalize_layer_report(layer_name, layer_report)

        if report:
            return report
        raise ValueError("Recipes file does not contain a valid sensitivity report.")

    def _report_from_records(self, records: list[Any]) -> SensitivityReport:
        """Build a sensitivity report from event-style records."""
        report: SensitivityReport = {}
        candidate_layers = set(self._get_candidate_layers())

        for record in records:
            if not isinstance(record, dict):
                continue
            layer_name = record.get("layer") or record.get("layer_name")
            if not isinstance(layer_name, str):
                continue
            if candidate_layers and layer_name not in candidate_layers:
                continue
            report[layer_name] = self._normalize_layer_report(layer_name, record)

        if report:
            return report
        raise ValueError("Recipes file does not contain any recognized layer entries.")

    def _normalize_layer_report(
        self,
        layer_name: str,
        raw_report: dict[str, Any],
    ) -> LayerReport:
        """Normalize one layer decision record into the standard report schema."""
        recommended = self._recommended_from_layer_report(raw_report)
        baseline_accuracy = self._float_or_default(
            raw_report.get("baseline_accuracy"),
            0.0,
        )
        optimized_accuracy = self._float_or_default(
            raw_report.get("optimized_accuracy"),
            baseline_accuracy,
        )
        accuracy_drop = self._float_or_default(
            raw_report.get("accuracy_drop"),
            baseline_accuracy - optimized_accuracy,
        )
        reason = str(
            raw_report.get(
                "reason",
                WITHIN_MAX_ACCURACY_DROP
                if recommended
                else EXCEEDS_MAX_ACCURACY_DROP,
            )
        )

        normalized: LayerReport = {
            "recommended": recommended,
            "baseline_accuracy": baseline_accuracy,
            "optimized_accuracy": optimized_accuracy,
            "accuracy_drop": accuracy_drop,
            "reason": reason,
        }
        if "error" in raw_report:
            normalized["error"] = str(raw_report["error"])
        normalized["layer"] = layer_name
        return normalized

    def _recommended_from_layer_report(self, raw_report: dict[str, Any]) -> bool:
        """Infer whether a layer is allowed for pruning from a raw record."""
        if "recommended" in raw_report:
            return self._bool_value(raw_report["recommended"])
        if "allow_pruning" in raw_report:
            return self._bool_value(raw_report["allow_pruning"])
        if "allowed_for_pruning" in raw_report:
            return self._bool_value(raw_report["allowed_for_pruning"])

        status = raw_report.get("status")
        if isinstance(status, str):
            return status.lower() in {"selected", "recommended", "allowed", "keep"}

        reason = raw_report.get("reason")
        if isinstance(reason, str):
            return reason == WITHIN_MAX_ACCURACY_DROP

        return False

    def _float_or_default(self, value: Any, default: float) -> float:
        """Return a float value or a fallback when conversion is not possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _bool_value(self, value: Any) -> bool:
        """Return a stable boolean for native or string-like JSON values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        """Validate and unpack one dataloader batch."""
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Dataloader must yield (inputs, targets).")
