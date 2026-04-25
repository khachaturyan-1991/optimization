from __future__ import annotations

import copy
from abc import ABC, abstractmethod
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
        active_report = report if report is not None else self.evaluate_sensitivity()
        selected_layers = self.select_layers(active_report)
        optimized_model = self._clone_model().to(self.device)

        if selected_layers:
            optimized_model = self._apply_optimization(
                optimized_model,
                selected_layers,
                self.final_sparsity,
            )

        return optimized_model, active_report

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

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any]:
        """Validate and unpack one dataloader batch."""
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Dataloader must yield (inputs, targets).")
