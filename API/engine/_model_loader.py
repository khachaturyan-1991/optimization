from __future__ import annotations

import re
from pathlib import Path

import torch

try:
    from API.engine._model_loader_abs import LayerSpec, ModelLoader
except ModuleNotFoundError:
    from _model_loader_abs import LayerSpec, ModelLoader


class LoaderTorchJit(ModelLoader):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

    def _load_model(self) -> None:
        """Load a TorchScript model from disk and switch it to eval mode."""
        self.model = torch.jit.load(str(self.path), map_location="cpu")
        self.model.eval()
        self.details.backend = "torchscript"
        self.details.model_name = self.path.stem

    def _extract_io_details(self) -> None:
        """
        TorchScript does not reliably expose full input/output metadata.
        Keep this empty for now and fill it later if you add shape inference.
        """
        self.details.inputs = []
        self.details.outputs = []

    def _extract_graph(self) -> None:
        """Extract named modules and best-effort shape metadata from TorchScript."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        scope_shapes = self._extract_scope_shapes()
        layers: list[LayerSpec] = []
        for name, module in self.model.named_modules():
            if not name:
                continue

            shape_details = scope_shapes.get(name, {})
            layers.append(
                LayerSpec(
                    name=name,
                    op_type=module.__class__.__name__,
                    input_shape=shape_details.get("input_shape"),
                    output_shape=shape_details.get("output_shape"),
                )
            )

        self.details.graph = layers

    def _extract_scope_shapes(
        self,
    ) -> dict[str, dict[str, tuple[int | str | None, ...] | None]]:
        """Map TorchScript node scopes back to named_modules with tensor shapes when available."""
        graph = getattr(self.model, "inlined_graph", None) or getattr(self.model, "graph", None)
        if graph is None:
            return {}

        module_names = {name for name, _ in self.model.named_modules() if name}
        scope_shapes: dict[str, dict[str, tuple[int | str | None, ...] | None]] = {}

        for node in graph.nodes():
            scope_name = self._normalize_scope_name(node.scopeName(), module_names)
            if not scope_name:
                continue

            input_shape = self._find_tensor_shape(node.inputs())
            output_shape = self._find_tensor_shape(node.outputs())

            if input_shape is None and output_shape is None:
                continue

            scope_data = scope_shapes.setdefault(
                scope_name,
                {"input_shape": None, "output_shape": None},
            )
            if scope_data["input_shape"] is None:
                scope_data["input_shape"] = input_shape
            if output_shape is not None:
                scope_data["output_shape"] = output_shape

        return scope_shapes

    @staticmethod
    def _find_tensor_shape(values) -> tuple[int | str | None, ...] | None:
        """Return the first statically known tensor shape from a TorchScript value list."""
        for value in values:
            value_type = value.type()
            sizes = getattr(value_type, "sizes", None)
            if not callable(sizes):
                continue

            shape = sizes()
            if shape is None:
                continue

            return tuple(shape)
        return None

    @staticmethod
    def _normalize_scope_name(scope_name: str, module_names: set[str]) -> str | None:
        """Normalize TorchScript scope names to the model.named_modules() naming scheme."""
        if not scope_name:
            return None

        candidates: list[str] = []
        for part in reversed(scope_name.split("/")):
            cleaned = re.sub(r"^(?:__module\.)+", "", part).strip(".")
            if cleaned:
                candidates.append(cleaned)

        for candidate in candidates:
            if candidate in module_names:
                return candidate

        return None
