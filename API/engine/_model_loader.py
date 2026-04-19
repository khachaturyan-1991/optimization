from __future__ import annotations

from pathlib import Path

import torch

from _model_loader import ModelLoader, ModelDetails, LayerSpec


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
        """Extract a deterministic list of named_modules entries."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        layers: list[LayerSpec] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            layers.append(
                LayerSpec(
                    name=name,
                    op_type=module.__class__.__name__,
                )
            )

        self.details.graph = layers
