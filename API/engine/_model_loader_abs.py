from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TensorSpec:
    name: str | None = None
    shape: tuple[int | str | None, ...] | None = None
    dtype: str | None = None


@dataclass
class LayerSpec:
    name: str
    op_type: str | None = None
    input_shape: tuple[int | str | None, ...] | None = None
    output_shape: tuple[int | str | None, ...] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDetails:
    model_name: str | None = None
    backend: str | None = None
    path: str | None = None
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    graph: list[LayerSpec] = field(default_factory=list)


class ModelLoader(ABC):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.model: Any = None
        self.details = ModelDetails(path=str(self.path))

        self._load_model()
        self._extract_io_details()
        self._extract_graph()

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model from self.path into self.model.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_io_details(self) -> None:
        """
        Extract model input/output names, shapes, and dtypes.
        Store them in self.details.inputs and self.details.outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_graph(self) -> None:
        """
        Extract layer/node-level structure and store it in self.details.graph.
        """
        raise NotImplementedError

    def get_details(self) -> ModelDetails:
        return self.details
