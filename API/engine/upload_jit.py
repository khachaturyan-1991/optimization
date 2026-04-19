import torch


class JITModelInspector:
    def __init__(self) -> None:
        self.model = None

    def load_model(self, path: str):
        """Load a TorchScript model from disk and switch it to eval mode."""
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()
        return self.model

    def get_layer_names(self):
        """Return a deterministic list of named_modules entries."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        layers = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            layers.append({"name": name, "type": module.__class__.__name__})
        return layers
