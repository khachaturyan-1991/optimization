"""Benchmark utilities for evaluation and visualization."""

import os
from typing import Dict

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data_loader import DataLoder
from model import MobileNetV2

try:
    from API.engine.structured_logging import log_event
except ModuleNotFoundError:
    from structured_logging import log_event


class Benchmark:
    def __init__(self, cfg: Dict) -> None:
        """Initialize model, dataloader, and output paths."""
        device_cfg = cfg["train"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Initialize quantization engine (required for quantized models)
        engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
        if torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = engine

        self.checkpoint_path = cfg.get("model", {}).get("checkpoint_path")
        self.model = None
        self.jit_model = None
        self.quantized_jit = False
        if self.checkpoint_path:
            try:
                self.jit_model = torch.jit.load(self.checkpoint_path, map_location="cpu")
                self.jit_model.eval()
                graph_str = str(self.jit_model.inlined_graph)
                self.quantized_jit = "quantized::" in graph_str
                if self.quantized_jit:
                    if not torch.backends.quantized.supported_engines:
                        raise RuntimeError(
                            "Quantized ops not supported in this PyTorch build. "
                            "Install a build with QuantizedCPU support to benchmark."
                        )
                    self.device = "cpu"
                else:
                    self.jit_model.to(self.device)
                log_event(
                    "checkpoint_loaded",
                    path=self.checkpoint_path,
                    device=self.device,
                )
            except Exception as e:
                try:
                    self.jit_model = torch.jit.load(self.checkpoint_path, map_location="cpu")
                    self.jit_model.eval()
                    self.device = "cpu"
                    log_event(
                        "checkpoint_loaded",
                        path=self.checkpoint_path,
                        device="cpu",
                        fallback_reason=str(e),
                    )
                except Exception as e2:
                    log_event(
                        "checkpoint_load_failed",
                        level="ERROR",
                        path=self.checkpoint_path,
                        error=str(e2),
                    )
                    self.jit_model = None
        if self.jit_model is None:
            from model import get_model
            self.model = get_model(cfg["model"]).to(self.device)
            self.model.eval()
        _, self.test_dataloader = DataLoder(cfg["data"]).get_dataloaders()
        dataset_name = cfg.get("data", {}).get("dataset", "cifar10").lower()
        if "classes" in cfg:
            self.classes = cfg["classes"]
        elif dataset_name == "mnist":
            self.classes = [str(i) for i in range(10)]
        else:
            self.classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        self.save_to = "benchmark"
        self.save_as = cfg.get("benchmark", {}).get("save_as", "benchmark_plot.png")
        os.makedirs(self.save_to, exist_ok=True)
        self.layers_to_investigate = cfg.get("benchmark", {}).get("plot_layer", False)

    def _compute(self):
        """Compute mAP and return sample images/labels/preds."""
        sample_images = None
        sample_labels = None
        sample_preds = None
        if self.jit_model is not None:
            log_event("inference_model_selected", model_type="jit")
            model = self.jit_model
        else:
            log_event("inference_model_selected", model_type="float")
            model = self.model

        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in tqdm(self.test_dataloader, desc="Benchmark", leave=False):
                X = X.to(self.device)
                y = y.to(self.device)
                pred = model(X)
                preds = pred.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
                if sample_images is None:
                    sample_images = X[:9].detach().cpu()
                    sample_labels = y[:9].detach().cpu()
                    sample_preds = preds[:9].detach().cpu()
        loss_mAP = (correct / total) if total > 0 else 0.0
        return loss_mAP, sample_images, sample_labels, sample_preds

    def _make_plot(self, save_as, sample_images, sample_labels, sample_preds):
        """Create and save a 3x3 grid plot with per-image titles."""
        fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=300)
        axes = axes.flatten()
        for i in range(9):
            ax = axes[i]
            if i < len(sample_labels):
                img = sample_images[i].permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    ax.imshow(img.squeeze(-1), cmap="gray")
                else:
                    ax.imshow(img)
                label = self.classes[sample_labels[i]]
                pred = self.classes[sample_preds[i]]
                ax.set_title(f"{label} / {pred}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.save_to, save_as)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_weight_histograms(self):
        """Plot and save weight histograms for model layers."""
        if self.jit_model is not None:
            log_event("weight_histograms_skipped", reason="jit_model")
            return
        layer_names = self.model.get_layer_names()
        name_to_module = dict(self.model.named_modules())
        for layer_name in layer_names:
            module = name_to_module.get(layer_name)
            if module is None or not hasattr(module, "weight"):
                log_event(
                    "layer_skipped",
                    level="DEBUG",
                    layer=layer_name,
                    reason="missing_or_no_weights",
                )
                continue
            weights = module.weight.detach().cpu().flatten().numpy()
            mean_val = float(weights.mean())
            std_val = float(weights.std())

            plt.figure()
            plt.hist(weights, bins=50)
            plt.title(f"mean={mean_val:.6f}, std={std_val:.6f}")
            plt.xlabel("Weight")
            plt.ylabel("Frequency")

            safe_name = layer_name.replace("/", "_")
            out_path = os.path.join(self.save_to, f"{safe_name}.png")
            plt.savefig(out_path)
            plt.close()

    def run(self):
        """Run benchmark evaluation and save visualizations."""
        loss_mAP, sample_images, sample_labels, sample_preds = self._compute()
        log_event("benchmark_result", accuracy=float(loss_mAP))
        if sample_images is not None:
            out_path = self._make_plot(
                self.save_as,
                sample_images,
                sample_labels.tolist(),
                sample_preds.tolist(),
            )
            log_event("plot_saved", path=out_path)

        if self.layers_to_investigate:
            self.plot_weight_histograms()
