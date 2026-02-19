import os
from typing import Dict

import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data_loader import DataLoder
from model import MobileNetV2


class Benchmark:
    def __init__(self, cfg: Dict) -> None:
        device_cfg = cfg["train"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = MobileNetV2(cfg=cfg["model"]).to(self.device)
        self.model.eval()
        _, self.test_dataloader = DataLoder(cfg["data"]).get_dataloaders()
        self.classes = cfg.get(
            "classes",
            [
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
            ],
        )
        self.save_to = cfg.get("benchmark", {}).get("save_to", "benchmark")
        os.makedirs(self.save_to, exist_ok=True)
        self.layers_to_investigate = cfg["benckmark"]["plot_layer"]

    def _compute(self):
        sample_images = None
        sample_labels = None
        sample_preds = None
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in tqdm(self.test_dataloader, desc="Benchmark", leave=False):
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
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
        fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=300)
        axes = axes.flatten()
        for i in range(9):
            ax = axes[i]
            if i < len(sample_labels):
                img = sample_images[i].permute(1, 2, 0).numpy()
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
        layer_names = self.model.get_layer_names()
        name_to_module = dict(self.model.named_modules())
        for layer_name in layer_names:
            module = name_to_module.get(layer_name)
            if module is None or not hasattr(module, "weight"):
                print(f"Skipping {layer_name}: no such layer or no weights")
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
        loss_mAP, sample_images, sample_labels, sample_preds = self._compute()
        print(f"mAP: {loss_mAP:.4f}")
        if sample_images is not None:
            out_path = self._make_plot(
                "benchmark_plot.png",
                sample_images,
                sample_labels.tolist(),
                sample_preds.tolist(),
            )
            print(f"Saved plot: {out_path}")

        if self.layers_to_investigate:
            self.plot_weight_histograms()
