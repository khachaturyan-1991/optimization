"""Post-training static quantization pipeline."""

import os
import torch
from typing import Dict

from data_loader import DataLoder
from model import MobileNetV2


class Quantizer:

    def __init__(self, cfg: Dict) -> None:
        """Initialize model, dataloader, and quantization config."""

        self.model = MobileNetV2(cfg=cfg["model"])
        self.model.eval()
        self.model.to("cpu")
        self.train_dataloader, _ = DataLoder(cfg["data"]).get_dataloaders()
        quant_cfg = cfg.get("quantization") or cfg.get("qunatization") or {}
        self.layers_to_keep_fp32 = quant_cfg["layers_to_keep_fp32"]
        self.num_calibration_batches = quant_cfg.get("num_calibration_batches", 1024)
        self.checkpoint_path = quant_cfg["checkpoint_path"]

    def prepare_model(self):
        """
        Fuses the layers and prepares the model for static quantization.
        """
        self.model.to("cpu")
        engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
        torch.backends.quantized.engine = engine
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(engine)

        for name, module in self.model.named_modules():
            for kept_name in self.layers_to_keep_fp32:
                if name == kept_name or name.startswith(kept_name + "."):
                    module.qconfig = None
                    print(f"Set module {name} to qconfig=None (FP32)")

        if hasattr(self.model, "_fuse_model"):
            self.model._fuse_model()

        torch.ao.quantization.prepare(self.model, inplace=True)

    def calibrate_model(self, num_batches=10):
        """
        Runs a few batches of data through the model to let observers
        calculate the activation ranges.
        """
        self.model.eval()
        self.model.to("cpu")

        with torch.no_grad():
            for i, (images, _) in enumerate(self.train_dataloader):
                if i >= num_batches:
                    break
                self.model(images.to("cpu"))

    def _save_quantized_model(self, model, checkpoint_path):
        """Save quantized JIT model only."""
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, checkpoint_path)
        print(f"Quantized JIT model saved to {checkpoint_path}")

    def run(self):
        """Run prepare, calibrate, convert, and save quantized model."""
        self.prepare_model()
        num_cal = self.num_calibration_batches
        print(f"Running calibration with {num_cal} batches...")
        self.calibrate_model(num_batches=num_cal)

        final_model = torch.ao.quantization.convert(self.model, inplace=True)
        self.model.save_model(self.checkpoint_path)
        # self._save_quantized_model(final_model, self.checkpoint_path)
