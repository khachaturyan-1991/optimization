"""Training loop for MobileNetV2 on CIFAR-10."""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import DataLoder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logs import Logs
from typing import Dict

from _model_loader import LoaderTorchJit

try:
    from API.engine.structured_logging import log_event
except ModuleNotFoundError:
    from structured_logging import log_event


class Train:

    def __init__(self, cfg: Dict) -> None:
        """Initialize model, loaders, optimizer, and logger."""
        self.cfg = cfg
        device_cfg = cfg["train"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        from model import get_model
        ckpt_path = cfg.get("model", {}).get("checkpoint_path")
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                model = LoaderTorchJit(ckpt_path)
                self.model = model.model
                self.model.to(self.device)
                log_event("checkpoint_loaded", path=ckpt_path)
            except Exception as exc:
                log_event(
                    "checkpoint_load_failed",
                    level="ERROR",
                    path=ckpt_path,
                    error=str(exc),
                )
                self.model = get_model(cfg["model"]).to(self.device)
        else:
            self.model = get_model(cfg["model"]).to(self.device)
        self.train_dataloader, self.test_dataloader = DataLoder(cfg["data"]).get_dataloaders()
        self.epochs = cfg["train"]["epochs"]
        self.output_freq = cfg["train"]["output_freq"]
        logs_cfg = cfg.get("logs", {})
        dataset_name = cfg.get("data", {}).get("dataset", "cifar10").lower()
        if not logs_cfg.get("classes") and dataset_name == "mnist":
            logs_cfg["classes"] = [str(i) for i in range(10)]
        base_log_dir = logs_cfg.get("log_dir", "runs")
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_log_dir, run_id)
        self.ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = Logs({"log_dir": self.log_dir, "classes": logs_cfg.get("classes")})
        self.logger.writer.add_text("config", str(cfg))
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"], 
            weight_decay=cfg["optimizer"]["weight_decay"]
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self):
        """Run one training epoch and return loss."""
        self.model.train()
        loss_per_trian_step = 0
        for X, y in tqdm(self.train_dataloader, desc="Train", leave=False):
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            loss_per_trian_step += loss.item()
        return loss_per_trian_step

    def test_step(self):
        """Run one evaluation epoch and return loss, mAP, and sample batch."""
        self.model.eval()
        sample_images = None
        sample_labels = None
        sample_preds = None
        with torch.no_grad():
            loss_per_test_step = 0
            correct = 0
            total = 0
            for X, y in tqdm(self.test_dataloader, desc="Test", leave=False):
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                loss_per_test_step += self.loss_fn(pred, y).item()
                preds = pred.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
                if sample_images is None:
                    sample_images = X[:5].detach().cpu()
                    sample_labels = y[:5].detach().cpu()
                    sample_preds = preds[:5].detach().cpu()
        self.model.train()
        loss_mAP = (correct / total) if total > 0 else 0.0
        return loss_per_test_step, loss_mAP, sample_images, sample_labels, sample_preds

    def run(self):
        """Execute training loop and checkpointing."""
        best_acc = 0.0
        best_state = None
        try:
            for epoch in tqdm(range(self.epochs), desc="Epochs"):
                train_loss = self.train_step()
                test_loss, loss_mAP, sample_images, sample_labels, sample_preds = self.test_step()
                self.logger.log_loss(epoch, train_loss, test_loss, loss_mAP)
                self.logger.log_learning_rate(epoch, self.optimizer.param_groups[0]["lr"])
                self.logger.log_weights(self.model, epoch)
                log_event("train_loss", epoch=int(epoch), train_loss=float(train_loss))

                if loss_mAP > best_acc:
                    best_acc = loss_mAP
                    best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

                if epoch % self.output_freq == 0 and epoch != 0:
                    self.logger.log_predictions(sample_images, sample_labels, sample_preds, epoch)
                    ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
                    self.model.eval()
                    # Trace on CPU to avoid device-mismatch issues during JIT save.
                    orig_device = next(self.model.parameters()).device
                    if orig_device.type != "cpu":
                        self.model.to("cpu")
                    if hasattr(self.model, "save_model"):
                        self.model.save_model(ckpt_path)
                    else:
                        torch.jit.save(self.model, ckpt_path)
                    if orig_device.type != "cpu":
                        self.model.to(orig_device)
                    self.model.train()
                    self.logger.log_text(ckpt_path, epoch)
                    log_event(
                        "training_epoch",
                        epoch=int(epoch),
                        train_loss=float(train_loss),
                        test_loss=float(test_loss),
                    )
            self.logger.writer.add_scalar("metrics/best_accuracy", best_acc, self.epochs)
            model_ckpt_path = self.cfg.get("model", {}).get("checkpoint_path")
            if model_ckpt_path and best_state is not None:
                self.model.eval()
                orig_device = next(self.model.parameters()).device
                if orig_device.type != "cpu":
                    self.model.to("cpu")
                self.model.load_state_dict(best_state, strict=True)
                if hasattr(self.model, "save_model"):
                    self.model.save_model(model_ckpt_path)
                else:
                    torch.jit.save(self.model, model_ckpt_path)
                self.logger.log_text(model_ckpt_path, self.epochs)
                if orig_device.type != "cpu":
                    self.model.to(orig_device)
        finally:
            self.logger.writer.close()
