"""Training loop for MobileNetV2 on CIFAR-10."""

import os
from datetime import datetime
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import DataLoder
from logs import Logs
from model import MobileNetV2

from typing import Dict


class Train:

    def __init__(self, cfg: Dict) -> None:
        """Initialize model, loaders, optimizer, and logger."""
        device_cfg = cfg["train"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        from model import get_model
        self.model = get_model(cfg["model"]).to(self.device)
        self.train_dataloader, self.test_dataloader = DataLoder(cfg["data"]).get_dataloaders()
        self.epochs = cfg["train"]["epochs"]
        self.output_freq = cfg["train"]["output_freq"]
        logs_cfg = cfg.get("logs", {})
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
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            train_loss = self.train_step()
            test_loss, loss_mAP, sample_images, sample_labels, sample_preds = self.test_step()
            self.logger.log_loss(epoch, train_loss, test_loss, loss_mAP)
            if epoch % self.output_freq == 0 and epoch != 0:
                self.logger.log_predictions(sample_images, sample_labels, sample_preds, epoch)
                ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
                self.model.eval()
                self.model.save_model(ckpt_path)
                self.model.train()
                self.logger.log_text(ckpt_path, epoch)
                print(f"epoch={epoch} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")
        self.logger.writer.close()
