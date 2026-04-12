"""Training loop for MobileNetV2 on CIFAR-10."""

import os
from datetime import datetime
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import DataLoder
from logs import Logs

from typing import Dict


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
                self.model = torch.jit.load(ckpt_path, map_location="cpu")
                self.model.to(self.device)
                print(f"Loaded JIT model: {ckpt_path}")
            except Exception as exc:
                print(f"Failed to load JIT model from {ckpt_path}: {exc}")
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
        self.mlflow_cfg = cfg.get("mlflow", {})
        self.mlflow_enabled = bool(self.mlflow_cfg.get("enabled", True))
        self.mlflow = None
        if self.mlflow_enabled:
            try:
                import mlflow
            except Exception as exc:
                raise RuntimeError(
                    "mlflow is required for this training run. "
                    "Install it in the configured venv."
                ) from exc
            tracking_uri = self.mlflow_cfg.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            experiment_name = self.mlflow_cfg.get("experiment_name", "default")
            mlflow.set_experiment(experiment_name)
            self.mlflow = mlflow

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
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.mlflow:
            self.mlflow.start_run(run_name=run_name)
            self.mlflow.log_params({
                "model_name": self.model.__class__.__name__,
                "epochs": self.epochs,
                "optimizer": "SGD",
                "lr": self.optimizer.param_groups[0]["lr"],
                "momentum": self.optimizer.param_groups[0].get("momentum", 0.0),
                "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0.0),
                "train_batch_size": self.train_dataloader.batch_size,
                "test_batch_size": self.test_dataloader.batch_size,
            })
        best_acc = 0.0
        best_state = None
        try:
            for epoch in tqdm(range(self.epochs), desc="Epochs"):
                train_loss = self.train_step()
                test_loss, loss_mAP, sample_images, sample_labels, sample_preds = self.test_step()
                self.logger.log_loss(epoch, train_loss, test_loss, loss_mAP)
                print(f"===== {train_loss}")
                if self.mlflow:
                    self.mlflow.log_metric("loss/train", train_loss, step=epoch)
                    self.mlflow.log_metric("loss/test", test_loss, step=epoch)
                    self.mlflow.log_metric("accuracy", loss_mAP, step=epoch)

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
                    if self.mlflow:
                        self.mlflow.log_artifact(ckpt_path)
                    print(f"epoch={epoch} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")
            if self.mlflow:
                self.mlflow.log_metric("best_accuracy", best_acc)
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
                if self.mlflow:
                    self.mlflow.log_artifact(model_ckpt_path)
                if orig_device.type != "cpu":
                    self.model.to(orig_device)
        finally:
            if self.mlflow:
                self.mlflow.end_run()
            self.logger.writer.close()
