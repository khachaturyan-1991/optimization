import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import DataLoder
from model import MobileNetV2

from typing import Dict


class Train:

    def __init__(self, cfg: Dict) -> None:
        """
        Docstring for __init__

        :param cfg: config dictionary
        """
        device_cfg = cfg["train"]["device"]
        if device_cfg == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = MobileNetV2(cfg=cfg["model"]).to(self.device)
        self.train_dataloader, self.test_dataloader = DataLoder(cfg["data"]).get_dataloaders()
        self.epochs = cfg["train"]["epochs"]
        self.output_freq = cfg["train"]["output_freq"]
        self.log_dir = cfg["train"].get("log_dir", "runs")
        self.ckpt_dir = cfg["train"].get("ckpt_dir", "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"], 
            weight_decay=cfg["optimizer"]["weight_decay"]
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self):
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
        self.model.eval()
        with torch.no_grad():
            loss_per_test_step = 0
            for X, y in tqdm(self.test_dataloader, desc="Test", leave=False):
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                loss_per_test_step += self.loss_fn(pred, y).item()
        self.model.train()
        return loss_per_test_step

    def run(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            train_loss = self.train_step()
            test_loss = self.test_step()
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/test", test_loss, epoch)
            if epoch % self.output_freq == 0 and epoch != 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    ckpt_path,
                )
                self.writer.add_text("checkpoint", ckpt_path, epoch)
                print(f"epoch={epoch} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")
        self.writer.close()
