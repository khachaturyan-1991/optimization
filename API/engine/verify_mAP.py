import logging

import torch
import yaml
from model import MobileNetV2
from data_loader import DataLoder
from tqdm import tqdm

def verify():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"

    logging.info("Using device: %s", device)

    # 1. Load model via state_dict (as done in MobileNetV2.__init__)
    model_raw = MobileNetV2(cfg=config["model"]).to(device)
    model_raw.eval()

    # 2. Load model via JIT (as done in Benchmark.__init__)
    ckpt_path = config["model"]["checkpoint_path"]
    logging.info("Loading JIT model from %s", ckpt_path)
    model_jit = torch.jit.load(ckpt_path, map_location=device)
    model_jit.eval()

    _, test_loader = DataLoder(config["data"]).get_dataloaders()

    def get_acc(model, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in tqdm(test_loader, desc=name):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    acc_raw = get_acc(model_raw, "Raw Model")
    acc_jit = get_acc(model_jit, "JIT Model")

    logging.info("Results:")
    logging.info("Raw Model Accuracy: %.4f", acc_raw)
    logging.info("JIT Model Accuracy: %.4f", acc_jit)


if __name__ == "__main__":
    verify()
