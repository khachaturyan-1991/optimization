"""Data loader utilities for CIFAR-10."""

import warnings
from typing import Any, Dict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class DataLoder:
    """Create train/test dataloaders from a config dict."""

    def __init__(self, cfg: Dict):
        """Store data config."""
        self.cfg = cfg

    def get_dataloaders(self):
        """Return (train_loader, test_loader) per config limits."""
        self.data_dir = str(self.cfg.get("data_dir", "./DATA"))
        num_train = int(self.cfg.get("num_of_train_img", 0))
        num_test = int(self.cfg.get("num_of_test_img", 0))
        train_bs = int(self.cfg.get("train_batch_size", 1))
        test_bs = int(self.cfg.get("test_batch_size", 1))
        dataset_name = self.cfg.get("dataset", "cifar10").lower()

        if dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            dataset_class = torchvision.datasets.MNIST
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_class = torchvision.datasets.CIFAR10

        train_set = self._load_dataset(
            dataset_class,
            train=True,
            transform=transform,
        )
        test_set = self._load_dataset(
            dataset_class,
            train=False,
            transform=transform,
        )

        if num_train > 0:
            num_train = min(num_train, len(train_set))
            train_set = Subset(train_set, list(range(num_train)))

        if num_test > 0:
            num_test = min(num_test, len(test_set))
            test_set = Subset(test_set, list(range(num_test)))

        train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False)

        return train_loader, test_loader

    def _load_dataset(self, dataset_class: type, *, train: bool, transform: Any):
        """Construct one dataset and suppress the known NumPy 2.4 CIFAR warning."""
        warning_category = getattr(np, "VisibleDeprecationWarning", Warning)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"dtype\(\): align should be passed as Python or NumPy boolean "
                    r"but got `align=0`.*"
                ),
                category=warning_category,
            )
            return dataset_class(
                root=self.data_dir,
                train=train,
                download=True,
                transform=transform,
            )
