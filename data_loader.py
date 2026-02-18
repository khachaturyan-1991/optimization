import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from typing import Dict


class DataLoder:
    """
    loades images from ./DATA
    takes config as input
    sets number of images and batch from config
    returns train, test dataloaders
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def get_dataloaders(self):
        self.data_dir = str(self.cfg.get("data_dir", "./DATA"))
        num_train = int(self.cfg.get("num_of_train_img", 0))
        num_test = int(self.cfg.get("num_of_test_img", 0))
        train_bs = int(self.cfg.get("train_batch_size", 1))
        test_bs = int(self.cfg.get("test_batch_size", 1))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=False, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=False, transform=transform
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
