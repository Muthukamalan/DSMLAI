from typing import Any, Callable, List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np


class Cifar10DataLoader:
    def __init__(self, batch_size=128, is_cuda_available=False) -> None:
        self.batch_size: int = batch_size
        self.means: List[float] = [0.4914, 0.4822, 0.4465]
        self.stds: List[float] = [0.2470, 0.2435, 0.2616]

        self.dataloader_args = {"shuffle": True, "batch_size": self.batch_size}
        if is_cuda_available:
            self.dataloader_args["num_workers"] = 2
            self.dataloader_args["pin_memory"] = True

        self.classes: List[str] = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def get_dataset(
        self,
        transforms: Optional[Callable],
        train=True,
        data_dir: str = "../../data",
    ):
        return datasets.CIFAR10(
            data_dir,
            train=train,
            transform=transforms,
            download=False,
        )

    def get_loader(self, transforms: Optional[Callable], train=True):
        return DataLoader(self.get_dataset(transforms, train), **self.dataloader_args)

    def get_classes(self):
        return self.classes