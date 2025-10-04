from typing import Any, Callable, List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np


class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, root="../../data", train=True, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    
class Cifar10DataLoader:
    def __init__(self,batch_size=128, is_cuda_available=False) -> None:
        self.batch_size = batch_size
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

    def get_classes(self):
        return self.classes
    
    def get_loader(self,datasets:datasets.CIFAR10, train=True):
        return DataLoader(datasets, **self.dataloader_args)