from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from shakespeare_ds import ShakespeareDS


import torch 
import pytorch_lightning as pl 
from data_config import data_dir,data_url
from model_config import model_args

class LitDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,data_url) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.data_url = data_url

    def setup(self, stage: str) -> None:
        ...

    def teardown(self, stage: str) -> None:
        ...

    def train_dataloader(self) -> torch.Any:
        torch.utils.data.DataLoader(
            ShakespeareDS(train=True,data_dir=self.data_dir,data_url=self.data_url),
            batch_size=32,
            num_workers = 0
        )

    def  test_dataloader(self) -> TRAIN_DATALOADERS:
        torch.utils.data.DataLoader(
            ShakespeareDS(train=False,data_dir=self.data_dir,data_url=self.data_url),
            batch_size=32,
            num_workers = 0
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        torch.utils.data.DataLoader(
            ShakespeareDS(train=False,data_dir=self.data_dir,data_url=self.data_url),
            batch_size=32,
            num_workers = 0
        )
    
