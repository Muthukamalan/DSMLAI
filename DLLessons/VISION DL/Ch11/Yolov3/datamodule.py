## Setup Data Loader

# import multiprocessing

# cpu_count = int(multiprocessing.cpu_count() / 2)

# print(cpu_count)

import torch
from typing import Any, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, default_collate

import lightning as pl
import albumentations as A

from torch.utils.data import ConcatDataset

from Yolov3.dataset import YOLODataset


class YOLODataModule(pl.LightningDataModule):
    def __init__(
        self,
        DATASET: str,
        ANCHORS: List[List],
        class_names: List[str],
        IMAGE_SIZE: int = 416,
        TRAIN_IMAGE_SIZES=[416],
        S: dict = {416: [416 // 32, 416 // 16, 416 // 8]},
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transforms: Union[A.Compose, None] = None,
        test_transforms: Union[A.Compose, None] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_csv_path = DATASET + "/8examples.csv"
        self.test_csv_path = DATASET + "/2examples.csv"
        self.IMG_DIR = DATASET + "/images/"
        self.LABEL_DIR = DATASET + "/labels/"

        # self.data_train: Optional[Dataset] = None
        # self.data_val: Optional[Dataset] = None
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.data_train = []

    # def prepare_data(self) -> None:
    #     CIFAR10("./data", train=True, download=True),
    #     CIFAR10("./data", train=False, download=True)

    def setup(self, stage="fit"):
        BASE_IMAGE_SIZE = self.hparams.IMAGE_SIZE

        self.data_train = []
        for tis in self.hparams.TRAIN_IMAGE_SIZES:
            self.data_train.append(
                YOLODataset(
                    self.train_csv_path,
                    transform=self.train_transforms(tis),
                    image_size=tis,
                    S=self.hparams.S[tis],
                    img_dir=self.IMG_DIR,
                    label_dir=self.LABEL_DIR,
                    anchors=self.hparams.ANCHORS,
                    mosaic_prob=0.5,
                )
            )

        # ## Define Combined Loader
        # my_data_train = []

        # for tis in self.hparams.TRAIN_IMAGE_SIZES:
        #     my_data_train.append(
        #         YOLODataset(
        #             self.train_csv_path,
        #             transform=self.train_transforms(tis),
        #             image_size=tis,
        #             S=self.hparams.S[tis],
        #             img_dir=self.IMG_DIR,
        #             label_dir=self.LABEL_DIR,
        #             anchors=self.hparams.ANCHORS,
        #             mosaic_prob=0.8,
        #         )
        #     )

        # self.data_train = ConcatDataset(my_data_train)
        

        self.data_val = YOLODataset(
            self.test_csv_path,
            transform=self.test_transforms,
            S=self.hparams.S[BASE_IMAGE_SIZE],
            image_size=BASE_IMAGE_SIZE,
            img_dir=self.IMG_DIR,
            label_dir=self.LABEL_DIR,
            anchors=self.hparams.ANCHORS,
            mosaic_prob=0.0,
        )

        self.data_train_eval = YOLODataset(
            self.train_csv_path,
            transform=self.test_transforms,
            S=self.hparams.S[BASE_IMAGE_SIZE],
            image_size=BASE_IMAGE_SIZE,
            img_dir=self.IMG_DIR,
            label_dir=self.LABEL_DIR,
            anchors=self.hparams.ANCHORS,
            mosaic_prob=0.0,
        )

    # def custom_collate_fn(self, batch):
    #     """
    #     This function combines a batch of samples into a single tensor.

    #     Args:
    #     batch: A list of samples. Each sample is a tuple of (image, target).

    #     Returns:
    #     A tuple of (image, target). The image is a tensor of size (batch_size, c, h, w).
    #      The target is a tensor of size (batch_size,).
    #     """

    #     # images, targets, S = zip(*batch)
    #     # # images = torch.stack(images, 0)
    #     # return images, targets

    #     # images, targets, Ss = zip(*batch)

    #     # import collections

    #     # # print(len(batch[0]))
    #     # elem = batch[0]

    #     # print("3", isinstance(elem, collections.abc.Sequence))
    #     # print("4", isinstance(elem, tuple))

    #     # images, targets, Ss = [], [], []

    #     # for (image, target, S) in batch:
    #     #     images.append(image)
    #     #     targets.append(target)
    #     #     Ss.append(S)

    #     # return images, default_collate(zip(*targets))

    #     transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

    #     print("")

    #     # len(transpose)

    #     # if isinstance(elem, tuple):
    #     return [transposed[0]] + [default_collate(transposed[1])]

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=custom_data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                # collate_fn=self.custom_collate_fn,
                shuffle=True,
                drop_last=False,
            )
            for custom_data_train in self.data_train
        ]

        # return DataLoader(
        #     dataset=self.data_train,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     # collate_fn=self.custom_collate_fn,
        #     shuffle=False,
        #     drop_last=False,
        # )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # collate_fn=self.custom_collate_fn,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_train_eval,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )


# data_module.prepare_data()
# data_module.setup("fit")

# steps_per_epoch = len(data_module.train_dataloader())


# IMAGE_SIZE = config.IMAGE_SIZE
# DATA_SET_PATH = config.DATASET

# train_dataset = YOLODataset(
#     DATA_SET_PATH + "/train.csv",
#     transform=config.train_transforms,
#     S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
#     img_dir=config.IMG_DIR,
#     label_dir=config.LABEL_DIR,
#     anchors=config.ANCHORS,
#     mosaic_prob=0.8,
# )

# val_dataset = YOLODataset(
#     DATA_SET_PATH + "/test.csv",
#     transform=config.test_transforms,
#     S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
#     img_dir=config.IMG_DIR,
#     label_dir=config.LABEL_DIR,
#     anchors=config.ANCHORS,
#     mosaic_prob=0.0,
# )

# train_eval_dataset = YOLODataset(
#     DATA_SET_PATH + "/train.csv",
#     transform=config.test_transforms,
#     S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
#     img_dir=config.IMG_DIR,
#     label_dir=config.LABEL_DIR,
#     anchors=config.ANCHORS,
#     mosaic_prob=0.0,
# )

# kwargs = {"num_workers": 4, "pin_memory": False}
# train_dataloader = DataLoader(
#     train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False, **kwargs
# )

# val_dataloader = DataLoader(
#     val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False, **kwargs
# )

# test_dataloader = DataLoader(
#     train_eval_dataset,
#     batch_size=config.BATCH_SIZE,
#     shuffle=False,
#     drop_last=False,
#     **kwargs
# )

# steps_per_epoch = len(train_dataloader)