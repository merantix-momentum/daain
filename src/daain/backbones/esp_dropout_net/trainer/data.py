import os
import pickle
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from daain.backbones.esp_dropout_net.trainer.data_statistics import DataStatistics
from daain.backbones.esp_dropout_net.trainer.dataset_collate import ConcatTupleDataset, DatasetCollate
from daain.backbones.esp_dropout_net.trainer.dataset_collate import MultipleRandomSampler
from daain.backbones.esp_dropout_net.trainer.dataset_collate import MultipleSequentialSampler
from daain.backbones.esp_dropout_net.trainer.transformations import Compose, Normalize, RandomFlip
from daain.backbones.esp_dropout_net.trainer.transformations import RandomResizedCrop, Resize, ToTensor
from daain.config_schema.datasets.dataset import DatasetPaths
from daain.data.datasets import get_split_sizes
from daain.data.datasets.cityscapes_dataset import Cityscapes


def _create_transforms_(mean, std, scale_in):
    """Create the tranformations used in the original paper.
    Note that this will then take some time to run."""
    default_post = (
        ToTensor(scale_in),
        Normalize(mean=mean, std=std),
    )

    training_transforms = [
        (Resize((512, 1024)),),
        (Resize((512, 1024)), RandomResizedCrop(32), RandomFlip()),
        (Resize((768, 1536)), RandomResizedCrop(128), RandomFlip()),
        (Resize((720, 1280)), RandomResizedCrop(128), RandomFlip()),
        (Resize((384, 768)), RandomResizedCrop(32), RandomFlip()),
        (Resize((256, 512)), RandomFlip()),
    ]

    val_transforms = [(Resize((512, 1024)),)]

    training_transforms = [Compose((*ts, *default_post)) for ts in training_transforms]
    val_transforms = [Compose((*ts, *default_post)) for ts in val_transforms]

    return training_transforms, val_transforms


class AugmentedCityscapesLikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        paths: DatasetPaths,
        batch_size: int,
        scale_input: int,
        mean=None,
        std=None,
        class_weights=None,
        meta_data_path: str = None,
        num_workers=1,
        dataset=Cityscapes,
        fast_dev_run=False,
    ):
        """Creates an Augmented Cityscape Like dataset, like with the meaning that the labels are the same as the
        Cityscapes-dataset."""
        super().__init__()
        self.batch_size = batch_size
        self.scale_input = scale_input

        self.dataset = dataset
        self.dataset_mean = mean
        self.dataset_std = std
        self.class_weights = class_weights
        if fast_dev_run:
            t = os.path.split(meta_data_path)
            self.meta_data_path = os.path.join(t[0], f"DEBUG_{t[1]}")
        else:
            self.meta_data_path = meta_data_path

        self.dataset_default = {"normalize": False, "fast_dev_run": fast_dev_run, "root": root, "paths": paths}

        self.data_loader_default = {
            "batch_size": batch_size,
            # "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": False,
        }

        self.train, self.test, self.val = None, None, None
        self.fast_dev_run = fast_dev_run

        self.ignore_index = None
        self.num_classes = None

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        if self.dataset_mean is None:
            if self.meta_data_path and os.path.exists(self.meta_data_path):
                with open(self.meta_data_path, "rb") as f:
                    loaded = pickle.load(f)
                    self.dataset_mean = loaded["mean"]
                    self.dataset_std = loaded["std"]
                    self.class_weights = loaded["class_weights"]
                    self.num_classes = loaded["num_classes"]
                    self.ignore_index = loaded["ignore_index"]
            else:
                data_statistics = DataStatistics(
                    loader=DataLoader(self.dataset(split="train", **self.dataset_default), **self.data_loader_default)
                )
                self.dataset_mean = data_statistics.mean
                self.dataset_std = data_statistics.std
                self.class_weights = data_statistics.class_weights
                self.num_classes = data_statistics.num_classes
                self.ignore_index = data_statistics.ignore_index

                if self.meta_data_path:
                    with open(self.meta_data_path, "wb") as f:
                        pickle.dump(
                            {
                                "mean": self.dataset_mean,
                                "std": self.dataset_std,
                                "class_weights": self.class_weights,
                                "num_classes": self.num_classes,
                                "ignore_index": self.ignore_index,
                            },
                            f,
                        )

    def setup(self, stage: Optional[str] = None):
        training_transforms, val_transforms = _create_transforms_(
            mean=self.dataset_mean, std=self.dataset_std, scale_in=self.scale_input
        )

        if stage == "fit" or stage is None:
            dataset_full = ConcatTupleDataset(
                [
                    self.dataset(split="train", transforms=transform, **self.dataset_default)
                    for transform in training_transforms
                ]
            )

            self.train, self.val = random_split(dataset_full, get_split_sizes(dataset_full))

        if stage == "test" or stage is None:
            self.test = self.dataset(split="test", transforms=val_transforms[0], **self.dataset_default)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train,
            collate_fn=DatasetCollate(),
            sampler=MultipleRandomSampler(self.train, num_times=len(self.train.dataset.datasets)),
            **self.data_loader_default,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val,
            collate_fn=DatasetCollate(),
            sampler=MultipleSequentialSampler(self.val, num_times=len(self.val.dataset.datasets)),
            **self.data_loader_default,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test, **self.data_loader_default)
