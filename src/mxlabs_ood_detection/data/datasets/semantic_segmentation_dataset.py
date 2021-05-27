import abc
import logging
import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mxlabs_ood_detection.config_schema.datasets.dataset import DatasetPaths
from mxlabs_ood_detection.constants import IMAGENET_MEAN, IMAGENET_STD
from mxlabs_ood_detection.utils.pil_utils import load_image


def _identity_transform_(x):
    return x


class SemanticSegmentationDataset(Dataset):
    """Special abstract Dataset where images and targets / masks are stored in separate dir with same names.
    Yields the tuple of (image, target).

    # TODO  complete this
    This is very close to the MaskedDatset
    """

    def __init__(
        self,
        root: str,
        paths: DatasetPaths,
        normalize: bool,
        transforms: Callable = None,
        use_cuda: bool = False,
        split: str = "train",
        fast_dev_run: bool = False,
    ):
        """
        Args:
            root:
            paths:
                Dict-like object that has the full paths for the splits
            normalize (Optional):
                whether to normalize input images using IMAGENET_MEAN & IMAGENET_STD
            transforms (Optional):
                A function/transforms that takes input sample and its target as entry and returns a transformed tuple.
            use_cuda:
            split:
                which split of the data to use, can be 'train' 'eval' 'test' 'inference'.
                Inference hints at data without ground-truth.
            fast_dev_run:
                similar to `fast_dev_run` from PyTorchLightning this uses only a subset of the data to allow faster
                debugging.
        """
        self.split = split
        self.root = root
        self.image_dir = paths.images[split]
        self.mask_dir = paths.targets[split]

        self.use_cuda = use_cuda

        self.paths = self._load_paths()

        self.transforms = transforms
        self.normalize = normalize
        self.normalize_func = (
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else _identity_transform_
        )

        if fast_dev_run:
            logging.warning("using only small test sample of dataset")
            self.paths = self.paths[:10]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Union[Tuple, Dict]:
        """The subscript `_u` denotes that our dataloader deals with unperturbed image only"""
        image, target = load_image(self.paths[idx][0]), load_image(self.paths[idx][1])

        if image.mode == "RGBA":
            # we want 3 channel tensors
            image = image.convert("RGB")

        image, target = self.transforms(image, target)

        image = self.normalize_func(image)
        target = self._mask_postprocess(target)

        # TODO move this into transforms
        if self.use_cuda:
            image, target = image.cuda(), target.cuda()

        return image, target

    @property
    def num_classes(self) -> int:
        """
        Generic algorithm for any dataset.
        Child datasets can also prefer to have a hardcoded class variable instead.
        Preferred to use `num_classes` as class variable.
        """
        return max(self._colormap.keys()) + 1

    @abc.abstractmethod
    def _load_paths(self) -> List[Tuple[str, str]]:
        """Method to load correct `image_paths` and `mask_paths`"""
        pass

    @abc.abstractmethod
    def _mask_postprocess(self, mask: Image) -> torch.Tensor:
        """Usually used to map the train ids method to process mask files before returning to dataloader"""
        pass

    @abc.abstractmethod
    def _colormap(self) -> Dict:
        """Maps label to color, for visualization"""
        pass

    @abc.abstractmethod
    def _labelmap(self) -> Dict:
        """Maps class label to class name"""
        pass

    def _get_names(self, files):
        """Method to extract common substrings from names of images and masks such that they can be associated."""
        names = np.asarray([os.path.splitext(os.path.basename(u))[0] for u in files])
        names.sort()
        return names
