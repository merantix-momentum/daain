import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from daain.config_schema.datasets.dataset import DatasetPaths
from daain.data.datasets.labels import cityscape
from daain.data.datasets.semantic_segmentation_dataset import SemanticSegmentationDataset


class Cityscapes(SemanticSegmentationDataset):
    """
    The Cityscapes dataset has two modes for using it. One uses 35 classes & another 19. Rarely are the models
    trained on 35 classes since the evaluation for cityscape considers the performance on 19 classes.
    """

    ignore_index = 255
    num_classes = 19  # 19 + 1 for rest

    def __init__(
            self,
            root: str,
            paths: DatasetPaths,
            split: str = "train",
            normalize: bool = True,
            transforms: Callable = None,
            use_train_id: bool = True,
            use_cuda: bool = False,
            fast_dev_run: bool = False,
    ):

        super().__init__(
            root=root,
            paths=paths,
            split=split,
            normalize=normalize,
            transforms=transforms,
            use_cuda=use_cuda,
            fast_dev_run=fast_dev_run,
        )

        void_classes = [label.id for label in cityscape.labels if label.category == "void"]
        self.max_void_label = max(void_classes)
        self.use_train_id = use_train_id

    def _load_paths(self) -> List[Tuple[str, str]]:
        def file_validator(u):
            return "._" not in u  # NOQA E731

        def mask_validator(u):
            return "_labelIds.png" in u and file_validator(u)  # NOQA E731

        def _get_paths(file_dir, validator):
            file_dir = os.path.join(self.root, file_dir)
            cities = [city for city in os.listdir(file_dir) if file_validator(city)]
            file_paths = list()
            for city in cities:
                localpath = os.path.join(file_dir, city)
                file_paths += [
                    os.path.join(localpath, filename) for filename in os.listdir(localpath) if validator(filename)
                ]
            return np.asarray(file_paths).flatten()

        self.image_paths = _get_paths(self.image_dir, file_validator)
        self.mask_paths = _get_paths(self.mask_dir, mask_validator)

        return list(zip(sorted(self.image_paths), sorted(self.mask_paths)))

    def _remove_void_labels(self, mask: torch.Tensor) -> torch.Tensor:
        """reduces all 'void' category labels to single class '0'"""
        mask = torch.where(mask <= self.max_void_label, torch.ones_like(mask) * self.max_void_label, mask)
        mask = mask - self.max_void_label * torch.ones_like(mask)
        return mask

    def _mask_postprocess(self, mask: Image) -> torch.Tensor:
        mask = torch.from_numpy(np.array(mask))
        if self.use_train_id:
            return self._translate_to_train_id(mask)
        else:
            return self._remove_void_labels(mask)

    @staticmethod
    def _translate_to_train_id(mask: torch.Tensor) -> torch.tensor:
        """maps 35 classes to 19 + 1 (void class)"""
        _ones = torch.ones_like(mask)
        for id, trainId in cityscape.id2trainId.items():
            mask = torch.where(mask == id, trainId * _ones, mask)
        mask = torch.where((mask == -1), _ones * 255, mask)
        return mask

    @property
    def _colormap(self):
        return {label.trainId: label.color for label in cityscape.labels}

    @property
    def _labelmap(self) -> Dict:
        return {label.trainId: label.name for label in cityscape.labels}

    def _get_names(self, files):
        names = [os.path.splitext(os.path.basename(u))[0] for u in files]
        names = np.asarray(["_".join(name.split("_")[:2]) for name in names])
        names.sort()
        return names
