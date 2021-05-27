import os
from typing import Callable, List, Tuple

import torch
import torchvision.transforms as T

from mxlabs_ood_detection.config_schema.datasets.dataset import DatasetPaths
from mxlabs_ood_detection.data.datasets.cityscapes_dataset import Cityscapes
from mxlabs_ood_detection.data.datasets.labels import bdd100k

# more of an example to make it the same size as the Cityscapes images, assuming the resizing
_MAKE_CITYSCAPE_COMPATIBLE_ = T.Compose([T.CenterCrop((640, 1280)), T.Resize(512), T.ToTensor()])


class BDD100k(Cityscapes):
    def __init__(
        self,
        root: str,
        paths: DatasetPaths,
        split: str = "train",
        normalize: bool = True,
        transforms: Callable = None,
        use_train_id: bool = True,
        use_cuda: bool = True,
        fast_dev_run: bool = True,
    ):
        super(BDD100k, self).__init__(
            root=root,
            paths=paths,
            split=split,
            normalize=normalize,
            transforms=transforms,
            use_train_id=use_train_id,
            use_cuda=use_cuda,
            fast_dev_run=fast_dev_run,
        )

    @staticmethod
    def _translate_to_train_id(mask: torch.Tensor) -> torch.tensor:
        """maps 35 classes to 19 + 1 (void class)"""
        _ones = torch.ones_like(mask)
        for id, trainId in bdd100k.id2trainId.items():
            mask = torch.where(mask == id, trainId * _ones, mask)
        mask = torch.where((mask == -1), _ones * 255, mask)
        return mask

    def _load_paths(self) -> List[Tuple[str, str]]:
        def _get_paths_(x):
            return [os.path.join(self.root, x, i) for i in sorted(os.listdir(os.path.join(self.root, x)))]

        self.image_paths = _get_paths_(self.image_dir)
        self.mask_paths = _get_paths_(self.mask_dir)

        return list(zip(sorted(self.image_paths), sorted(self.mask_paths)))
