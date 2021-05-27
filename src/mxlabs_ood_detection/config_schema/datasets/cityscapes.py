from dataclasses import dataclass, field
from typing import List

from mxlabs_ood_detection.config_schema.datasets.dataset import DatasetConfigSchema, DatasetPaths, Transformation

_DEFAULT_TRANSFORMATIONS_ = [{"name": "Resize", "kwargs": {"size": 512}}]

_DEFAULT_PATHS_ = {
    "IMAGE_DIR": {
        "train": "leftImg8bit_trainvaltest/leftImg8bit/train",
        "val": "leftImg8bit_trainvaltest/leftImg8bit/val",
        "test": "leftImg8bit_trainvaltest/leftImg8bit/test",
    },
    "MASK_DIR": {
        "train": "gtFine_trainvaltest/gtFine/train",
        "val": "gtFine_trainvaltest/gtFine/val",
        "test": "gtFine_trainvaltest/gtFine/test",
    },
}


@dataclass
class CityscapesConfigSchema(DatasetConfigSchema):
    name: str = "Cityscapes"
    paths: DatasetPaths = field(default_factory=lambda: _DEFAULT_PATHS_)
    transformations: List[Transformation] = field(default_factory=lambda: _DEFAULT_TRANSFORMATIONS_)
