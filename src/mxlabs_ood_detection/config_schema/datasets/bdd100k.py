from dataclasses import dataclass

from mxlabs_ood_detection.config_schema.datasets.dataset import DatasetConfigSchema

#_DEFAULT_PATHS_ = {
#         "IMAGE_DIR": {"train": "seg/images/train", "test": "seg/images/test", "val": "seg/images/val"},
#         "MASK_DIR": {"train": "seg/labels/train", "val": "seg/labels/val"},
#}

@dataclass
class BDD100KConfigSchema(DatasetConfigSchema):
    # TODO fix this
    name = "bdd100k"
    transformations = [("center_crop", 512), ("resize", 512)]