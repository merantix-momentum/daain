from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import MISSING

from mxlabs_ood_detection.config_schema.backbones.backbone import BackboneConfigSchema
from mxlabs_ood_detection.config_schema.backbones.esp_net import ESPNetConfigSchema
from mxlabs_ood_detection.config_schema.datasets.dataset import DatasetPaths


@dataclass
class ESPDropoutTrainingSchema:
    paths: Dict[str, Any] = MISSING
    backbone: BackboneConfigSchema = ESPNetConfigSchema
    dataset_paths: DatasetPaths = MISSING