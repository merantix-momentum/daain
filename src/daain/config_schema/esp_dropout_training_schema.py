from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import MISSING

from daain.config_schema.backbones.backbone import BackboneConfigSchema
from daain.config_schema.backbones.esp_net import ESPNetConfigSchema
from daain.config_schema.datasets.dataset import DatasetPaths


@dataclass
class ESPDropoutTrainingSchema:
    paths: Dict[str, Any] = MISSING
    backbone: BackboneConfigSchema = ESPNetConfigSchema
    dataset_paths: DatasetPaths = MISSING