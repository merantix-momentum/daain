from dataclasses import dataclass, field
from typing import Dict

from omegaconf import MISSING

from mxlabs_ood_detection.config_schema.backbones.backbone import BackboneConfigSchema


@dataclass
class ModelArgs:
    p: int = 2
    q: int = 8
    num_classes: int = 19


@dataclass
class ESPNetConfigSchema(BackboneConfigSchema):
    model: ModelArgs = MISSING
