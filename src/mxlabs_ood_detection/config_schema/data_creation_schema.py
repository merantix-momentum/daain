from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import MISSING

from mxlabs_ood_detection.config_schema.activation_spaces.activation_space import ActivationSpace
from mxlabs_ood_detection.config_schema.activation_spaces.esp_net_256_512 import ESPNet_256_512
from mxlabs_ood_detection.config_schema.backbones.backbone import BackboneConfigSchema
from mxlabs_ood_detection.config_schema.backbones.esp_net import ESPNetConfigSchema
from mxlabs_ood_detection.config_schema.detection_schema import SamplingMaskType

_SAMPLE_MASKS_ = [
    SamplingMaskType.FULL_DIST_64,
    SamplingMaskType.FULL_DIST_128,
    SamplingMaskType.POST_CONV_64,
    SamplingMaskType.PRE_BATCH_NORM_64,
]


@dataclass
class DataCreationSchema:
    paths: Dict[str, str] = MISSING
    backbone: BackboneConfigSchema = ESPNetConfigSchema
    sampling_masks: List[SamplingMaskType] = field(default_factory=lambda: _SAMPLE_MASKS_)
    activation_spaces: ActivationSpace = ESPNet_256_512
    force: bool = False
    random_seed: int = 42
