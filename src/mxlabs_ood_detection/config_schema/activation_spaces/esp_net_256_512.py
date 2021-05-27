from dataclasses import dataclass, field
from typing import Dict, List

from mxlabs_ood_detection.config_schema.activation_spaces.activation_space import ActivationSpace

_PRE_BATCH_NORM_REJECTIONS_ = [
    [0, 3],
    [19, 44],
    [63, 206],
    [270, 334],
    [398, 586],
    [650, 714],
    [845, 1229],
    [1357, 1485],
    [1613, 1994],
    [2122, 2503],
    [2631, 3012],
    [3140, 3521],
    [3649, 4030],
    [4158, 4539],
    [4667, 5048],
    [5176, 5304],
    [5560, 6243],
    [6283, 6383],
    [6403, 6443],
    [6463, 6519],
]

_POST_CONV_REJECTIONS_ = [
    [16, 95],
    [171, 427],
    [503, 893],
    [1046, 1558],
    [1711, 1967],
    [2120, 2376],
    [2529, 2785],
    [2938, 3194],
    [3347, 3603],
    [3756, 4012],
    [4165, 4933],
    [4953, 4993],
    [5013, 5093],
    [5117, 5217],
]

_REJECTION_BOUNDARIES_ = {"pre_batch_norm": _PRE_BATCH_NORM_REJECTIONS_, "post_conv": _POST_CONV_REJECTIONS_}

_ACTIVATION_SPACES_ = {"pre": [6579, 256, 512], "post": [5297, 256, 512]}


@dataclass
class ESPNet_256_512(ActivationSpace):
    rejection_boundaries: Dict[str, List[List[int]]] = field(default_factory=lambda: _REJECTION_BOUNDARIES_)
    activation_space: Dict[str, List[int]] = field(default_factory=lambda: _ACTIVATION_SPACES_)
