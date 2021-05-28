from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from omegaconf import MISSING

from daain.config_schema.backbones.backbone import BackboneConfigSchema
from daain.config_schema.backbones.esp_net import ESPNetConfigSchema
from daain.model.normalising_flow.coupling_blocks.subnet_constructors import CouplingBlockType


@dataclass
class Optimizer:
    name: str
    kwargs: Dict[str, Any]


_ADAM_DEFAULT_KWARGS_ = {"learning_rate": 2e-4, "betas": (0.8, 0.8), "eps": 1e-04, "weight_decay": 1e-5}


@dataclass
class AdamOptimizer(Optimizer):
    name: str = "Adam"
    kwargs: Dict[str, Any] = field(default_factory=lambda: _ADAM_DEFAULT_KWARGS_)


@dataclass
class DetectionTrainer:
    accumulate_grad_batches: int = 4
    batch_size: int = 32
    enable_early_stopping: bool = True
    fast_dev_run: bool = False
    max_epochs: int = 300
    num_workers: int = 4
    random_seed: int = 42
    use_cuda: bool = True
    # determines if the weights from the non-dropout model should be used
    pretrained: bool = True
    optimizer: Optimizer = AdamOptimizer


class SamplingMaskType(Enum):
    FULL_DIST_128 = "full_dist_128"
    FULL_DIST_64 = "full_dist_64"
    POST_CONV_64 = "post_conv_64"
    PRE_BATCH_NORM_64 = "pre_batch_norm_64"


@dataclass
class NormalisingFlow:
    coupling_block_type: CouplingBlockType = CouplingBlockType.GLOW_LINEAR
    num_blocks: int = 8


class ClassifierType(Enum):
    HISTOGRAM_BASED = "histogram_based"
    MEAN_BASED = "mean"


_HISTOGRAM_BASED_DEFAULT_KWARGS_ = {
    "n_bins": 40,
    "contamination": 0.001,
    "aggregation_method": "bmean_squared_over_voxels",
}


@dataclass
class Classifier:
    type: ClassifierType = ClassifierType.HISTOGRAM_BASED
    kwargs: Dict[str, Any] = field(default_factory=lambda: _HISTOGRAM_BASED_DEFAULT_KWARGS_)


@dataclass
class DetectionModelSchema:
    normalising_flow: NormalisingFlow = NormalisingFlow
    classifier: Classifier = Classifier


@dataclass
class DetectionTrainingSchema:
    paths: Dict[str, Any] = MISSING
    backbone: BackboneConfigSchema = ESPNetConfigSchema
    trainer: DetectionTrainer = DetectionTrainer
    mask: SamplingMaskType = SamplingMaskType.FULL_DIST_64
    model: DetectionModelSchema = DetectionModelSchema
