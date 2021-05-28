from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING

from daain.config_schema.backbones.esp_net import ESPNetConfigSchema, ModelArgs
from daain.config_schema.datasets import CityscapesConfigSchema, DatasetConfigSchema


class DropoutPlacement(Enum):
    REGULAR = 1
    CUSTOM = 2


class ModelPart(Enum):
    FULL = 1
    ENCODER = 2


@dataclass
class ESPDropoutTrainerConfigSchema:
    accumulate_grad_batches: int = 4
    batch_size: int = 6
    dataset: DatasetConfigSchema = CityscapesConfigSchema
    enable_early_stopping: bool = True
    fast_dev_run: bool = False
    learning_rate: float = 5e-4
    max_epochs: int = 300
    num_workers: int = 4
    random_seed: int = 42
    step_loss: int = 100  # when the loss should be adapted
    use_cuda: bool = True
    # determines if the weights from the non-dropout model should be used
    pretrained: bool = True
    model_part: ModelPart = ModelPart.FULL


@dataclass
class ESPDropoutModelArgs(ModelArgs):
    dropout_rate: float = 0.2
    dropout_placements: DropoutPlacement = DropoutPlacement.REGULAR


@dataclass
class ESPDropoutModelTrainingConfigSchema(ESPNetConfigSchema):
    model: ESPDropoutModelArgs = MISSING
    trainer: ESPDropoutTrainerConfigSchema = MISSING


def get_id(cfg: ESPDropoutModelTrainingConfigSchema, model_part_to_train: str = "full") -> str:
    return "_".join(
        s
        for s in [
            ("DEBUG" if cfg.trainer.fast_dev_run else ""),
            ("PRETRAINED" if cfg.trainer.pretrained else ""),
            model_part_to_train,
            f"dropout-{cfg.model.dropout_rate:0.1f}",
            f"dataset-{cfg.trainer.dataset.name}",
        ]
        if s
    )
