import logging
import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from daain.backbones.esp_dropout_net.lightning_module import ESPDropoutNetLightning
from daain.backbones.esp_dropout_net.trainer.data import AugmentedCityscapesLikeDataModule
from daain.config_schema import PATH_TO_CONFIGS
from daain.config_schema.backbones.esp_dropout import ESPDropoutModelTrainingConfigSchema, get_id
from daain.config_schema.datasets.dataset import DatasetPaths
from daain.config_schema.esp_dropout_training_schema import ESPDropoutTrainingSchema
from daain.data.datasets.bdd100k_dataset import BDD100k
from daain.data.datasets.cityscapes_dataset import Cityscapes
from daain.utils.pytorch_utils import enforce_reproducibility


def setup_logger(logging_dir, experiment_id):
    return TensorBoardLogger(save_dir=logging_dir, name=experiment_id)


def setup_trainer(
    logging_dir, checkpoints_path, experiment_id, use_cuda=True, enable_early_stopping=False, trainer_args=None
):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, dirpath=checkpoints_path, filename=f"{experiment_id}"
    )

    callbacks = [checkpoint_callback]

    if enable_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    if trainer_args is None:
        trainer_args = {}

    return Trainer(
        logger=setup_logger(logging_dir, experiment_id),
        gpus=(1 if use_cuda else 0),
        callbacks=callbacks,
        **trainer_args,
    )


logger = logging.getLogger(__name__)
cs = ConfigStore.instance()
cs.store(group="backbone", name="esp_dropout", node=ESPDropoutModelTrainingConfigSchema)
cs.store(group="dataset_paths", name="cityscapes", node=DatasetPaths)
cs.store(group="dataset_paths", name="bdd100k", node=DatasetPaths)
cs.store(name="esp_dropout_training", node=ESPDropoutTrainingSchema)


@hydra.main(config_path=PATH_TO_CONFIGS, config_name="esp_dropout_training")
def main(cfg: ESPDropoutTrainingSchema):
    # TODO remove this
    # print(OmegaConf.to_yaml(cfg))

    # TODO move this to default
    if cfg.backbone.trainer.num_workers > 0:
        torch.multiprocessing.set_start_method("spawn")

    enforce_reproducibility(cfg.backbone.trainer.random_seed)

    root_dir = cfg.paths["derived"]
    checkpoints_path = cfg.paths["model_checkpoints_template"].format(experiment_id="esp_dropout_net")
    logging_dir = cfg.paths["training_logs"]
    preprocess_path = os.path.join(root_dir, "preprocess_data/cityscapes.pkl")
    os.makedirs(os.path.dirname(preprocess_path), exist_ok=True)

    default_data_args = {
        "batch_size": cfg.backbone.trainer["batch_size"],
        "num_workers": cfg.backbone.trainer["num_workers"],
        "fast_dev_run": cfg.backbone.trainer["fast_dev_run"],
        "meta_data_path": preprocess_path,
        "dataset": Cityscapes if cfg.backbone.trainer["dataset"]["name"] == "cityscapes" else BDD100k,
        # TODO is there a better way to do this automatically given the config?
        "root": os.path.join(cfg.paths.datasets, cfg.backbone.trainer["dataset"]["name"]),
        "paths": cfg.dataset_paths,
    }

    default_trainer_args = {
        "logging_dir": logging_dir,
        "checkpoints_path": checkpoints_path,
        "use_cuda": cfg.backbone.trainer["use_cuda"],
        "enable_early_stopping": cfg.backbone.trainer["enable_early_stopping"],
        "trainer_args": {
            "fast_dev_run": False,  # args.fast_dev_run,
            "max_epochs": cfg.backbone.trainer["max_epochs"],
            "accumulate_grad_batches": cfg.backbone.trainer["accumulate_grad_batches"],
        },
    }

    #
    # Encoder part
    #

    logging.info("prepping data ...")
    data = AugmentedCityscapesLikeDataModule(**{**default_data_args, "scale_input": 8})

    default_model_args = {
        "num_classes": data.num_classes,
        "learning_rate": cfg.backbone.trainer["learning_rate"],
        "step_loss": cfg.backbone.trainer["step_loss"],
        "p": cfg.backbone.model["p"],
        "q": cfg.backbone.model["q"],
        "class_weights": data.class_weights,
        "ignore_index": data.ignore_index,
        "dropout_rate": cfg.backbone.model["dropout_rate"],
    }

    # either trainer the encoder, or jump directly to the full model
    if cfg.backbone.trainer["pretrained"]:
        # can't directly load it as some we introduced new layers
        model = ESPDropoutNetLightning.load_from_base_model(root_path=cfg.paths.model_weights, **default_model_args)
    else:
        logging.info("prepping data ... done")
        _exp_id = get_id(cfg.backbone, model_part_to_train="encoder")
        trainer = setup_trainer(**{**default_trainer_args, "experiment_id": _exp_id})

        model = ESPDropoutNetLightning(**default_model_args)
        logging.info(f"training encoder {_exp_id}...")
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)
        logging.info(f"training encoder {_exp_id} ... done")

        model = ESPDropoutNetLightning(**{**default_model_args, "encoder": model.model})

    #
    # Full model
    #

    data = AugmentedCityscapesLikeDataModule(**{**default_data_args, "scale_input": 1})

    logging.info("prepping data ... done")
    _exp_id = get_id(cfg.backbone)
    trainer = setup_trainer(**{**default_trainer_args, "experiment_id": _exp_id})

    logging.info(f"training full model {_exp_id} ...")
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
    logging.info(f"training full model {_exp_id} ... done")


if __name__ == "__main__":
    main()
