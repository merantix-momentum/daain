import logging
import os
import pickle
from pickle import PicklingError
from typing import Iterable

import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from mxlabs_ood_detection.backbones.esp_net.espnet import CustomESPNet
from mxlabs_ood_detection.config_schema import PATH_TO_CONFIGS
from mxlabs_ood_detection.model import classifiers as C
from mxlabs_ood_detection.model.normalising_flow.coupling_blocks.subnet_constructors import CouplingBlockType
from mxlabs_ood_detection.model.normalising_flow.lightning_module import OODDetectionFlow
from mxlabs_ood_detection.trainer.data import get_data, get_data_transform
from mxlabs_ood_detection.utils.array_utils import t2np
from mxlabs_ood_detection.utils.evaluation_utils import one_vs_one_scores_over_classifiers
from mxlabs_ood_detection.utils.pytorch_utils import enforce_reproducibility, to_device


def model_predict(
    model, single_loaders: Iterable[DataLoader], multi_loaders: Iterable[Iterable[DataLoader]], use_gpu=True
):
    def eval_single_loader(loader):
        # best to move and remove them from gpu
        return np.vstack([t2np(model(to_device(data, cuda=use_gpu))) for data in loader])

    def eval_multiple_loaders(loaders):
        return [(name, eval_single_loader(l)) for name, l in loaders]

    return (
        *[eval_single_loader(loader) for loader in single_loaders],
        *[eval_multiple_loaders(loaders) for loaders in multi_loaders],
    )


def setup_logger(logging_dir, experiment_id):
    return TensorBoardLogger(save_dir=logging_dir, name=experiment_id)


class LossRecorderCallback(Callback):
    def __init__(self, monitor="val_loss"):
        super(LossRecorderCallback, self).__init__()
        self.losses = []
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if self.monitor in trainer.callback_metrics:
            self.losses.append(trainer.callback_metrics[self.monitor].detach().cpu().numpy())


def setup_trainer(
    logging_dir, checkpoints_path, experiment_id, use_gpu, trainer_args, enable_early_stopping=True,
):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, dirpath=checkpoints_path, filename=experiment_id,
    )

    loss_recorder = LossRecorderCallback()

    callbacks = [checkpoint_callback, loss_recorder]

    if enable_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min"))

    return (
        Trainer(
            logger=setup_logger(logging_dir, experiment_id),
            gpus=(1 if use_gpu else 0),
            callbacks=callbacks,
            **dict(filter(lambda x: x[0] not in "enable_early_stopping", trainer_args.items())),
        ),
        loss_recorder,
        f"{checkpoint_callback.filename}.ckpt",
    )


def _get_dataset_zarr_group(dataset_dict, activation_path_template):
    return zarr.group(
        store=zarr.DirectoryStore(
            activation_path_template.format(data_split=dataset_dict["split"], dataset=dataset_dict["dataset"])
        )
    )


logger = logging.getLogger(__name__)


@hydra.main(config_path=PATH_TO_CONFIGS, config_name="detection_training")
# def main(cfg: DetectionTrainingSchema):
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))

    # for the poisson disc sampling algorithm
    enforce_reproducibility(cfg.random_seed)
    root_dir = cfg.paths["derived"]
    masks_zarr = zarr.group(store=zarr.DirectoryStore(os.path.join(cfg.paths["activations"], "masks.zarr")))
    activation_path_template = cfg.paths["activation_path_template"]
    backbone_model = CustomESPNet(
        device=torch.device(cfg.device),
        p=cfg.backbone.model.p,
        q=cfg.backbone.model.q,
        num_classes=cfg.backbone.model.num_classes,
    ).load_pretrained_weights(cfg.paths.model_weights)

    plot_dir = cfg.paths["plots"]
    os.makedirs(plot_dir, exist_ok=True)

    datasets = [
        (ds["dataset"], ds, _get_dataset_zarr_group(ds, activation_path_template)) for ds in cfg.dataset_descriptions
    ]

    data_loader_args = cfg.trainer["data_loader"]
    trainer_args = cfg.trainer["trainer"]

    if cfg.trainer["trainer"]["fast_dev_run"]:
        trainer_args["max_epochs"] = 10

    model_args = {
        "coupling_block_type": getattr(CouplingBlockType, cfg.normalising_flow.coupling_block_type),
        "num_coupling_blocks": cfg.normalising_flow.num_coupling_blocks,
    }

    experiment_id = f"{'DEBUG' if cfg.trainer.trainer.fast_dev_run else ''}" "-".join(
        (
            cfg.normalising_flow.coupling_block_type,
            f"flow_steps_{cfg.normalising_flow.num_coupling_blocks}",
            cfg.mask.mask_name,
            cfg.mask.mask_type,
        )
    )

    logging.info(f"running model run with id: {experiment_id}")
    transform = get_data_transform(add_keys=model_args["coupling_block_type"] == CouplingBlockType.GLOW_POS_ATTENTION)

    (
        train_loader,
        val_loader,
        test_loader,
        perturbed_dataset_loaders,
        out_of_distribution_loaders,
        input_shape,
    ) = get_data(
        datasets,
        dataset_assignments=cfg.dataset_assignments,
        mask=cfg.mask.mask_name,
        submask=cfg.mask.mask_type,
        transform=transform,
        data_loader_args=data_loader_args,
        fast_dev_run=cfg.trainer.trainer.fast_dev_run,
    )

    #
    # END - Experiment Configurations
    # Everything below here should be fixed (and defined above)
    #

    model = OODDetectionFlow(input_shape=input_shape, **model_args)

    checkpoints_path = cfg.paths["model_checkpoints_template"].format(experiment_id=experiment_id)
    os.makedirs(checkpoints_path, exist_ok=True)
    trainer, loss_recorder, model_ckpt_filename = setup_trainer(
        logging_dir=cfg.paths.training_logs,
        checkpoints_path=checkpoints_path,
        experiment_id=experiment_id,
        trainer_args=trainer_args,
        use_gpu=cfg.device == "cuda",
    )
    if not cfg.trainer.force and cfg.trainer.lazy_training and os.path.exists(os.path.join(checkpoints_path,
                                                                                 model_ckpt_filename)):
        logging.info("loading model")
        model.load_checkpoint(os.path.join(checkpoints_path, model_ckpt_filename))
    else:
        logging.info("fitting model")
        trainer.fit(model, train_loader, val_loader)

    #
    # evaluation of model
    #
    logging.info("generating output")
    model.eval(cuda=cfg.device == "cuda")
    out_train, out_val, out_test, out_perturbed, out_ood = model_predict(
        model,
        (train_loader, val_loader, test_loader),
        (perturbed_dataset_loaders, out_of_distribution_loaders),
        use_gpu=cfg.device == "cuda",
    )

    # Pattern matching would be nice...

    logging.info("training classifier and evaluating model")
    classifier = getattr(C, cfg.classifier.type)(out_train, **cfg.classifier.kwargs)

    try:
        with open(f"{checkpoints_path}/{classifier}.pkl", "wb") as f:
            pickle.dump(classifier, f)
    except PicklingError as e:
        print(f"pickling error: {e}")

    run_result = one_vs_one_scores_over_classifiers([(str(classifier), classifier),], out_test, out_ood, out_perturbed)
    run_result["model_id"] = experiment_id
    os.makedirs(cfg.paths.run_results, exist_ok=True)

    try:
        with open(f"{cfg.paths.run_results}/{experiment_id}.pkl", "wb") as f:
            pickle.dump(run_result, f)
    except PicklingError as e:
        print(f"pickling error: {e}")


if __name__ == "__main__":
    main()
