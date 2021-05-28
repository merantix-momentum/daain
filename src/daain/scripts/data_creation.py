"""
The data creation script. Assumes a certain folder structure for the attacks (best viewed in the `paths` config (
`attacks_path_template`.
"""
import logging
import os
from collections import namedtuple

import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from daain.backbones.esp_net.espnet import CustomESPNet
from daain.config_schema import PATH_TO_CONFIGS
from daain.data import PerturbedDataset
from daain.data.activations_dataset import compute_activations, create_mask
from daain.utils.pytorch_utils import enforce_reproducibility
from daain.utils.zarr_utils import open_group


# def _get_transforms(dataset_dict):
#    return T.Compose([getattr(T, n)(**kwargs) for n, kwargs in dataset_dict["transforms"].items()])


def _get_loader(dataset_dict, attacks_path_template):
    """Source"""
    root_path_to_attacked_images = attacks_path_template.format(
        data_split=dataset_dict["split"], dataset=dataset_dict["dataset"]
    )

    data = PerturbedDataset(
        attacks=dataset_dict["attacks"], root_path=root_path_to_attacked_images, transforms=dataset_dict["transforms"]
    )
    return DataLoader(dataset=data, batch_size=1, shuffle=False)


def _get_activations_group(dataset_dict, activations_path_template, force=False):
    """Sink"""
    activations_path = activations_path_template.format(
        data_split=dataset_dict["split"], dataset=dataset_dict["dataset"]
    )
    return zarr.group(store=zarr.DirectoryStore(activations_path), overwrite=force)


def _gen_masks(
    mask_name, rejection_boundaries, activation_space_shape, min_distance, activations_group, masks_zarr, force=False
):
    group = open_group(activations_group, mask_name)

    group_raw = open_group(group, "raw")
    group_gridified = open_group(group, f"gridified")

    if force or f"{mask_name}_raw" not in masks_zarr:
        logging.info(f"using new mask for {mask_name}")
        t = create_mask(min_distance, rejection_boundaries, activation_space_shape)
        if rejection_boundaries is None:
            masks_zarr[f"{mask_name}_raw"] = t[0]
            masks_zarr[f"{mask_name}_gridified"] = t[1]
        else:
            masks_zarr[f"{mask_name}_raw"] = t
    else:
        logging.info(f"reusing existing mask {mask_name}")

    if rejection_boundaries is None:
        return zip(
            [np.array(masks_zarr[f"{mask_name}_{s}"]) for s in ["raw", "gridified"]], [group_raw, group_gridified]
        )
    else:
        return zip([np.array(masks_zarr[f"{mask_name}_{s}"]) for s in ["raw",]], [group_raw,])


def _gen_all_sampling_masks(activations_group, masks_zarr, mask_descriptions, force=False):
    """Used to combine all masks such that we load the image and thus activations only once. Then sample from all
    recorded activations in one go."""
    for mask in mask_descriptions:
        for selection, zarr_group in _gen_masks(
            mask_name=mask.mask_name,
            rejection_boundaries=mask.rejection_boundaries,
            activation_space_shape=mask.activation_space_shape,
            min_distance=mask.min_distance,
            activations_group=activations_group,
            masks_zarr=masks_zarr,
            force=force,
        ):
            yield selection, zarr_group, mask.recording_type


def generate_activations(
    model,
    dataset_descriptions,
    masks_zarr,
    mask_descriptions,
    attacks_path_template,
    activations_path_template,
    force=False,
    fast_dev_run=False,
):
    for dataset_dict in dataset_descriptions:
        loader = _get_loader(dataset_dict, attacks_path_template=attacks_path_template)
        activations_group = _get_activations_group(
            dataset_dict, activations_path_template=activations_path_template, force=force
        )

        logging.info(f"computing activations for {dataset_dict['split']}, {dataset_dict['dataset']}")
        selections = list(
            _gen_all_sampling_masks(
                activations_group=activations_group, masks_zarr=masks_zarr, mask_descriptions=mask_descriptions
            )
        )
        compute_activations(
            model,
            loader,
            attacks=dataset_dict["attacks"],
            selections=selections,
            force=force,
            n_data_points=10 if fast_dev_run else -1,
        )


logger = logging.getLogger(__name__)


# TODO this is causing some issues with typing a list of lists
# cs = ConfigStore.instance()
#
# cs.store(group="backbone", name="esp_dropout", node=ESPDropoutModelTrainingConfigSchema)
# cs.store(group="dataset_paths", name="cityscapes", node=DatasetPaths)
# cs.store(group="dataset_paths", name="bdd100k", node=DatasetPaths)
# cs.store(group="activation_spaces", name="esp_net_256_512", node=ActivationSpace)
# cs.store(name="data_creation", node=DataCreationSchema)


@hydra.main(config_path=PATH_TO_CONFIGS, config_name="data_creation")
# def main(cfg: DataCreationSchema):
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))

    # for the poisson disc sampling algorithm
    enforce_reproducibility(cfg.random_seed)
    root_dir = cfg.paths["derived"]
    masks_zarr = zarr.group(store=zarr.DirectoryStore(os.path.join(cfg.paths["activations"], "masks.zarr")))
    activations_path_template = os.path.join(root_dir, cfg.paths["activation_path_template"])
    attacks_path_template = os.path.join(root_dir, cfg.paths["attacks_path_template"])

    MaskDescription = namedtuple(
        "MaskDescription",
        ["mask_name", "min_distance", "recording_type", "rejection_boundaries", "activation_space_shape"],
    )
    masks = [
        MaskDescription(
            **{
                **kwargs,
                "activation_space_shape": cfg.activation_spaces.activation_space_shapes[kwargs["recording_type"]],
            }
        )
        for kwargs in cfg.masks
    ]

    backbone_model = CustomESPNet(
        device=torch.device(cfg.device), p=cfg.backbone.model.p, q=cfg.backbone.model.q, num_classes=cfg.num_classes
    ).load_pretrained_weights(cfg.paths.model_weights)

    generate_activations(
        backbone_model,
        masks_zarr=masks_zarr,
        dataset_descriptions=cfg.dataset_descriptions,
        attacks_path_template=attacks_path_template,
        activations_path_template=activations_path_template,
        mask_descriptions=masks,
    )


if __name__ == "__main__":
    main()
