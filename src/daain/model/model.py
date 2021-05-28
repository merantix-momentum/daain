import logging
import os
from collections import OrderedDict
from typing import Tuple, Union

import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig

from daain.backbones.esp_net.espnet import CustomESPNet
from daain.config_schema import PATH_TO_CONFIGS
from daain.data.activations_dataset import create_post_hook_for_layer, retrieve_modules_from_esp_net
from daain.data.activations_dataset import upsample_activations
from daain.model.classifiers import Classifier
from daain.model.normalising_flow.lightning_module import OODDetectionFlow
from daain.utils.array_utils import t2np


class DetectionModel:
    def __init__(
        self,
        backbone: CustomESPNet,
        sampling_mask: np.ndarray,
        normalising_flow: OODDetectionFlow,
        classifier: Classifier,
    ):
        """
        In general there are a lot of assumptions that have to be true:
            - Data dimensions are fixed (no change, no rotation, ...)
            - Input data are images

        Args:
            backbone: It is not strictly necessary for it to be a CustomESPNet, but the module extraction would have
                      to be adapted if you change the backbone
            sampling_mask:
            normalising_flow:
            classifier:
        """
        self.backbone = backbone
        self.normalising_flow = normalising_flow
        self.classifier = classifier  # thresholds are inside here (if any)
        self.sampling_mask = sampling_mask
        # sink for the activations, the order is not strictly necessary given that
        self.activations = OrderedDict()
        # we sample from all activations but makes the interpretation about them
        # easier

        self._register_activation_sampling()

    def _register_activation_sampling(self):
        """Use this to overwrite sampling method if you use a different backbone"""
        for n, m in retrieve_modules_from_esp_net(self.backbone):
            m.register_forward_hook(create_post_hook_for_layer(self.activations, n))

    def predict_proba(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: the image data as a torch.Tensor (can be batched)

        Returns:
            A tuple of the segmentation mask and the score for the image, note that the score is not necessarily
            normalised, this depends on the classifier you use. HBOS - the default - is normalised.
        """
        self.normalising_flow.eval()
        backbone_output, activations = self._get_backbone_output_and_activations(data)
        return backbone_output, self.classifier(t2np(self.normalising_flow(activations)))

    def predict(self, data: Union[np.ndarray, torch.Tensor], threshold: float) -> Tuple[torch.Tensor, bool]:
        backbone_output, anomaly_score = self.predict_proba(data)
        return backbone_output, anomaly_score > threshold

    def _get_backbone_output_and_activations(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        # strictly not necessary since you overwrite the values anyway.
        self.activations.clear()

        # this triggers the recording as well
        backbone_output = self.backbone(data)
        upsampled_activations = upsample_activations(self.activations.values(), img_size=data.shape[-2:])

        if str(self.sampling_mask.dtype).startswith("int"):
            # sampling mask is a list of coordinates
            selection_shape = self.sampling_mask.shape
            _sel = self.sampling_mask.reshape(-1, 3)
            selected_upsampled_activations = upsampled_activations[:, _sel[:, 0], _sel[:, 1], _sel[:, 2]].reshape(
                -1, *selection_shape[:-1]
            )
        else:
            # sampling mask is a boolean mask with the same shape as the activation space
            selected_upsampled_activations = upsampled_activations[:, self.sampling_mask]

        return backbone_output, selected_upsampled_activations

    @staticmethod
    def load_from_params(
        model_checkpoints_template,
        backbone_weight_path,
        mask_path,
        backbone,
        normalising_flow,
        classifier,
        mask,
        device="cuda",
    ):
        import pickle
        from daain.model import classifiers as C

        backbone_model = CustomESPNet(
            device=torch.device(device),
            p=backbone["model"]["p"],
            q=backbone["model"]["q"],
            num_classes=backbone["model"]["num_classes"],
        ).load_pretrained_weights(backbone_weight_path)

        sampling_mask = np.array(
            zarr.group(store=zarr.DirectoryStore(mask_path))[f"{mask['mask_name']}_{mask['mask_type']}"]
        )

        # TODO make this work with coordinate mask as well
        # this is a bit convoluted... the experiment id can be hard-coded or inferred from the parameters
        # it's just there to provide the path
        model = OODDetectionFlow(input_shape=(np.sum(sampling_mask),), **normalising_flow)
        experiment_id = "-".join((str(model), mask["mask_name"], mask["mask_type"]))
        checkpoints_path = model_checkpoints_template.format(experiment_id=experiment_id)
        model.load_checkpoint(os.path.join(checkpoints_path, f"{checkpoints_path}/{experiment_id}.ckpt"))

        classifier_mdl = getattr(C, classifier["type"])(**classifier["kwargs"])

        with open(f"{checkpoints_path}/{classifier_mdl}.pkl", "rb") as f:
            classifier_mdl = pickle.load(f)

        mdl = DetectionModel(
            backbone=backbone_model, normalising_flow=model, classifier=classifier_mdl, sampling_mask=sampling_mask
        )

        return mdl


@hydra.main(config_path=PATH_TO_CONFIGS, config_name="detection_inference")
def _main_(cfg: DictConfig):
    from daain.scripts.data_creation import _get_loader

    mdl = DetectionModel.load_from_params(
        model_checkpoints_template=cfg.paths.model_checkpoints_template,
        backbone_weight_path=cfg.paths.model_weights,
        mask_path=f"{cfg.paths['activations']}/masks.zarr",
        backbone=cfg.backbone,
        normalising_flow=cfg.normalising_flow,
        classifier=cfg.classifier,
        mask=cfg.mask,
    )

    logging.info("loaded mdl")

    sample_loader = _get_loader(dataset_dict=cfg.data, attacks_path_template=cfg.paths.attacks_path_template)

    # just load one example image
    sample_input = next(iter(sample_loader))
    for a in [*cfg.data.attacks, "unperturbed"]:
        if a == "unperturbed":
            input_image = sample_input["normal_image"]
        else:
            input_image = sample_input["perturbed"][a]["perturbed_image"]

        sample_segmentation_mask, anomaly_score = mdl.predict_proba(input_image)
        logging.info(f"sample output: {a} anomly_score: {anomaly_score}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    _main_()
