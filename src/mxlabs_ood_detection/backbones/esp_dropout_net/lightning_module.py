from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch

from mxlabs_ood_detection.backbones.esp_dropout_net.esp_dropout_net import ESPDropoutNet, ESPDropoutNetEncoder
from mxlabs_ood_detection.backbones.esp_dropout_net.trainer.criteria import CrossEntropyLoss
from mxlabs_ood_detection.backbones.esp_dropout_net.trainer.iou_eval import IoUMetric
from mxlabs_ood_detection.backbones.esp_net.espnet import CustomESPNet


class ESPDropoutNetLightning(pl.LightningModule):
    def __init__(
        self,
        encoder: ESPDropoutNetEncoder = None,
        num_classes: int = 20,
        learning_rate: float = 5e-4,
        step_loss: int = 100,
        p: int = 2,
        q: int = 8,
        class_weights: np.ndarray = None,
        ignore_index: int = None,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        if encoder:
            self.model = ESPDropoutNet(num_classes, p, q, encoder=encoder, dropout_rate=dropout_rate)
        else:
            self.model = ESPDropoutNetEncoder(num_classes, p=p, q=q, dropout_rate=dropout_rate)

        self.criteria = CrossEntropyLoss(class_weights, ignore_index)
        self.learning_rate = learning_rate
        self.step_loss = step_loss
        self.mIoUMetric = IoUMetric(num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, prefix):
        self.mIoUMetric.reset()
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)

        overall_acc, per_class_acc, per_class_iu, mIOU = self.mIoUMetric(y_hat.argmax(1), y)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_mean_iou", mIOU)

        return loss

    def _shared_epoch_end(self, outs, prefix):
        overall_acc, per_class_acc, per_class_iu, mIOU = self.mIoUMetric.compute()
        self.log(f"{prefix}_mean_iou_epoch", mIOU)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "trainer")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._shared_epoch_end(outputs, "trainer")

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._shared_epoch_end(outputs, "test")

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._shared_epoch_end(outputs, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=5e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_loss, gamma=0.5)

        return [optimizer], [scheduler]

    def parameter_count(self, only_trainable=False):
        return sum(p.numel() for p in self.model.parameters() if (p.requires_grad if only_trainable else True))

    @classmethod
    def load_from_base_model(
        cls,
        root_path: str,
        num_classes: int = 20,
        learning_rate: float = 5e-4,
        step_loss: int = 100,
        p: int = 2,
        q: int = 8,
        class_weights: np.ndarray = None,
        ignore_index: int = None,
        dropout_rate: float = 0.5,
    ):
        """Loads the weights from the ESPNet model without Dropout"""
        model_lightning = cls(
            p=p,
            q=q,
            num_classes=num_classes,
            class_weights=class_weights,
            learning_rate=learning_rate,
            step_loss=step_loss,
            ignore_index=ignore_index,
            encoder=ESPDropoutNetEncoder(p=p, q=q, classes=num_classes),
            dropout_rate=dropout_rate,
        )

        espnet_full_model_weights = torch.load(CustomESPNet.get_weight_path(root=root_path, p=p, q=q))

        esp_dropout_net_dict = model_lightning.model.state_dict()
        for k, v in espnet_full_model_weights.items():
            esp_dropout_net_dict[cls._remap_non_dropout_key(k)] = v

        model_lightning.model.load_state_dict(esp_dropout_net_dict)

        return model_lightning

    @staticmethod
    def _remap_non_dropout_key(key):
        if key.startswith("encoder.level2.") or key.startswith("encoder.level3."):
            t = key.split(".")
            return ".".join([*t[:2], str(int(t[2]) * 2), *t[3:]])
        else:
            return key
