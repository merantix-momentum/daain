from typing import Any, Optional

import pytorch_lightning as pl
import torch

from daain.utils.pytorch_utils import get_device


class IoUMetric(pl.metrics.Metric):
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
    def __init__(
        self,
        num_classes: int,
        use_cuda: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(compute_on_step, dist_sync_on_step, process_group)

        self.num_classes = num_classes

        self.add_state("overall_acc", torch.tensor(0.0))
        self.add_state("overall_acc", torch.tensor(0.0))
        self.add_state("per_class_acc", torch.zeros(self.num_classes, dtype=torch.float32))
        self.add_state("per_class_iu", torch.zeros(self.num_classes, dtype=torch.float32))
        self.add_state("mIOU", torch.tensor(0.0))
        self.add_state("total", torch.tensor(0))

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

    def fast_hist(self, true, pred):
        # ignoring the labels which are not wanted, it's assumed that they are below 0 or above the number of classes
        # to detect
        mask = (true >= 0) & (true < self.num_classes)
        return torch.bincount(
            self.num_classes * torch.as_tensor(pred[mask], dtype=torch.int, device=get_device(true)) + true[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        predict = torch.flatten(preds)
        target = torch.flatten(targets)

        epsilon = 0.00000001
        hist = self.fast_hist(true=target, pred=predict)
        overall_acc = torch.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = torch.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + epsilon)
        mIou = per_class_iu[~torch.isnan(per_class_iu)].mean()

        self.overall_acc += overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.total += targets.numel()

    def compute(self):
        overall_acc = self.overall_acc / self.total
        per_class_acc = self.per_class_acc / self.total
        per_class_iu = self.per_class_iu / self.total
        mIOU = self.mIOU / self.total

        return overall_acc, per_class_acc, per_class_iu, mIOU
