import operator
from functools import reduce
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F

# fmt: off
# the order is important, otherwise we get a circular import error
# fmt: on
from mxlabs_ood_detection.data.datasets.bdd100k_dataset import BDD100k, Cityscapes

__ALL__ = [Cityscapes, BDD100k]

from mxlabs_ood_detection.utils.pytorch_utils import tensor_to_PIL


def colored_mask(mask: torch.Tensor, mapping: Dict) -> Image:
    """
    Takes a single-channel numpy array and converts to correct color-scheme defined by the dataset.
    Args:
        mask: mask as tensor to be converted
        mapping: the dictionary obtained from `dataset._colormap`.

    Returns: PIL image with the given color encoding for the input mask

    """

    mask_ndarray = tensor_to_PIL(mask, numpy_only=True, denormalize=False)
    # Get numpy-mask of shape (1, H, W)
    mask_ndarray_exp = np.expand_dims(mask_ndarray, 0)
    # Tile dimensions to make shape (H, W, 3)
    mask_ndarray_3_channel = np.tile(np.moveaxis(mask_ndarray_exp, 0, -1), 3)
    for k, v in mapping.items():
        mask_ndarray_3_channel[mask_ndarray == k] = v
    return Image.fromarray(mask_ndarray_3_channel)


# def move_to_device(tensors: List[torch.Tensor], gpu: bool = False, numpy: bool = False) -> List:
#    """Moves a list of ts from a one device to another.
#
#    Args:
#        `ts` (List[torch.Tensor]): List of ts to process
#        `gpu` (bool): if True, ts are moved to GPU. Default False.
#        `numpy` (bool): if True, numpy vectors are returned
#
#    Returns:
#        `_tensors`: List of torch.Tensor or np.ndarray.
#    """
#
#    def _to_device(ts: List, device: str):
#        return [getattr(tensor, device)() for tensor in ts]
#
#    _device = "cuda" if gpu else "cpu"
#    _tensors = _to_device(tensors, _device)
#    if numpy:
#        _tensors = [tensor.cpu().numpy() for tensor in _tensors]
#    return _tensors


def match_shape(
    img: torch.Tensor,
    segmentation_mask: torch.Tensor,
    resize: bool = True,
    target: Union[str, None] = "mask",
    target_shape: Tuple[int] = None,
) -> Tuple[torch.Tensor, Any]:
    """Utility to reshape images / mask before calculating loss
    Args:
        `img` (torch.Tensor):
            Image tensor. Shape: (N, Ci, Hi, Wi)
        `segmentation_mask` (torch.Tensor):
            Mask tensor. Shape: (N, Hm, Wm)
        `resize` (bool):
            Switch between resizing or padding with zeros. Default: True.
        `target` (str):
            Reshape / pad to shape of this `target` variable. Default: `mask`.
            Possible choices [`mask`, `image`, `None`]
        `target_shape` (tuple):
            Target shape if `target` is set to None. To be used only if user wants to reshape/pad both image and mask
            to a new shape (H, W).
    Returns:
        `img` (torch.Tensor): transformed image tensor of shape (N, Ci, H, W)
        `segmentation_mask` (torch.LongTensor): transformed mask tensor of type Long and shape (N, H, W)
    """
    if target not in ("mask", "image", None):
        raise TypeError("Unsupported target type.")
    if target is None and target_shape is None:
        raise ValueError(
            "Contradictory arguments. Exactly one argument should hold value between `target` and `target_shape`"
        )

    if img.shape[2:] == segmentation_mask.shape[1:]:
        return img, segmentation_mask.long()

    if target_shape is not None:
        _target_shape = target_shape
    elif target == "mask":
        _target_shape = segmentation_mask.shape[1:]
    elif target == "image":
        _target_shape = img.shape[2:]
    else:
        raise ValueError("Could not determine target_shape, check `target` and `target_shape` arguments.")

    if not resize:
        raise NotImplementedError("Padding not implemented as of now.")

    img = (
        F.interpolate(input=img, size=_target_shape, mode="bilinear", align_corners=True) if target != "image" else img
    )
    segmentation_mask = (
        F.interpolate(input=segmentation_mask.long(), size=_target_shape, mode="nearest")
        if target != "mask"
        else segmentation_mask
    )

    segmentation_mask = torch.squeeze(segmentation_mask, dim=1)
    return img, segmentation_mask


def get_dataset(ds_dict: Dict, use_cuda: bool = False, fast_dev_run: bool = False) -> torch.utils.data.Dataset:
    """
    Instantiates a dataset object given the parameter dict defined in the config_schema.
    The dict must have the key "type" and its value must be the exact name of one of the implemented dataset classes
    Args:
        ds_dict:
            dict obj of the dataset section in the config_schema, parsed by mparams
        use_cuda:
            Use GPU or not
        fast_dev_run:
            similar to `fast_dev_run` from PyTorchLightning, limits the number of elements to allow faster debugging
            and development
    Returns:
        The dataset object with the properties specified in the dict

    """

    for c in IMPLEMENTED_DATASETS:
        if c.__name__ == ds_dict["type"]:
            ds_dict_copy = ds_dict.copy()
            ds_dict_copy.pop("type")  # contains all params but not the type

            if (
                "preprocess" in ds_dict_copy
                and "transforms" in ds_dict_copy["preprocess"]
                and ds_dict_copy["preprocess"]["transforms"] is not None
            ):
                # if we have a segmentation_preprocess section and if it has transforms
                # Parse segmentation_preprocess section to a tuple of torch transforms
                img_preprocess = tuple(
                    [
                        getattr(torchvision.transforms, transform)(**kwargs)
                        for (transform, kwargs) in ds_dict_copy["preprocess"]["transforms"].items()
                    ]
                )
            else:
                img_preprocess = ()

            ds_dict_copy["preprocess"] = img_preprocess
            if "preprocess" in ds_dict_copy:
                ds_dict_copy.pop("preprocess")

            # Instantiate obj with the whole dict as kwargs
            return c(**ds_dict_copy, cuda=use_cuda, fast_dev_run=fast_dev_run)


def get_split_sizes(dataset, pct=0.8):
    num_elements = len(dataset)
    lower = int(num_elements * pct)
    return [lower, num_elements - lower]


def get_normalisation_factors(
    training_data: torch.utils.data.DataLoader, stabilize_results=True
) -> (torch.Tensor, torch.Tensor):
    """Computes the mean and std for each channel in the given training data. Assumes the data comes in [b, c, h, w]

    Args:
        training_data
        stabilize_results: if `True` adds to small epsilon to std

    Returns:
        (mean, std) of shape
    """

    def _step(x):
        x = x.view(*x.shape[:2], -1)
        return (
            x.shape[0],
            # mean over all pixels (a full layer), then sum up the batched items
            x.mean(dim=2).sum(0),
            x.std(dim=2).sum(0),
        )

    def _element_wise_tuple_sum(a, b):
        return map(operator.add, a, b)

    ret = dict(zip(["n_items", "mean", "std"], reduce(_element_wise_tuple_sum, (_step(x) for x in training_data))))

    mean = ret["mean"] / ret["n_items"]
    std = ret["std"] / ret["n_items"]

    if stabilize_results:
        std += 1e-7

    return mean, std
