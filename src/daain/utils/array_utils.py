from itertools import filterfalse, tee
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch


def create_random_mask(size: Union[int, List[int], np.ndarray], num_true: int) -> np.ndarray:
    """Returns an np.array in the given shape / size where *exactly* `num_true` randomly chosen
    elements are set to `True`, all other elements are set to `False`.

    Args:
       size (int or a tuple of ints): shape, dimensions, you name it
       num_true (int): Number of True elements, rest is False

    Returns:
        np.array with dtype==bool. A mask.
    """

    if np.prod(size) < num_true:
        raise ValueError("Cannot set more values to true than available.")

    mask = np.zeros(np.prod(size), dtype=np.int)
    mask[:num_true] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool)

    mask.resize(size)

    return mask


def create_grid_mask_2d(size: Union[List[int], np.ndarray], num_selected: Union[List[int], np.ndarray]) -> np.ndarray:
    """Returns an evenly spaced mask over both dimensions."""
    y = np.linspace(0, size[0] - 1, num=num_selected[0], endpoint=True, dtype=int)
    x = np.linspace(0, size[1] - 1, num=num_selected[1], endpoint=True, dtype=int)

    xx, yy = np.meshgrid(x, y, sparse=True)

    mask = np.zeros(size, dtype=bool)
    mask[yy, xx] = True

    return mask


def create_patch_mask(
        mask_shape: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        patch_loc: Union[str, int, Tuple[int, int]] = "random",
        shift: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns a boolean mask

    Args:
        mask_shape:
        patch_size:
        patch_loc:
            Can be either 'random' (default) or a given location, in which case it is the upper left corner of the mask

    Returns:
        np.array with dtype==bool. A mask.
    """
    if patch_loc == "random":
        start = np.random.randint(np.array(mask_shape) - patch_size + 1)
    elif patch_loc == "center":
        start = [(mask_shape[0] - patch_size[0]) // 2, (mask_shape[1] - patch_size[1]) // 2]
    else:
        start = patch_loc

    if shift is not None:
        start = np.array(start) + shift

    mask = np.zeros(mask_shape, dtype=np.bool)
    if isinstance(mask_shape, (tuple, list)):
        if isinstance(patch_size, (tuple, list)):
            mask[start[0]: start[0] + patch_size[0], start[1]: start[1] + patch_size[1]] = True
        else:
            mask[start[0]: start[0] + patch_size, start[1]: start[1] + patch_size] = True
    else:
        mask[start: start + patch_size] = True

    return mask


def get_mask_shape(mask: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    """Returns the inner shape of the mask, assuming it is of rectangular shape.

    Args:
        mask: boolean or int {0, 1} mask

    Returns:
        The inner shape of the mask
    """

    if isinstance(mask, torch.Tensor):
        def inner_fn(mask, axis):
            t = mask.sum(dim=axis)
            return t[t != 0][0]

    else:
        def inner_fn(mask, axis):
            t = np.sum(mask, axis=axis)
            return t[t != 0][0]

    return (inner_fn(mask, 0), inner_fn(mask, 1))


def cut_out_using_mask(
        data: torch.Tensor, masks: Union[torch.Tensor, np.ndarray], reshape_to_rectangle=True
) -> torch.Tensor:
    """Cuts out mask, expects both data and masks to be batched.

    Args:
        data:
        masks:

    Returns:

    """

    # using the shape to get [batch, features, mask height, mask weight]
    def _reshaping(x, mask):
        if reshape_to_rectangle:
            return x.reshape(-1, *get_mask_shape(mask))
        else:
            return x

    if len(masks.shape) == 3:
        return torch.stack([_reshaping(data[i, :, masks[i]], masks[i]) for i in range(data.shape[0])])
    else:
        return torch.stack([_reshaping(data[i, :, masks], masks) for i in range(data.shape[0])])


def t2np(tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """pytorch tensor -> numpy array"""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def partition(pred: Callable[[Any], bool], iterable: List) -> Tuple[List, List]:
    """Use a predicate to partition entries into false entries and true entries.
    Note that this will iterate twice over the given iterable.
    Example: partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    """
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def flatten(xs: List[List[Any]]) -> List[Any]:
    """Flattens the given list of lists."""
    return [item for sublist in xs for item in sublist]


def bflatten(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Flattens the given array `x` batch wise.
    Example: Given x.shape == [10, 3, 5, 5], bflatten(x).shape == [10, 3 * 5 * 5
    """
    return reshape(x, (x.shape[0], -1))


def reshape(x: Union[np.ndarray, torch.Tensor], shape: Union[Any, List[int], Tuple[int]]) -> Union[np.ndarray,
                                                                                                   torch.Tensor]:
    """Torch or Numpy agnostic reshaping"""
    if isinstance(x, np.ndarray):
        return x.reshape(*shape)
    else:
        return torch.reshape(x, shape)


def bmean_over_voxels(xs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch-wise mean computation"""
    if isinstance(xs, np.ndarray):
        return np.mean(bflatten(xs), axis=1)
    else:
        return torch.mean(bflatten(xs), dim=1)


def bmean_squared_over_voxels(xs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Batch-wise mean squared computation"""
    # mean over all but the batch, full image
    return bmean_over_voxels(xs ** 2)


def btranspose(tensor: torch.Tensor) -> torch.Tensor:
    """Batch-wise transpose. Assumes that tensor has dimension of 3: [batch, features, samples]"""
    if tensor.dim() != 3:
        raise ValueError("The given shape is not supported.")

    return torch.transpose(tensor, 1, 2)
