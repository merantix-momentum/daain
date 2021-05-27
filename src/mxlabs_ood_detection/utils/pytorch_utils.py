import random
from enum import Enum
from operator import methodcaller

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.backends import cudnn

from mxlabs_ood_detection.constants import IMAGENET_MEAN, IMAGENET_STD


class Device(Enum):
    CUDA = "cuda"
    CPU  = "cpu"


def apply_xavier_normal_initialization(layer):
    if layer.weight:
        torch.nn.init.xavier_normal_(layer.weight)
    if layer.bias:
        torch.nn.init.zeros_(layer.bias)

def get_device(obj):
    if is_on_cuda(obj):
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def is_on_cuda(obj):
    """
    Get cuda state of any object.

    :param obj: an object (a tensor or an `torch.nn.Module`)
    :raise TypeError:
    :return: True if the object or the parameter set of the object
             is on GPU
    """
    if isinstance(obj, nn.Module):
        try:
            return next(obj.parameters()).is_cuda
        except StopIteration:
            return None
    elif hasattr(obj, "is_cuda"):
        return obj.is_cuda
    else:
        raise TypeError("unrecognized type ({}) in args".format(type(obj)))


def is_cuda_consistent(*args):
    """
    See if the cuda states are consistent among variables (of type either
    tensors or torch.autograd.Variable). For example,

        import torch
        from torch.autograd import Variable
        import torch.nn as nn

        net = nn.Linear(512, 10)
        tensor = torch.rand(10, 10).cuda()
        assert not is_cuda_consistent(net=net, tensor=tensor)

    :param args: the variables to test
    :return: True if len(args) == 0 or the cuda states of all elements in args
             are consistent; False otherwise
    """
    result = dict()
    for v in args:
        cur_cuda_state = get_device(v)
        cuda_state = result.get("cuda", cur_cuda_state)
        if cur_cuda_state is not cuda_state:
            return False
        result["cuda"] = cur_cuda_state
    return True


def make_cuda_consistent(refobj, *args):
    """
    Attempt to make the cuda states of args consistent with that of ``refobj``.
    If any element of args is a Variable and the cuda state of the element is
    inconsistent with ``refobj``, raise ValueError, since changing the cuda state
    of a Variable involves rewrapping it in a new Variable, which changes the
    semantics of the code.

    :param refobj: either the referential object or the cuda state of the
           referential object
    :param args: the variables to test
    :return: tuple of the same data as ``args`` but on the same device as
             ``refobj``
    """
    ref_cuda_state = refobj if type(refobj) is bool else get_device(refobj)
    if ref_cuda_state is None:
        raise ValueError("cannot determine the cuda state of `refobj` ({})".format(refobj))
    move_to_device = methodcaller("cuda" if ref_cuda_state else "cpu")

    result_args = list()
    for v in args:
        cuda_state = get_device(v)
        if cuda_state != ref_cuda_state:
            if isinstance(v, Variable):
                raise ValueError("cannot change cuda state of a Variable")
            elif isinstance(v, nn.Module):
                move_to_device(v)
            else:
                v = move_to_device(v)
        result_args.append(v)
    return tuple(result_args)


def enforce_reproducibility(random_seed):
    """Tries to enforce reproducibility of experiments"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(random_seed)


def to_device(x, cuda=True):
    if cuda:
        return x.cuda()
    else:
        return x.cpu()


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes a normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)  # small epsilon added to prevent division errors
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        # todo why the clone?
        return super().__call__(tensor.clone())


DENORMALIZE_TRANS = NormalizeInverse()  # so we dont need to instantiate for every tensor_to_PIL call

def tensor_to_PIL(x: torch.Tensor, denormalize: bool = True, numpy_only: bool = False) -> object:
    """
    Helper func to get a PIL from a tensor.
    Args:
        x: Tensor representing image
        denormalize: If true will undo a normalization with ImageNet values
        numpy_only: If true will return an ndarray instead of an Image

    Returns: A numpy array if numpy_only=True, otherwise a PIL Image

    """
    if x.dim() == 3:
        if denormalize:
            x = DENORMALIZE_TRANS(x)
        # Assuming that 3 channel tensors only need to switch axis
        # CHW -> HWC
        x = x.permute(1, 2, 0) * 255
    img_ndarray = x.byte().cpu().numpy()
    if numpy_only:
        return img_ndarray
    return Image.fromarray(img_ndarray)