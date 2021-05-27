from collections import OrderedDict
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mxlabs_ood_detection.backbones.esp_net.espnet import CustomESPNet
from mxlabs_ood_detection.utils.poisson_disc_sampling import gridify_poisson_disc_sampling, poisson_disc_samples


def create_mask(
    min_distance: float, rejection_boundaries: bool = None, activation_space_shape: bool = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Use this to create the sampling masks

    Args:
        min_distance: The minimal distance for the Poisson Disc Sampling algorithm
        rejection_boundaries: If certain layers should be ignored (set to False). Mostly used to confine the sampling
                              to certain layer types, e.g. only convolutional layers.
        activation_space_shape: The space from which to sample from. The default value corresponds to the activations
                                created by the ESPNet with input dimensions of 256 x 512.
    Returns:
        Either only the boolean mask or the boolean mask and the gridified version of it, but only if no
        rejection_boundaries are given. Sadly the gridifying-algorithm was written for wholes in the mask.
    """
    if activation_space_shape is None:
        # activation space for esp_net with image dim 256 x 512
        activation_space_shape = (5297, 256, 512)
    grid = poisson_disc_samples(activation_space_shape, minimum_distance_between_samples=min_distance)
    mask = np.zeros(activation_space_shape, dtype=bool)
    t = grid.astype(int)
    mask[t[:, 0], t[:, 1], t[:, 2]] = True

    # TODO just do the gridifying algorithm AFTER this. But this would require ~5min of my time. -> Next intern.
    if rejection_boundaries is not None:
        for low, high in rejection_boundaries:
            mask[low:high, :, :] = False

    if rejection_boundaries is None:
        return mask, gridify_poisson_disc_sampling(grid, activation_space_shape, min_distance=min_distance).astype(int)
    else:
        return mask


def upsample_activations(
    activations: Iterable[Union[np.ndarray, torch.Tensor]], img_size=None
) -> Union[np.ndarray, torch.Tensor]:
    # we want to upsample the last two dimensions as they correspond to the width and height
    # remember the input / output shape [batch_size, n_channels, height, width],
    # see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # get the max shapes in both dimensions
    if img_size is None:
        max_shapes = np.max(np.array([np.array(a.shape[-2:]) for a in activations]), axis=0)
        upsampler = torch.nn.UpsamplingNearest2d(size=max_shapes.tolist())
    else:
        upsampler = torch.nn.UpsamplingNearest2d(size=img_size)

    # flattens and concats all activations, still in [batch_size, n_channels, width, height]
    upsampled_activations = torch.cat([upsampler(a).detach().cpu() for a in activations], dim=1)

    return upsampled_activations


def compute_coactivations(activations: torch.Tensor) -> torch.Tensor:
    """Use this to compute the coactivations, nice to plot and reason about them."""
    on_cuda = activations.cuda()
    a_act_r = torch.abs(on_cuda).view(1, -1)
    # now make out of a 1-d array a 2-d, square matrix by copying the values
    a_act_r = torch.repeat_interleave(a_act_r, repeats=a_act_r.size()[-1], dim=0)
    a_act_r = a_act_r * torch.transpose(a_act_r, 0, 1)
    return a_act_r.cpu()


def _write_activations_to_zarr(activations, image_ids, path_template, zarr_root):
    for i, image_id in enumerate(image_ids):
        zarr_root[path_template.format(image_id=image_id)] = activations[i, :].detach().cpu().numpy()


def create_pre_hook_for_layer(layer_db, layer):
    def hook(model, input, output):
        layer_db[layer] = input[0].detach()

    return hook


def create_post_hook_for_layer(layer_db, layer):
    def hook(model, input, output):
        layer_db[layer] = output.detach()

    return hook


def retrieve_modules_from_esp_net(esp_net: CustomESPNet) -> List[Tuple[str, torch.nn.Module]]:
    """Retrieves modules for the ESPnet and only the that.

    Use it as a starting ground if you want to record activations from different models.
    """

    def get_modules(layer, mod):
        new_modules = []

        for n, m in mod.named_children():
            kids = list(m.named_children())

            if len(kids):
                new_modules += get_modules(f"{layer}_{n}", m)
            else:
                new_modules += [(f"{layer}_{n}", m)]

        return new_modules

    return get_modules("esp_net", esp_net.esp_net)


def compute_activations(
    model: CustomESPNet,
    loader: DataLoader,
    attacks: List[str],
    selections: List[Tuple[np.ndarray, zarr.DirectoryStore, str]],
    force: bool = False,
    n_data_points: int = -1,
) -> None:
    """Use this to record activations for multiple masks at the same time. They will be stored in the corresponding
    zarr-directory-store (see parameters)

    Args:
        model: The model / backbone from which the activations should be recorded
        loader: The dataloader to load the data (images in this case), has to be a loader around a PerturbedDataset
        attacks: List of the attacks and out-of-distribution distortions. Just the strings to load them from a
                 PerturbedImageTensor
        selections: List of tuples (the mask (either boolean or coordinates), the zarr target storage, mask type (
                    either pre or post)
        force: Indicates if already present values should be overwritten or not
        n_data_points: Use this for debugging. If below 0 it will run through all data points.
    """

    # note that not the activation-space for these two is not the same! depending on the layout of the backbone-model
    # some layer-outputs might be reused at multiple locations
    activations_post = OrderedDict()  # sink for post activations
    activations_pre = OrderedDict()  # sink for pre activations

    for n, m in retrieve_modules_from_esp_net(model):
        m.register_forward_hook(create_post_hook_for_layer(activations_post, n))
        m.register_forward_hook(create_pre_hook_for_layer(activations_pre, n))

    # both next functions are not necessary, but might be useful for further exploration / debugging
    def _downsample_mask(size):
        def to_tensor(tensor, dtype):
            return torch.as_tensor(tensor, dtype=dtype)

        def fn(x):
            return to_tensor(torch.nn.functional.adaptive_avg_pool2d(to_tensor(x, dtype=float), size), dtype=bool)

        return fn

    def _get_layer_shapes(activations: Iterable[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the max and min shapes for the given list of activations. Highest and lowest resolution."""
        shapes = np.array([np.array(a.shape[-2:]) for a in activations])

        return np.max(shapes, axis=0), np.min(shapes, axis=0)

    mask_transform = None  # noqa F841
    mask = None
    reshape_to_rectangle = True  # noqa F841
    for batch_idx, x in tqdm(
        enumerate(loader), desc=f"computing activations for model: {model.__class__.__name__}", total=len(loader)
    ):
        for attack in [*attacks, "unperturbed"]:
            activations_post.clear()
            activations_pre.clear()
            zarr_path_template = f"{attack}/{{image_id}}"

            if attack == "unperturbed":
                input_image = x["normal_image"]
            else:
                input_image = x["perturbed"][attack]["perturbed_image"]

            model(input_image)  # triggering the recording

            upsampled_post_activations = upsample_activations(activations_post.values(), img_size=(256, 512))
            upsampled_pre_activations = upsample_activations(activations_pre.values(), img_size=(256, 512))

            # selections is a list of masks (either as a list of ints or a boolean matrix) and the corresponding
            # zarr storage
            # TODO add second loading bar
            for selection, zarr_root, recording_type in selections:
                if not force and all(
                    [zarr_path_template.format(image_id=image_id) in zarr_root for image_id in x["image_id"]]
                ):
                    continue

                if selection is None:
                    _write_activations_to_zarr(upsampled_post_activations, x["image_id"], zarr_path_template, zarr_root)
                else:
                    if recording_type == "post":
                        upsampled_of_interest = upsampled_post_activations
                    else:
                        upsampled_of_interest = upsampled_pre_activations

                    if str(selection.dtype).startswith("int"):
                        selection_shape = selection.shape
                        _sel = selection.reshape(-1, 3)
                        selected_upsampled_activations = upsampled_of_interest[
                            :, _sel[:, 0], _sel[:, 1], _sel[:, 2]
                        ].reshape(-1, *selection_shape[:-1])
                    else:
                        selected_upsampled_activations = upsampled_of_interest[:, selection]

                    _write_activations_to_zarr(
                        selected_upsampled_activations, x["image_id"], zarr_path_template, zarr_root
                    )

        if n_data_points > 0 and batch_idx > n_data_points:
            break

    return mask


class FixedPermutationTransform:
    def __init__(self):
        self.original_shape = None
        self.random_index = None

    def __call__(self, x):
        if self.original_shape is None:
            self.original_shape = x.shape
            self.random_index = torch.randperm(x.nelement())

        return x.reshape(-1)[self.random_index].reshape(self.original_shape)


class ActivationDataset(Dataset):
    """ActivationDataset helper loader class inheriting from Dataset, to be used with torch.utils.data.DataLoader.
    Or just use the `loader` method to create a Dataloader."""

    def __init__(self, zarr_root, transform=None, limit_elements=-1):
        """

        Args:
            zarr_store:
            mask:
            original_data_split: `trainer`, `test` or `val`, refers to the data on which the original model was trained on
            attack:
            transform:
        """

        self.zarr_root = zarr_root

        self.keys = list(self.zarr_root.keys())

        if limit_elements > 0:
            self.keys = self.keys[:limit_elements]

        if transform is None:
            self.transform = lambda x: torch.Tensor(x)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        return self.transform(self.zarr_root[self.keys[idx]])

    def input_shape(self):
        return self.__getitem__(0).shape

    def loader(self, **kwargs):
        return DataLoader(self, **{"batch_size": 1, "num_workers": 1, **kwargs})
