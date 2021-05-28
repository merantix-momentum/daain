import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from daain.data import ActivationDataset
from daain.data.datasets import get_split_sizes


def _get_zarr_root(dataset_data, attack_name, mask, submask):
    return dataset_data[f"/{mask}/{submask}/{attack_name}"]


def add_key(x):
    # adding key and reshaping such that it conforms to the PyTorch Tensor Guide
    # [batch (not present here, comes after), channels, element]
    x = torch.cat((x.unsqueeze(1), torch.eye(x.shape[0])), dim=1).permute(1, 0)
    return x


def get_data_transform(add_keys=False):
    """
    Args:
        add_keys: For attention layers
    """
    data_transforms = [lambda x: torch.Tensor(x)]

    if add_keys:
        data_transforms.append(add_key)

    return torchvision.transforms.Compose(data_transforms)


def get_data(datasets, dataset_assignments, mask, submask, transform, data_loader_args, fast_dev_run=False):
    train_dataset = ActivationDataset(
        zarr_root=_get_zarr_root(datasets[0][2], dataset_assignments["train"], mask=mask, submask=submask),
        transform=transform,
    )

    test_dataset = ActivationDataset(
        zarr_root=_get_zarr_root(datasets[1][2], dataset_assignments["test"], mask=mask, submask=submask),
        transform=transform,
    )

    train, val = random_split(train_dataset, get_split_sizes(train_dataset))
    train_loader = DataLoader(train, **{**data_loader_args, "shuffle": True})
    val_loader = DataLoader(val, **{**data_loader_args, "shuffle": False})

    test_loader = DataLoader(test_dataset, **{**data_loader_args, "shuffle": False})

    def _get_activations_loader(attack_name):
        return DataLoader(
            # Subset(
            ActivationDataset(
                zarr_root=_get_zarr_root(
                    dataset_data=datasets[1][2], attack_name=attack_name, mask=mask, submask=submask
                ),
                limit_elements=100 if fast_dev_run else -1,
                transform=transform,
            ),
            # ),
            # val_loader.dataset.indices), # to insure that only the validation data is used for the adversary as
            # well
            **{**data_loader_args, "shuffle": False},
        )

    perturbed_dataset_loaders, out_of_distribution_loaders = [
        [(f"{attack_name}", _get_activations_loader(attack_name)) for attack_name in dataset_assignments[k]]
        for k in ["attacks", "out_of_distributions"]
    ]

    return (
        train_loader,
        val_loader,
        test_loader,
        perturbed_dataset_loaders,
        out_of_distribution_loaders,
        train_dataset.input_shape(),
    )
