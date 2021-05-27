import sys

import numpy as np
import torch
import torch.nn as nn


def _enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def get_monte_carlo_predictions(
    data_loader: torch.utils.data.Dataloader,
    num_forward_passes: int,
    model: torch.nn.Module,
    n_classes: int,
    n_samples: int,
):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    num_forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(num_forward_passes):
        predictions = np.empty((0, n_classes))
        # to allow get multiple different dropout masks
        model.eval()
        _enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            image = image.to(torch.device("cuda"))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))
        # shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes
    # shape (n_samples, n_classes)
    variance = np.var(dropout_predictions, axis=0)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(
        np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon), axis=-1), axis=0
    )  # shape (n_samples,)

    return mean, variance, entropy, mutual_info
