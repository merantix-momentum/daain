import torch


def rgb2gray(img: torch.Tensor) -> torch.Tensor:
    """Converts a single image (not batched) from RGB to gray scale.

    source: https://stackoverflow.com/questions/14330/rgb-to-monochrome-conversion
    """
    return (0.2125 * img[:, 0]) + (0.7154 * img[:, 1]) + (0.0721 * img[:, 2])
