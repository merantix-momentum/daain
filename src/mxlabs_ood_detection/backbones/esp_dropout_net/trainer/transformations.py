import random
from functools import reduce

import PIL
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize:
    def __init__(self, size):
        self.fn_img = T.Resize(size)
        self.fn_label = T.Resize(size, interpolation=PIL.Image.NEAREST)

    def __call__(self, img, label):
        img = self.fn_img(img)
        label = self.fn_label(label)

        return img, label


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
        image = F.crop(img, *crop_params)
        target = F.crop(label, *crop_params)
        return image, target


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=[0.5, 0.75, 1.0, 1.5, 2.0]):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, label):
        crop_params = T.RandomResizedCrop.get_params(img, scale=self.scale, ratio=self.ratio)  # (size)
        img = F.resized_crop(img, *crop_params, size=self.size)
        label = F.resized_crop(label, *crop_params, size=self.size)
        return img, label


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(label)
        else:
            return image, label


class Normalize:
    def __init__(self, mean, std):
        self.fn = T.Normalize(mean, std)

    def __call__(self, image, label):
        return self.fn(image), label


class ToTensor:
    def __init__(self, scale=1):
        """
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        """
        self.scale = scale  # original images are 2048 x 1024

        self.to_tensor_transform = T.ToTensor()

    def __call__(self, image, label):
        if self.scale != 1:
            model_output_size = np.array(image.size) // self.scale
            label = T.Resize(model_output_size.tolist(), interpolation=PIL.Image.NEAREST)(label)
            image = self.to_tensor_transform(image).transpose(2, 1)
        else:
            # for some weird reason
            image = self.to_tensor_transform(image)

        return image, torch.as_tensor(np.array(label), dtype=torch.int64)


class Compose:
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        return reduce(lambda acc, el: el(*acc), self.transforms, args)