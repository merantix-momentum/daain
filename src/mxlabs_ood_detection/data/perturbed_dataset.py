import os
from dataclasses import asdict as dataclass_to_dict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, OrderedDict, Union

import gcsfs
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

#
# Just some helper PODs to increase readability
#
from mxlabs_ood_detection.constants import IMAGENET_MEAN, IMAGENET_STD
from mxlabs_ood_detection.data.datasets.labels import cityscape
from mxlabs_ood_detection.utils.pil_utils import load_image


@dataclass
class AttackedImage:
    perturbed_image: PIL.Image.Image
    output: PIL.Image.Image
    mask: np.ndarray


@dataclass
class PerturbedImage:
    image_id: int
    # not correct, too lazy too look how they handle images, but not an np.array
    normal_image: PIL.Image.Image
    normal_output: PIL.Image.Image
    perturbed: Dict[str, AttackedImage]


@dataclass
class AttackedImagePath:
    perturbed_image: str
    output: str
    mask: str


@dataclass
class PerturbedImagePath:
    image_id: str
    normal_image: str
    normal_output: str
    perturbed: Dict[str, AttackedImagePath]


@dataclass
class AttackedImageTensor:
    perturbed_image: torch.Tensor
    output: torch.Tensor
    mask: torch.Tensor  # TODO remove this


@dataclass
class PerturbedImageTensor:
    image_id: str
    normal_image: torch.Tensor
    normal_output: torch.Tensor
    perturbed: Dict[str, AttackedImageTensor]


# and here we see that a dict might have been better...
def _apply_transform_(fn, sample):
    return PerturbedImageTensor(
        image_id=sample.image_id,
        normal_image=fn(sample.normal_image),
        normal_output=fn(sample.normal_output),
        perturbed={
            k: AttackedImageTensor(perturbed_image=fn(v.perturbed_image), output=fn(v.output), mask=v.mask)
            for k, v in sample.perturbed.items()
        },
    )


class _Transform_:
    def __init__(self, transformation):
        self.transformation = transformation

    def __call__(self, sample):
        return _apply_transform_(self.transformation, sample)


def create_preprocessing_pipeline(
    transforms: OrderedDict = None, normalize: bool = False, resize: bool = True, use_cuda: bool = True
):
    if transforms is not None:
        pipeline = [_Transform_(getattr(T, transform_name)(**kwargs)) for transform_name, kwargs in transforms.items()]
    else:
        pipeline = []
        if resize:
            # super bad hard-coded value...
            pipeline.append(_Transform_(T.Resize(size=512)))

    pipeline.append(_Transform_(F.to_tensor))

    if normalize:
        pipeline.append(_Transform_(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)))

    if use_cuda:
        pipeline.append(_Transform_(lambda x: x.cuda()))

    return T.Compose(pipeline)


class PerturbedDataset(Dataset):
    """Loads the perturbed data generated by the `Testing Framework`

    The loading processes goes like this:
        1. Get all image paths
        2. Load the image for the given index
        3. Transform the image to a tensor

    """

    # TODO move this somewhere else
    ignore_index = 255
    num_classes = 19

    # TODO add model and what not as options
    # TODO IMO preprocessing
    def __init__(
        self,
        attacks: List[str] = None,
        project: str = None,  # gcs bucket
        bucket: str = None,
        root_path: str = None,  # local
        transforms: List = None,
        use_cuda: bool = True,
    ):
        """
        Args:
            attacks: List of attacks to use (datasets to load)
            project: `project` and `bucket` have both to be present in order to use streaming data from a google
                cloud bucket
            bucket: `project` and `bucket` have both to be present in order to use streaming data from a google cloud
                bucket
            root_path: uses local data
            transforms: List of PyTorch-style transformations of the images. If none the default transformation will
                be applied: [Resize(512), ToTensor(), (lambda x: x.cuda())]
        """

        if all(i is None for i in (project, bucket)) and root_path is None:
            raise ValueError("Set at least the root_path or the project+bucket pair.")

        if all(i is not None for i in (project, bucket)):
            self.gfs = gcsfs.GCSFileSystem(project=project)
            self.root_path = None
        else:
            self.gfs = None  # meaning we'll use the local file-system
            self.root_path = root_path

        self.project = project
        self.bucket = bucket
        self.attacks = attacks or list()

        self.image_paths = list(self._get_image_paths_())
        self.images_by_id = {img.image_id: img for img in self.image_paths}

        self.transform = create_preprocessing_pipeline(transforms, use_cuda=use_cuda)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        return dataclass_to_dict(self.transform(self.get_image(idx)))

    def _get_image_paths_(self) -> Iterable[PerturbedImagePath]:
        if self.gfs:
            raise NotImplementedError
        else:
            # to make sure we only select the images that are present in all attacks
            # image_candidates = [os.listdir(os.path.join(self.root_path, f"unperturbed_images_{a}")) for a in
            #                    self.attacks]
            # for i in min(image_candidates, key=len):
            for i in os.listdir(os.path.join(self.root_path, "unperturbed_images")):
                yield PerturbedImagePath(
                    image_id=i[:-4],  # basename of the file minus the ending
                    normal_image=os.path.join(self.root_path, "unperturbed_images", i),
                    normal_output=os.path.join(self.root_path, "unperturbed_output", i),
                    perturbed={
                        attack: AttackedImagePath(
                            perturbed_image=os.path.join(self.root_path, f"perturbed_images_{attack}", i),
                            output=os.path.join(self.root_path, f"perturbed_output_{attack}", i),
                            mask=os.path.join(self.root_path, f"patch_mask_{attack}", f"{i[:-4]}.npy"), # can be ignored
                        )
                        for attack in self.attacks
                    },
                )

    def get_image(self, idx: int) -> PerturbedImage:
        return self._get_entry_(self.image_paths[idx])

    def _open_image_(self, image_path) -> PIL.Image.Image:
        if self.gfs:
            with self.gfs.open(image_path, "rb") as f:
                img = PIL.Image.open(f)
                img.load()
                # _ = np.asarray(img)  # simple op to trigger the fetching process, super weird
        else:
            img = load_image(image_path)
            img.load()

        if img.mode == "RGBA":
            # we want 3 channel tensors
            img = img.convert("RGB")

        return img

    def _open_mask_(self, mask_path, default_size=None) -> Union[None, np.ndarray]:
        if os.path.isfile(mask_path):
            return np.load(mask_path)
        else:
            return np.ones(default_size, dtype=bool)

    def _get_entry_(self, img: PerturbedImagePath) -> PerturbedImage:
        return PerturbedImage(
            image_id=int(img.image_id),
            normal_image=self._open_image_(img.normal_image),
            normal_output=self._open_image_(img.normal_output),
            perturbed={
                k: AttackedImage(
                    perturbed_image=self._open_image_(a.perturbed_image),
                    output=self._open_image_(a.output),
                    mask=self._open_mask_(a.mask, self._open_image_(a.output).size),
                )
                for k, a in img.perturbed.items()
            },
        )

    def get_image_by_id(self, image_id: str) -> PerturbedImage:
        return self._get_entry_(self.images_by_id[image_id])

    @property
    def colormap(self):
        return {label.trainId: label.color for label in cityscape.labels}

    @property
    def labelmap(self) -> Dict:
        return {label.trainId: label.name for label in cityscape.labels}
