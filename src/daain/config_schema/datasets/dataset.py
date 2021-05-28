from dataclasses import dataclass
from typing import Any, Dict, List

from omegaconf import MISSING


@dataclass
class DatasetPaths:
    images: Dict[str, str] = MISSING
    targets: Dict[str, str] = MISSING


@dataclass
class Transformation:
    name: str = MISSING
    kwargs: Dict[str, Any] = MISSING


@dataclass
class DatasetConfigSchema:
    name: str = MISSING
    paths: DatasetPaths = MISSING
    transformations: List[Transformation] = MISSING
