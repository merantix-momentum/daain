from dataclasses import dataclass
from typing import Dict, List, Tuple

from omegaconf import MISSING


@dataclass
class ActivationSpace:
    rejection_boundaries: Dict[str, List[List[int]]] = MISSING
    activation_space_shapes: Dict[str, List[List[int]]] = MISSING
    # rejection_boundaries: Dict[str, List[Tuple[int, int]]] = MISSING
    # activation_space_shapes: Dict[str, List[Tuple[int, int]]] = MISSING
