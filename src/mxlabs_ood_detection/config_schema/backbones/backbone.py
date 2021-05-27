from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class BackboneConfigSchema:
    model: Dict[str, Any] = MISSING
    trainer: Optional[Dict[str, Any]] = None
