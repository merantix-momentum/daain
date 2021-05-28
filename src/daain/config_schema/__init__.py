import os

# TODO find a better way
# most ugly way to get the correct path...
# PATH_TO_CONFIGS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                               "../../../../../../configs/"))
from typing import Any

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

PATH_TO_CONFIGS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../configs/"))


def get_relpath_to_configs(start) -> str:
    return os.path.relpath(PATH_TO_CONFIGS, start)


def load_config_part(start: str, config_path: str, config_name: str) -> Any:
    """Loads a config (or part of it) given by the config_path (inside the config directory)
    and the config_name (including `.yaml`).

    Example:
        # will return the cityscapes config stored in `dataset_paths`.
        load_config_part(__file__, "dataset_paths", "cityscapes.yaml")

    Parameters
    ----------
    start        # sadly necessary because hydra expects a relative path...
    config_path  # the /config/path/inside/the/config/folder
    config_name  # the name of the config, including `.yaml`

    """
    initialize(config_path=os.path.join(get_relpath_to_configs(os.path.dirname(start)), config_path))
    cfg = compose(config_name)
    return OmegaConf.to_yaml(cfg, resolve=True)
