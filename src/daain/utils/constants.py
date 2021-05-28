import logging
import os

from daain.constants import IMAGENET_MEAN, IMAGENET_STD
from daain.utils import file_transfer_utils

LOCAL_DATA_DIR = "/your/path/to/the/data"

CLOUD_DATA_DIR = None  # can be set to a google cloud path starting with gs://
CACHE_DATA = True
# if cache_data is true and if CLOUD_DATA_DIR is set, the cloud bucket will be stored under
# local_data_dir if it doesnt exist yet and the local paths pointing there will be used.
# Otherwise only cloud paths are used, which only works on a GCE instance

# If we normalize the image tensors that are in the range of [0,1],
# the RGB channels of the normalized tensor are in the range of the two extreme values below:
MAX_RGB_VALS_NORMALIZED = [(1 - IMAGENET_MEAN[x]) / IMAGENET_STD[x] for x in range(len(IMAGENET_STD))]
MIN_RGB_VALS_NORMALIZED = [(0 - IMAGENET_MEAN[x]) / IMAGENET_STD[x] for x in range(len(IMAGENET_STD))]

if CACHE_DATA:
    DATA_DIR = LOCAL_DATA_DIR
    if not os.path.isdir(LOCAL_DATA_DIR):
        if CLOUD_DATA_DIR is None:
            logging.info(f"Use local data at {LOCAL_DATA_DIR}.")
        else:
            logging.info(f"Cloud bucket {CLOUD_DATA_DIR} will be downloaded to {LOCAL_DATA_DIR}.")
            file_transfer_utils.copy_directory_to_local(cloud_path=CLOUD_DATA_DIR, local_path=LOCAL_DATA_DIR)
    else:
        logging.info(f"{LOCAL_DATA_DIR} already exists. Cloud bucket {CLOUD_DATA_DIR} will not be downloaded.")
else:
    DATA_DIR = CLOUD_DATA_DIR


WEIGHT_PATH = {
    "esp_net": os.path.join(DATA_DIR, "model_weights", "esp_net", "decoder"),
}


