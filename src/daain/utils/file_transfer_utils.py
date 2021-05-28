import logging
import os
import subprocess

AWS_PREFIX = "arn:aws:s3:::"
GS_PREFIX = "gs://"


def is_gcs_uri(path):
    return path.startswith(GS_PREFIX)


def is_aws_uri(path):
    return path.startswith(AWS_PREFIX)


def is_cloud_uri(path):
    return is_aws_uri(path) or is_gcs_uri(path)


def create_directory(path, exist_ok=False):
    """Creates a local directory.

    Does nothing if a Google Cloud Storage or AWS-S3 path is passed.

    Args:
      path: the path to create.
      exist_ok: Does not raise if directory exists. Behaves like mkdir -p.

    Raises:
      ValueError: if path is a file or os.makedir fails.
    """
    if is_cloud_uri(path):
        return
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return
    if os.path.isfile(path):
        raise ValueError('Unable to create location. "%s" exists and is a file.' % path)

    try:
        os.makedirs(path)
    except OSError:
        # Copy cpython os.makedirs exist_ok handling: https://github.com/python/cpython/blob/master/Lib/os.py
        if not exist_ok or not os.path.isdir(path):
            raise
    except:  # pylint: disable=broad-except # noqa
        raise ValueError('Unable to create location. "%s"' % path)


def copy_dir(src: str, dest: str, verbose: bool = False, overwrite: bool = False):
    """Copies the contents of src to dest.

    Args:
        src: The path to the source directory, cloud or local, e.g., "gs://mx-automotive-data/kitti/"
        dest: The path to the destination directory, cloud or local
        verbose: Set `True` if the output of the used gsutil command should be printed
        overwrite: Set `True` to overwrite existing files
    """

    # ensure a single trailing slash in src path
    src = os.path.join(src, "")

    verbosity = "" if verbose else "-q"
    overwrite = "" if overwrite else "-n"
    cmd = f"gsutil -m {verbosity} cp -r {overwrite} {src}* {dest}"
    subprocess.check_call(cmd, shell=True)


def copy_directory_to_local(cloud_path: str, local_path: str, verbose: bool = False):
    """Copies a directory from google cloud storage to a local path.

    Args:
        cloud_path: The cloud path to the directory, e.g., "gs://mx-automotive-data/kitti/". If file
                    is already local, does nothing.
        prefix: Prefix for local path
        verbose: Set `True` if the output of the used gsutil command should be printed

    Returns:
        The path to the local directory where the files got copied to.
        E.g., "/merantix_core/data/mx-automotive-data/kitti/"
    """
    if not is_gcs_uri(cloud_path):
        logging.warning(f"Called copy_directory_to_local() on local dir {cloud_path}. Taking no additional action")
        return cloud_path

    create_directory(local_path)
    logging.info(f"Copying {cloud_path} to {local_path}.")
    copy_dir(cloud_path, local_path, verbose=verbose)
    return local_path
