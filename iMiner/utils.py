"""
Author: Eric Wang, Jerry Li, Oufan Zhang
Date: 01/23/2023

Codes of iMiner util functions
"""
import random
import time
import datetime
import sys, os
from io import StringIO
from pathlib import Path
import contextlib
from typing import Optional

import pandas as pd
import numpy as np
import subprocess

from logging import Logger


@contextlib.contextmanager
def timer(name: Optional[str] = None, logger: Optional[Logger] = None):
    """
    Logger timer
    """
    start = time.time()
    yield
    end = time.time()
    msg = f"{name} finished. " if name else ""
    msg += f"Time Elapsed: {end - start:.3f} seconds"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def file_abspath(path: os.PathLike) -> Path:
    """
    Check whether a file path is exist and return its absoulute path
    """
    abs_path = Path(path).resolve()
    if not abs_path.is_file():
        raise FileNotFoundError(f'{abs_path} not exist')
    return abs_path


def dir_abspath(path: os.PathLike) -> Path:
    """
    Check whether a directory path is exist and return its absoulute path
    """
    abs_path = Path(path).resolve()
    if not abs_path.is_dir():
        raise FileNotFoundError(f'{abs_path} not exist')
    return abs_path


def dist_mat(crd1: np.ndarray, crd2: np.ndarray) -> np.ndarray:
    '''
    Calculate distance matrix between two coordinates sets

    Parameters
    ----------
    crd1: numpy.ndarray
        The first coordinate set with dim N1 x 3
    crd2: numpy.ndarray
        The second coordinate set with dim N2 x 3
        
    Return
    ------
    dist_mat: numpy.ndarray
        Distance matrix with dim N1 x N2
    '''
    
    expand_crd1 = np.expand_dims(crd1, 1).repeat(crd2.shape[0], axis=1)
    expand_crd2 = np.broadcast_to(crd2, (crd1.shape[0], crd2.shape[0], 3))
    dist_mat = np.linalg.norm(expand_crd1 - expand_crd2, ord=2, axis=-1)
    return dist_mat


def hash_str(string) -> str:
    '''
    Generate a unique hash code for a string
    '''
    hashcode = hex(hash(string) % ((sys.maxsize + 1) * 2)).replace("0x", "")
    return hashcode

def random_id():
    '''
    Generate a random ID
    '''
    return str(random.randint(0,65536))

def timestamp(hashed=False) -> str:
    '''
    Generate timestamp like 2019-04-17_11:27:07 or a hashed timestamp
    '''
    now = datetime.datetime.now()
    value = now.strftime("%Y-%m-%d_%H:%M:%S")
    if hashed:
        value = hash_str(value)
    return value


def box_from_center_and_size(center, size):
    '''
    Generate a box definition (xmin, ymin, zmin, xmax, ymax, zmax)
    from center (x, y, z) and size (size_x, size_y, size_z)
    '''
    xmin = center[0] - size[0] / 2
    ymin = center[1] - size[1] / 2
    zmin = center[2] - size[2] / 2
    xmax = center[0] + size[0] / 2
    ymax = center[1] + size[1] / 2
    zmax = center[2] + size[2] / 2
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode("utf-8").replace("MiB", "")),
        names=['memory.used', 'memory.free'],
        skiprows=1
    )
    gpu_df["avail"] = gpu_df["memory.free"] / (gpu_df["memory.used"] + gpu_df["memory.free"])
    idx = gpu_df['avail'].idxmax()
    if gpu_df.loc[idx, "avail"] < 0.1:
        idx = None
    return str(idx)

def check_dict_identity(dict1, dict2):
    '''
    Check whether two dictionaries have the same keys and values
    '''
    if dict1.keys() != dict2.keys():
        return False
    for k in dict1.keys():
        if str(dict1[k]) != str(dict2[k]):
            return False
    return True