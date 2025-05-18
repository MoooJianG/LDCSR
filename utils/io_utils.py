import os
import re
import shutil
from multiprocessing import Pool
from os import path as osp
from typing import Iterable, Optional

import cv2
import numpy as np
import yaml
from data.transforms import uint2single
from tqdm import tqdm
from natsort import natsorted


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir_clean(path):
    """Create a directory that is guaranteed to be empty"""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            os.makedirs(path)
        except:
            print("warning! Cannot remove {0}".format(path))
    else:
        os.makedirs(path)


def mkdir_clean(path):
    """Create a directory that is guaranteed to be empty"""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            os.makedirs(path)
        except:
            print("warning! Cannot remove {0}".format(path))
    else:
        os.makedirs(path)


def load_file_list(path: str, regx: str) -> list:
    return_list = []
    for path, _, file_lst in os.walk(path):
        for f in file_lst:
            if re.match(regx, f):
                fullname = os.path.join(path, f)
                return_list.append(fullname)
    return natsorted(return_list)


def read_image(
    path: str, mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:
    """Read image to a 3 dimentional numpy by OpenCV"""
    img = cv2.imread(path)
    assert mode in ("RGB", "BGR")
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_float:
        img = uint2single(img)
    return img


def read_images(
    path_list: Iterable[str], mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:
    """Read images to a 4 dimentional numpy array by OpenCV"""
    rslt = []
    for path in tqdm(path_list):
        rslt.append(read_image(path, mode, to_float))
    return np.array(rslt)


def read_images_parallel(
    path_list: Iterable[str], mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:
    import concurrent.futures

    """Read images to a 4-dimensional numpy array by OpenCV using parallel processing"""

    # 包装一个辅助函数，将 `read_image` 的参数传入
    def read_image_wrapper(path: str) -> np.ndarray:
        return read_image(path, mode, to_float)

    # 使用线程池并行读取图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 这里使用 tqdm 显示进度条
        results = list(
            tqdm(executor.map(read_image_wrapper, path_list), total=len(path_list))
        )

    return np.array(results)
