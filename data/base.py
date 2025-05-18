import os
import os.path as osp
import pickle
import re
from abc import abstractmethod

import cv2
from scipy import io as scio
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from natsort import natsorted

from data import transforms


def load_mat(path):
    mat = scio.loadmat(path, verify_compressed_data_integrity=False)
    return mat["t"]


def load_file_list(path: str, regx: str) -> list:
    return_list = []
    for path, _, file_lst in os.walk(path):
        for f in file_lst:
            if re.match(regx, f):
                fullname = os.path.join(path, f)
                return_list.append(fullname)
    return natsorted(return_list)


def load_file_list_from_split(path: str, split_file_path: str, split: str) -> list:
    with open(split_file_path, "rb") as f_:
        split_file = pickle.load(f_)
    return_list = split_file[split]
    return_list = [os.path.join(path, f) for f in return_list]
    return natsorted(return_list)


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class ImageFolder(Dataset):
    """
    Construct a dataset from a folder
    """

    def __init__(
        self,
        datapath,
        repeat=1,
        cache=None,
        first_k=None,
        data_length=None,
        regx=".*.png",
        split_file_path=None,
        split=None,
    ):
        self.datapath = datapath
        self.repeat = repeat or 1
        self.cache = cache
        self.data_length = data_length
        self.regx = regx
        self.split_file_path = split_file_path
        self.split = split

        if cache == "bin":
            [self.filenames, self.files] = self.load_data_from_bin(first_k)
        elif cache == "memory":
            [self.filenames, self.files] = self.load_data_from_memory(first_k)
        else:
            [self.filenames, self.files] = self.load_data()

        if first_k:
            self.filenames = self.filenames[:first_k]
            self.files = self.files[:first_k]

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "memory":
            return x
        elif self.cache == "bin":
            with open(x, "rb") as f:
                x = pickle.load(f)
            return x
        else:
            return self.read_image(x)

    def load_file_list(self):
        if self.split_file_path:
            return load_file_list_from_split(
                self.datapath, self.split_file_path, self.split
            )
        else:
            return load_file_list(self.datapath, self.regx)

    def read_image(self, path, mode="RGB"):
        """Read image to a 3 dimentional numpy by OpenCV"""
        img = cv2.imread(path)
        assert mode in ("RGB", "BGR")
        if mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_data_from_bin(self, first_k=None):
        file_names = self.load_file_list()

        if first_k:
            file_names = file_names[:first_k]

        bin_root = osp.join(
            osp.dirname(self.datapath), "_bin_" + osp.basename(self.datapath)
        )
        if not osp.exists(bin_root):
            os.mkdir(bin_root)
            print("mkdir", bin_root)
        files = []

        for f in file_names:
            bin_file = osp.join(bin_root, osp.basename(f).split(".")[0] + ".pkl")
            if not osp.exists(bin_file):
                with open(bin_file, "wb") as bin_f:
                    pickle.dump(
                        self.read_image(osp.join(f)),
                        bin_f,
                    )
                print("dump", bin_file)
            files.append(bin_file)
        return file_names, files

    def load_data_from_memory(self, first_k=None, from_pickle=True):
        if from_pickle:
            file_names, filenames = self.load_data_from_bin(first_k)
            files = []
            pbar = tqdm(filenames)
            pbar.set_description("load data (from pickle)")
            for f in pbar:
                with open(f, "rb") as f_:
                    files.append(pickle.load(f_))

            return file_names, files

        file_names = self.load_file_list()
        file_names = [osp.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        files = []
        pbar = tqdm(file_names)
        pbar.set_description("load data")
        for f in pbar:
            files.append(self.read_image(osp.join(self.datapath, f)))

        return file_names, files

    def load_data(self):
        files = self.load_file_list()
        file_names = [osp.basename(n) for n in files]
        return file_names, files

    def __len__(self):
        if self.data_length:
            return self.data_length
        return int(len(self.files) * self.repeat)


class SingleImageDataset(Dataset):
    def __init__(
        self,
        img_path,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        mean=None,
        std=None,
        return_img_name=False,
        regx=".*.(png|jpg|jpeg)",
    ):
        self.dataset = ImageFolder(
            img_path, repeat=repeat, cache=cache, first_k=first_k, regx=regx
        )
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        img = self.dataset[idx]

        img = transforms.uint2single(img)
        img = transforms.single2tensor(img) * self.rgb_range

        if self.mean and self.std:
            transforms.normalize(img, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            file_name = osp.split(file_name)[1]
            example = {"lr": img, "file_name": file_name}
            return example
        else:
            return {"lr": img}

    def __len__(self):
        return len(self.dataset)

class LROnlyWithPseudoHRDataset(Dataset):
    def __init__(
        self,
        img_path,
        scale,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        mean=None,
        std=None,
        return_img_name=False,
        regx=".*.(png|jpg|jpeg)",
    ):
        self.dataset = ImageFolder(
            img_path, repeat=repeat, cache=cache, first_k=first_k, regx=regx
        )
        self.scale = scale
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        scale = self.scale
        lr = self.dataset[idx]
        [h, w] = lr.shape[0:2]
        pseudo_hr = transforms.resize_pillow(lr, size=(int(h*scale), int(w*scale)))
        lr = transforms.uint2single(lr)
        pseudo_hr = transforms.uint2single(pseudo_hr)
        
        lr = transforms.single2tensor(lr) * self.rgb_range
        pseudo_hr = transforms.single2tensor(pseudo_hr) * self.rgb_range

        if self.mean and self.std:
            transforms.normalize(lr, self.mean, self.std, inplace=True)
            transforms.normalize(pseudo_hr, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            file_name = osp.split(file_name)[1]
            example = {"lr": lr, "hr": pseudo_hr, "file_name": file_name}
            return example
        else:
            return {"lr": lr, "hr": pseudo_hr}

    def __len__(self):
        return len(self.dataset)

class PairedImageFolder(Dataset):
    def __init__(
        self, path1, path2, repeat=1, cache=None, first_k=None, data_length=None
    ):
        self.dataset_1 = ImageFolder(
            path1, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )
        self.dataset_2 = ImageFolder(
            path2, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )

        self.filenames = self.dataset_2.filenames

        assert len(self.dataset_1) == len(self.dataset_2)

    def __getitem__(self, i):
        return tuple([self.dataset_1[i], self.dataset_2[i]])

    def __len__(self):
        return len(self.dataset_1)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
