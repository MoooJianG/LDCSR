import os
import torch.utils.data as data
from data.base import PairedImageFolder
from data import transforms

class PairedImageDataset(data.Dataset):
    def __init__(
        self,
        lr_path,
        hr_path,
        scale,
        is_train,
        lr_img_sz=None,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        mean=None,
        std=None,
        return_img_name=False,
    ):
        assert not is_train ^ bool(lr_img_sz)

        self.dataset = PairedImageFolder(
            lr_path, hr_path, repeat=repeat, cache=cache, first_k=first_k
        )
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.scale = scale
        self.lr_img_sz = lr_img_sz
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        lr, hr = self.dataset[idx]
        if self.is_train:
            lr, hr = self._transform_train(lr, hr)
        else:
            lr, hr = self._transform_test(lr, hr)

        if self.mean and self.std:
            transforms.normalize(lr, self.mean, self.std, inplace=True)
            transforms.normalize(hr, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            file_name = os.path.split(file_name)[1]
            return {"lr": lr, "hr": hr, "file_name": file_name}
        else:
            return {"lr": lr, "hr": hr}

    def _transform_train(self, lr, hr):
        scale = self.scale
        lr_img_sz = self.lr_img_sz
        hr_img_sz = int(lr_img_sz * scale)

        hr = transforms.mod_crop(hr, self.scale)
        lr, hr = transforms.paired_random_crop(lr, hr, hr_img_sz, self.scale)
        lr, hr = transforms.augment([lr, hr])
        lr, hr = (
            transforms.uint2single(lr) * self.rgb_range,
            transforms.uint2single(hr) * self.rgb_range,
        )
        lr, hr = transforms.single2tensor(lr), transforms.single2tensor(hr)

        return lr, hr

    def _transform_test(self, lr, hr):
        scale = self.scale
        # hr = transforms.mod_crop(hr, scale)
        lr, hr = (
            transforms.uint2single(lr) * self.rgb_range,
            transforms.uint2single(hr) * self.rgb_range,
        )
        lr, hr = transforms.single2tensor(lr), transforms.single2tensor(hr)

        return lr, hr

    def __len__(self):
        return len(self.dataset)