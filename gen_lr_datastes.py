from litsr.data import SingleImageDataset
from easydict import EasyDict
from utils.srmd_degrade import SRMDPreprocessing
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import path as osp
from litsr.utils import mkdir

datase_name = "sr-geo-122"
config = EasyDict(rgb_range=1)

dataset = SingleImageDataset(
    img_path="load/benchmark/{0}/HR".format(datase_name),
    rgb_range=config.rgb_range,
    cache="bin",
    return_img_name=True,
)

lambda_1 = 0
lambda_2 = 0
noise = 0

degrade = SRMDPreprocessing(
    scale=4,
    kernel_size=21,
    blur_type="iso_gaussian",
    sig=0,
    # sig_min=0.2,
    # sig_max=4.0,
    # lambda_1=lambda_1,
    # lambda_2=lambda_2,
    noise=noise,
)

loader = DataLoader(dataset, batch_size=1)

for batch in tqdm(loader):
    hr, name = batch
    hr.mul_(255.0)

    lr, _ = degrade(hr.unsqueeze(1), random=False)
    lr.mul_(1 / 255.0)

    lr_numpy = lr.squeeze().numpy().transpose(1, 2, 0)
    path = "logs/lr_aniso/lam1_{:.1f}_lam2_{:.1f}_noise_{:.1f}".format(
        lambda_1, lambda_2, noise
    )
    mkdir(path)
    plt.imsave(osp.join(path, name[0]), lr_numpy)
