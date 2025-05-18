import argparse
import glob
import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm
import pyiqa

from data.base import SingleImageDataset, LROnlyWithPseudoHRDataset
from data.downsampled_dataset import DownsampledDataset
from data.paired_dataset import PairedImageDataset
from metrics import calc_fid, batched_iqa
from metrics.psnr_ssim import calc_psnr_ssim
from utils import get_obj_from_str
from utils.io_utils import mkdir

seed_everything(123)

iqa_lpips = pyiqa.create_metric("lpips").to("cuda:0")

def load_model(config, ckpt_path, strict=False):
    LightningModel = get_obj_from_str(config["target"])
    params = config["params"]
    model = LightningModel.load_from_checkpoint(ckpt_path, strict=strict, **params)
    return model


def tensor2uint8(tensors):
    to_list_flag = 0
    if not isinstance(tensors, list):
        to_list_flag = 1
        tensors = [tensors]

    def quantize(img):
        return img.mul(255).clamp(0, 255).round()

    array = [
        np.transpose(quantize(tensor).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        for tensor in tensors
    ]
    if to_list_flag:
        array = array[0]

    return array


def make_dataloader(type, datsetName, scale):
    mean, std = [0, 0, 0], [1, 1, 1]
    if type == "LRHR_paired":
        dataset = PairedImageDataset(
            lr_path="load/benchmark/{0}/LR".format(
                datsetName, scale
            ),
            hr_path="load/benchmark/{0}/HR".format(datsetName),
            scale=scale,
            is_train=False,
            cache="bin",
            mean=mean,
            std=std,
            return_img_name=True,
        )
    elif type == "LR_only":
        dataset = LROnlyWithPseudoHRDataset(
            img_path="load/benchmark/{0}".format(datsetName),
            scale=float(scale),
            cache="bin",
            mean=mean,
            std=std,
            return_img_name=True,
        )
    elif type == "HR_only":
        dataset = SingleImageDataset(
            img_path="load/benchmark/{0}/HR".format(datsetName),
            cache="bin",
            mean=mean,
            std=std,
            return_img_name=True,
            regx=".*.jpg",
        )
    elif type == "HR_downsampled":
        dataset = DownsampledDataset(
            datapath="load/benchmark/{0}/HR".format(datsetName),
            scale=float(scale),
            is_train=False,
            cache="bin",
            mean=mean,
            std=std,
            return_img_name=True,
        )
    else:
        raise "Unknown dataset type"
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


@torch.no_grad()
def test_for_one_loader(model, loader, rslt_path):
    rslts = []
    for batch in tqdm(loader, total=len(loader)):
        rslt_batch = model.test_step(batch, 0)
        for idx, file_name in enumerate(batch["file_name"]):
            _rslt = dict()
            for key, value in rslt_batch.items():
                if isinstance(value, torch.Tensor):
                    _rslt[key] = value[idx]
                    _rslt[key + "_np"] = tensor2uint8(value[idx])

                    # save to path
                    if key in ["sr", "hr_rec"]:
                        mkdir(os.path.join(rslt_path, key))
                        file_path = os.path.join(rslt_path, key, file_name)
                        plt.imsave(file_path, _rslt[key + "_np"])
                else:
                    _rslt[key] = value
            rslts.append(_rslt)
    return rslts


def calc_metrics(
    rslts,
    rslt_key,
    gt_key,
    crop_border,  # crop_border: math.ceil(scale)
    rslt_path,
    gt_path,
    test_Y=False,
    is_calc_fid=True,
    device="cuda",
):
    psnrs, ssims, run_times, losses = [], [], [], []

    for _rslt in rslts:
        rslt = _rslt[rslt_key]
        gt = _rslt[gt_key]
        rslt_np, gt_np = tensor2uint8([rslt, gt])
        psnr, ssim = calc_psnr_ssim(
            rslt_np, gt_np, crop_border=crop_border, test_Y=test_Y
        )
        psnrs.append(psnr)
        ssims.append(ssim)
        if "runtime" in _rslt.keys():
            run_times.append(_rslt["runtime"])
    mean_psnr = np.array(psnrs).mean()
    mean_ssim = np.array(ssims).mean()
    mean_runtime = np.array(run_times).mean()

    print("- PSNR: {:.4f}".format(mean_psnr))
    print("- SSIM: {:.4f}".format(mean_ssim))
    print("- Runtime: {:.4f}".format(mean_runtime))

    rslt_tensor = torch.stack([x[rslt_key] for x in rslts], dim=0)
    hr_tensor = torch.stack([x[gt_key] for x in rslts],dim=0)

    if is_calc_fid:
        _rslt_path = os.path.join(rslt_path, rslt_key)
        # gt_path = os.path.join(rslt_path, gt_key)
        paths = [_rslt_path, gt_path]
        fid_score = calc_fid(paths, device=device)
        print("- FID : {:.4f}".format(fid_score))

        lpips = (
            batched_iqa(
                iqa_lpips, rslt_tensor, hr_tensor, desc="calculating LPIPS: "
            ).mean().item()
            )
        print("- LPIPS : {:.4f}".format(lpips))
    print("=" * 42)


def test(opt):
    # setup device
    device = (
        torch.device("cuda", index=int(opt.gpu)) if opt.gpu else torch.device("cpu")
    )

    # setup datasets
    test_datasets = [_ for _ in opt.datasets.split(",")]
    logdir = os.path.dirname(os.path.dirname(opt.checkpoint))
    ckpt_path = opt.checkpoint

    # read config
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    # create model
    model = load_model(config.model, ckpt_path, strict=False)
    model.to(device)
    model.on_fit_start()
    model.eval()

    opt.scales = [float(s) for s in opt.scales.split(",")]

    for dataset_name in test_datasets:
        for scale in opt.scales:
            print("=" * 42)
            print(f"== test for dataset: {dataset_name}, scale: {scale}")
            print("=" * 42)
            loader = make_dataloader(opt.datatype, dataset_name, scale)
            # config result path
            rslt_path = os.path.join(
                logdir,
                "results",
                dataset_name,
                "x" + str(scale),
            )
            rslts = test_for_one_loader(model, loader, rslt_path)
            if opt.first_stage:
                rslt_key, gt_key = "reconstructions", "inputs"
                # rslt_key, gt_key = "xrec", "hr"
            else:
                rslt_key, gt_key = "sr", "hr"

            if opt.datatype == "LRHR_paired":
                gt_path = loader.dataset.hr_path
            else:
                gt_path = loader.dataset.dataset.datapath

            calc_metrics(
                rslts,
                rslt_key,
                gt_key,
                crop_border=math.ceil(scale),
                rslt_path=rslt_path,
                gt_path=gt_path,
                test_Y=False,
                is_calc_fid=True,
                device=device,
            )
            calc_metrics(
                rslts,
                "sr_kd",
                gt_key,
                crop_border=math.ceil(scale),
                rslt_path=rslt_path,
                gt_path=gt_path,
                test_Y=False,
                is_calc_fid=True,
                device=device,
            )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint index")
    parser.add_argument(
        "-g",
        "--gpu",
        default="0",
        type=str,
        help="index of GPU to enable",
    )
    parser.add_argument(
        "--datasets", type=str, default="AID_tiny", help="dataset names"
    )
    parser.add_argument("--scales", default="4", type=str, help="scale factors")
    parser.add_argument(
        "--datatype",
        default="HR_downsampled",
        type=str,
        help="dataset type, options: (HR_only, LR_only, LRHR_paired)",
    )
    parser.add_argument("--first_stage", action="store_true")
    return parser


if __name__ == "__main__":
    test_parser = get_parser()
    opt = test_parser.parse_args()
    test(opt)
