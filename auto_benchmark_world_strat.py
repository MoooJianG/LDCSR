import json, os
from os import path as osp
import numpy as np
import pandas as pd
import pyiqa
import torch
from metrics import calc_fid, calc_psnr_ssim, batched_iqa, calc_psnr_only
from data.transforms import uint2single
from utils.io_utils import load_file_list, read_images_parallel as read_images
from tqdm import tqdm
from natsort import natsorted
from datetime import datetime

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
iqa_psnr = pyiqa.create_metric("psnr").to(device)
iqa_lpips = pyiqa.create_metric("lpips").to(device)
iqa_niqe = pyiqa.create_metric("niqe").to(device)
iqa_brisque = pyiqa.create_metric("brisque").to(device)
iqa_clipiqa = pyiqa.create_metric("clipiqa").to(device)
iqa_piqe = pyiqa.create_metric("piqe").to(device)


def writeRslt(rsltPath, rslt):
    with open(osp.join(rsltPath, "rslt.txt"), "w") as f:
        f.write(json.dumps(rslt))


def readRslt(rsltPath):
    with open(osp.join(rsltPath, "rslt.txt"), "r") as f:
        rslt = json.load(f)
    return rslt

def handleDataset(dataset):
    rsltList = []
    if dataset == "WHU-RS19":
        hrPath = osp.join("load/benchmark", dataset)
    else:
        hrPath = osp.join("logs/_results", dataset, "HR")
    print("loading HR images...")
    hrs = read_images(load_file_list(hrPath, ".*.png"))

    for methodName in methodList:
        if not osp.exists(osp.join(root, dataset, methodName)):
            continue
        degradationList = natsorted(os.listdir(osp.join(root, dataset, methodName)))
        for degradName in degradationList:
            print("==" * 20)
            print("{0} | {1} | {2}".format(dataset, methodName, degradName))
            rsltPath = osp.join(root, dataset, methodName, degradName)
            if not osp.exists(rsltPath):
                continue
            elif osp.exists(osp.join(rsltPath, "rslt.txt")) and (not force_recalc):
                rslt = readRslt(rsltPath)
                rslt["methodName"] = methodName
                rslt["degradName"] = degradName
            else:
                print(f"loading SR results from {rsltPath}...")
                rsltPathList = load_file_list(rsltPath, ".*.png")
                rslts = read_images(rsltPathList)

                print("calculating FID ...")
                print(rsltPath)
                fid = calc_fid([hrPath, rsltPath])

                hr_tensor = torch.from_numpy(
                    uint2single(np.array(hrs).transpose(0, 3, 1, 2))
                ).to(device)
                rslt_tensor = torch.from_numpy(
                    uint2single(np.array(rslts).transpose(0, 3, 1, 2))
                ).to(device)

                psnr = (
                    batched_iqa(
                        iqa_psnr, rslt_tensor, hr_tensor, desc="calculating PSNR: "
                    )
                    .mean()
                    .item()
                )
                lpips = (
                    batched_iqa(
                        iqa_lpips, rslt_tensor, hr_tensor, desc="calculating LPIPS: "
                    )
                    .mean()
                    .item()
                )
                niqe = (
                    batched_iqa(iqa_niqe, rslt_tensor, desc="calculating NIQE: ")
                    .mean()
                    .item()
                )
                brisque = (
                    batched_iqa(iqa_brisque, rslt_tensor, desc="calculating BRISQUE: ")
                    .mean()
                    .item()
                )
                clipiqa = (
                    batched_iqa(iqa_clipiqa, rslt_tensor, desc="calculating CLIPIQA: ")
                    .mean()
                    .item()
                )
                piqe = (
                    batched_iqa(iqa_piqe, rslt_tensor, desc="calculating PIQE: ")
                    .mean()
                    .item()
                )

                rslt = {
                    "methodName": methodName,
                    "degradName": degradName,
                    "psnr": psnr,  # np.array(psnrList).mean(),
                    "ssim": None,  # np.array(ssimList).mean(),
                    "fid": fid,
                    "lpips": lpips,
                    "niqe": niqe,
                    "brisque": brisque,
                    "clipiqa": clipiqa,
                    "piqe": piqe,
                    "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            rslt["dataset"] = dataset
            rsltList.append(rslt)
            writeRslt(rsltPath, rslt)
            print(rslt)
    return rsltList


def format_result(df):
    datasets = ["WorldStrat"]
    metrics = ["PSNR", "FID", "LPIPS", "NIQE", "BRISQUE"]
    methods = [
        "Bicubic",
        "DASR-WorldStrat",
        "RealESRGAN-WorldStrat",
        "SR3-WorldStrat",
        "BlindSRSNF-WorldStrat",
        "LDCSR_ours",
        "LDCSR_rec"
    ]

    def get_val(dataset, method, scale, metric):
        for idx, row in df.iterrows():
            if (
                row["methodName"] == method
                and row["dataset"] == dataset
                and row["degradName"] == scale
            ):
                return row[metric.lower()]

    results = []
    for dataset in datasets:
        for method in methods:
            rslt = {
                "Datasset": dataset,
                "Methods": method,
            }
            for metric in metrics:
                rslt[metric] = get_val(dataset, method, f"x4.0", metric)
            results.append(rslt)

    formatted_df = pd.DataFrame(results)
    return formatted_df


if __name__ == "__main__":
    root = "logs/_results"
    datasets = ["WorldStrat"]
    methodList = [
        "Bicubic",
        "DASR-WorldStrat",
        "RealESRGAN-WorldStrat",
        "SR3-WorldStrat",
        "BlindSRSNF-WorldStrat",
        "LDCSR_ours",
        "LDCSR_rec"
    ]

    force_recalc = True
    rslt_list = []
    for dataset in datasets:
        rslt_list.extend(handleDataset(dataset))
    df = pd.DataFrame(rslt_list)
    df.to_excel(
        osp.join(root, "result_all_real.xlsx"),
        index=False,
    )

    formatted_df1 = format_result(df)
    formatted_df1.to_excel(
        osp.join(root, "result_world_strat.xlsx"),
        index=False,
    )