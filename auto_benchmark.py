import json, os
from os import path as osp
import concurrent.futures
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


def writeRslt(rsltPath, rslt):
    with open(osp.join(rsltPath, "rslt.txt"), "w") as f:
        f.write(json.dumps(rslt))


def readRslt(rsltPath):
    with open(osp.join(rsltPath, "rslt.txt"), "r") as f:
        rslt = json.load(f)
    return rslt


# def calc_psnr_ssim_wrapper(hr_sr_pair):
#     hr, sr = hr_sr_pair
#     return calc_psnr_only(hr, sr, crop_border=4, test_Y=False), 0


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

                # print("calculating PSNR & SSIM ...")
                # psnrList = [0]
                # ssimList = [0]
                # hr_sr_pairs = list(zip(hrs, rslts))
                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     results = list(
                #         tqdm(
                #             executor.map(calc_psnr_ssim_wrapper, hr_sr_pairs),
                #             total=len(hr_sr_pairs),
                #         )
                #     )
                # for psnr, ssim in results:
                #     psnrList.append(psnr)
                #     ssimList.append(ssim)

                print("calculating FID ...")
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

                rslt = {
                    "methodName": methodName,
                    "degradName": degradName,
                    "psnr": psnr,  # np.array(psnrList).mean(),
                    "ssim": None,  # np.array(ssimList).mean(),
                    "fid": fid,
                    "lpips": lpips,
                    "niqe": niqe,
                    "brisque": brisque,
                    "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            rslt["dataset"] = dataset
            rsltList.append(rslt)
            writeRslt(rsltPath, rslt)
            print(rslt)
    # df = pd.DataFrame(rsltList)
    # df.to_excel(
    #     osp.join(root, dataset + "_" + rslt_file_name),
    #     index=False,
    # )
    return rsltList


def format_result(df):
    metrics = ["PSNR", "LPIPS", "FID"]
    methods = [
        "Bicubic",
        "HAT-L",
        "SR3",
        "ESRGAN",
        "EDiffSR",
        "SPSR",
        "BlindSRSNF",
        "TTST",
        "LIIF",
        "SADN",
        "IDM",
        "CiaoSR",
        "LMF",
        "FunSR",
        "LDCSR_ours",
        # "SRLCM_e199",
        # "SRLCM_wokd",
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
    for method in methods:
        for metric in metrics:
            results.append(
                {
                    "Methods": method,
                    "Metric": metric,
                    "AID_x2": get_val("AID", method, "x2.0", metric),
                    "AID_x4": get_val("AID", method, "x4.0", metric),
                    "AID_x8": get_val("AID", method, "x8.0", metric),
                    "DOTA_x2": get_val("DOTA", method, "x2.0", metric),
                    "DOTA_x4": get_val("DOTA", method, "x4.0", metric),
                    "DOTA_x8": get_val("DOTA", method, "x8.0", metric),
                    "DIOR_x2": get_val("DIOR", method, "x2.0", metric),
                    "DIOR_x4": get_val("DIOR", method, "x4.0", metric),
                    "DIOR_x8": get_val("DIOR", method, "x8.0", metric),
                }
            )

    formatted_df = pd.DataFrame(results)
    return formatted_df


def format_result2(df):
    datasets = ["AID", "DOTA", "DIOR"]
    metrics = ["FID", "LPIPS"]
    methods = ["Bicubic", "LIIF", "SADN", "IDM", "CiaoSR", "LMF", "LDCSR_ours"]

    def get_val(dataset, method, scale, metric):
        for idx, row in df.iterrows():
            if (
                row["methodName"] == method
                and row["dataset"] == dataset
                and row["degradName"] == scale
            ):
                return row[metric.lower()]

    scales = [2.0, 2.6, 3.0, 3.4, 4.0, 6.0, 8.0, 10.0]

    results = []
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                rslt = {
                    "Datasset": dataset,
                    "Methods": method,
                    "Metric": metric,
                }
                for s in scales:
                    rslt[f"x{s:.1f}"] = get_val(dataset, method, f"x{s:.1f}", metric)
                results.append(rslt)

    formatted_df = pd.DataFrame(results)
    return formatted_df


def format_result3(df):
    datasets = ["AID", "DOTA", "DIOR"]
    metrics = ["PSNR", "SSIM", "FID", "LPIPS", "NIQE", "BRISQUE"]
    methods = [
        "Bicubic",
        "LIIF",
        "SADN",
        "HAT-L",
        "ESRGAN",
        "SR3",
        "EDiffSR",
        "BlindSRSNF",
        "IDM",
        # "SRLCM_e199",
        # "SRLCM_wokd",
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
    datasets = ["AID", "DIOR", "DOTA"]
    # datasets = ["NWPU-RESISC45"]
    methodList = [
        "Bicubic",
        "LIIF",
        "SADN",
        "HAT-L",
        "ESRGAN",
        "BlindSRSNF",
        "SPSR",
        "SR3",
        "EDiffSR",
        "IDM",
        "CiaoSR",
        "LMF",
        "TTST",
        "FunSR",
        "LDCSR_ours",
        # "LDCSR_ours-Epoch500",
        # "SRLCM",
        # "SRLCM_e199",
        # "SRLCM_wokd"
    ]

    force_recalc = False
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
        osp.join(root, "result_format1.xlsx"),
        index=False,
    )

    formatted_df2 = format_result2(df)
    formatted_df2.to_excel(
        osp.join(root, "result_format2.xlsx"),
        index=False,
    )

    formatted_df3 = format_result3(df)
    formatted_df3.to_excel(
        osp.join(root, "result_format3.xlsx"),
        index=False,
    )
