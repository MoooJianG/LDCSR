#!/usr/bin/env python
import os
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image


def resize_pillow(img, scale=None, size=None, mode="bicubic"):
    """Resize image using Pillow with specified scale or size and interpolation method."""
    assert scale or size
    if type(img) == np.ndarray:
        h, w = img.shape[0:2]
    else:
        h, w = img.size[0:2]
    if not size:
        size = (int(h * scale), int(w * scale))

    size = (size[1], size[0])  # PIL uses (width, height)
    to_numpy_flag = 0
    if type(img) == np.ndarray:
        to_numpy_flag = 1
        img = Image.fromarray(img)

    interpolation = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }[mode]

    img = img.resize(size, resample=interpolation)
    if to_numpy_flag:
        img = np.array(img)

    return img


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset splits and generate LR images')
    parser.add_argument('--split_file', type=str, required=True, help='Path to the split pickle file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path for the output split dataset')
    parser.add_argument('--scale', type=float, default=4.0, help='Downscale factor (default: 4.0)')
    parser.add_argument('--mode', type=str, default='bicubic', choices=['bicubic', 'bilinear', 'nearest'],
                        help='Interpolation mode (default: bicubic)')
    args = parser.parse_args()

    print(f"Loading split file from: {args.split_file}")
    with open(args.split_file, 'rb') as f:
        split = pickle.load(f)

    for split_name, split_list in split.items():
        split_path = os.path.join(args.output_path, split_name)
        hr_folder = os.path.join(split_path, "HR")
        lr_folder = os.path.join(split_path, "LR")

        os.makedirs(hr_folder, exist_ok=True)
        os.makedirs(lr_folder, exist_ok=True)

        print(f"Processing {split_name} split with {len(split_list)} images...")
        for filename in tqdm(split_list, desc=f"Preparing {split_name}"):
            file_path = os.path.join(args.data_path, filename)
            
            try:
                hr = cv2.imread(file_path)
                if hr is None:
                    print(f"Warning: Could not read {file_path}")
                    continue
                    
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

                lr = resize_pillow(hr, 1/args.scale, mode=args.mode)

                output_filename = os.path.splitext(os.path.basename(filename))[0] + ".png"
                
                lr_path = os.path.join(lr_folder, output_filename)
                hr_path = os.path.join(hr_folder, output_filename)

                cv2.imwrite(lr_path, lr)
                cv2.imwrite(hr_path, hr)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()