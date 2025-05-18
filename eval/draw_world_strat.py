import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib.transforms as transforms
import pyiqa
from data.transforms import uint2single
import cv2

# Settings
rootpath = r"logs/_results"
alphabet = "abcdefghijklmnopqrstuvwxyz"
isPrintAlphabet = False
fontsize = 14
rows = 2
cols = 3
savepath = r"logs/_results_paper"

offset = np.array([[0.5, 0.5]])

# Set dataset (selected examples, please uncomment as needed)
# =====
setName = "WorldStrat"
degrade = "x4.0"  # scale factor
img_name = "Amnesty POI-1-2-1.png"
zoom_p = np.array([[150, 150]])
# img_name = "Landcover-772992.png"
# zoom_p = np.array([[360, 10]])
zoom_size = [120, 120]

method_name = [
    # ["LR", "LR"],
    # ["hr", "GT"],
    ["LR", "LR"],
    ["DASR-WorldStrat", "DASR"],
    ["RealESRGAN-WorldStrat", "RealESRGAN"],
    ["SR3-WorldStrat", "SR3"],
    ["BlindSRSNF-WorldStrat", "BlindSRSNF"],
    ["LDCSR_rec", "OURS"],
]


def addborder(im, border_color, border_size):
    return ImageOps.expand(im, border=border_size, fill=tuple(border_color))


def draw_box_zoom(img, zoom_p, zoom_size, offset):
    rgb = np.array([[0, 255, 0], [0, 0, 255]])  # Define colors for the borders
    x_length, y_length = zoom_size
    width = 4
    zoom_border = 6
    ori_size = img.size

    draw = ImageDraw.Draw(img)
    for z_idx, (z_x, z_y) in enumerate(zoom_p):
        rgb_temp = tuple(rgb[z_idx % len(rgb)])
        # Draw rectangle borders
        draw.rectangle(
            [z_y, z_x, z_y + y_length, z_x + x_length], outline=rgb_temp, width=width
        )

    # Prepare image for zoom views with borders
    rimg = Image.new("RGB", (ori_size[0] + int(ori_size[1] * 0), ori_size[1]), "white")
    rimg.paste(img, (0, 0))

    for z_idx, (z_x, z_y) in enumerate(zoom_p):
        rgb_temp = tuple(rgb[z_idx % len(rgb)])
        # Crop and resize the zoom area
        im_zoom = img.crop((z_y, z_x, z_y + y_length, z_x + x_length))
        im_zoom = im_zoom.resize(
            (int(ori_size[0] / 2), int(ori_size[1] / 2)), Image.LANCZOS
        )
        # im_zoom = addborder(im_zoom, rgb_temp, zoom_border)
        offset_temp = np.round(np.multiply(offset[z_idx], ori_size)).astype(int)
        rimg.paste(im_zoom, (offset_temp[1], offset_temp[0]))
    return rimg


# Initialize NIQE metric
niqe_metric = pyiqa.create_metric("niqe")
brisque_metric = cv2.quality.QualityBRISQUE_create(
    "eval/brisque_model_live.yml", "eval/brisque_range_live.yml"
)

# Main processing
hr_path = os.path.join(rootpath, setName, "HR")
rslt_img_scale = 150 / zoom_size[0]
zoom_image_dir = os.path.join(savepath, "zoom_image", f"{setName}_{degrade}")

if not os.path.exists(zoom_image_dir):
    os.makedirs(zoom_image_dir)

im_hr = Image.open(os.path.join(hr_path, img_name))
ori_size = im_hr.size
im_hr = draw_box_zoom(im_hr, zoom_p, zoom_size, offset)
im_hr.save(os.path.join(zoom_image_dir, "hr.png"))

niqe_scores = {}
brisque_scores = {}

for method in method_name:
    if method[0] == "hr":
        path = os.path.join(rootpath, setName, "HR")
    else:
        path = os.path.join(rootpath, setName, method[0], degrade)
    if not os.path.exists(path):
        im_zoom = Image.new("RGB", ori_size, "gray")
        im_zoom = draw_box_zoom(im_zoom, zoom_p, zoom_size, offset)
        im_zoom.save(os.path.join(zoom_image_dir, f"{method[0]}.png"))
        continue
    images = os.listdir(
        path
    )  # Replace this with a function to get image files if necessary
    flag = False

    for img_file in images:
        if img_file == img_name:
            flag = True
            img_path = os.path.join(path, img_file)
            im = Image.open(img_path)
            if im.mode == "RGBA":
                im = im.convert("RGB")
            if method[0] == "LR":
                im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)

            # Calculate NIQE score (before zoom box for original image quality)
            im_torch = torch.from_numpy(
                uint2single(np.array(im).transpose(2, 0, 1))
            ).unsqueeze(0)
            niqe_score = niqe_metric(im_torch).item()
            brisque_score = brisque_metric.compute(np.array(im))[0]
            niqe_scores[method[0]] = niqe_score
            brisque_scores[method[0]] = brisque_score

            im_zoom = draw_box_zoom(im, zoom_p, zoom_size, offset)
            im_zoom.save(os.path.join(zoom_image_dir, f"{method[0]}.png"))

    if not flag:
        im_zoom = Image.new("RGB", ori_size, "gray")
        im_zoom = draw_box_zoom(im_zoom, zoom_p, zoom_size, offset)
        im_zoom.save(os.path.join(zoom_image_dir, f"{method[0]}.png"))
        niqe_scores[method[0]] = float("inf")
        brisque_scores[method[0]] = float("inf")

# Plotting the final figure
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7.5, 5.8))
plt.subplots_adjust(wspace=0.05, hspace=-0.25)
plt.subplots_adjust(left=0, bottom=-0.05, top=1.12, right=1)
for i, method in enumerate(method_name):
    ax = axes[i // cols, i % cols]
    img_path = os.path.join(zoom_image_dir, f"{method[0]}.png")
    im = Image.open(img_path)
    ax.imshow(
        im.resize((int(im.width * rslt_img_scale), int(im.height * rslt_img_scale)))
    )
    # Set title with NIQE score
    niqe_score = niqe_scores.get(method[0], float("inf"))
    brisque_score = brisque_scores.get(method[0], float("inf"))
    if niqe_score == float("inf"):
        niqe_text = "N/A"
    else:
        niqe_text = f"{brisque_score:.2f}/{niqe_score:.2f}"
    if isPrintAlphabet:
        title_text = f"{alphabet[i]}) {method[1]} ({niqe_text})"
    else:
        title_text = f"{method[1]} ({niqe_text})"
    ax.text(
        0.5,
        -0.05,
        title_text,
        fontsize=fontsize,
        ha="center",
        va="top",
        transform=ax.transAxes,
    )
    ax.axis("off")

plt.savefig(
    os.path.join(savepath, f'fs_{setName}_{degrade.replace(".", "")}.png'),
    dpi=200,
)
shutil.rmtree(zoom_image_dir)
# plt.savefig(
#     os.path.join(savepath, f'mainrslt_{setName}_{degrade.replace(".", "")}.eps'),
#     format="eps",
# )
