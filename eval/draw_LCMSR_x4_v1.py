import os
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import shutil
import matplotlib.transforms as transforms

# Settings
rootpath = r"logs/_results"
alphabet = "abcdefghijklmnopqrstuvwxyz"
isPrintAlphabet = False
fontsize = 24
rows = 2
cols = 4
savepath = r"logs/_results_paper"

offset = np.array([[0.5, 0.5]])

# Set dataset (selected examples, please uncomment as needed)
# =====
# setName = "AID"
# degrade = "x4.0"  # scale factor
# # img_name = "stadium_38.png"
# img_name = "school_264.png"
# zoom_p = np.array([[150, 0]])
# zoom_size = [125, 125]
# =====
setName = "DOTA"
degrade = "x4.0"  # scale factor
img_name = "P1472.png"
zoom_p = np.array([[20, 320]])
zoom_size = [100, 100]
# =====
# setName = "DIOR"
# degrade = "x8.0"  # scale factor
# img_name = "14539.png"
# zoom_p = np.array([[420, 420], [400, 120]])
# zoom_size = [150, 150]

method_name = [
    ["LIIF", "LIIF"],
    ["SADN", "SADN"],
    ["HAT-L", "HAT-L"],
    ["IDM", "IDM"],
    ["SR3", "SR3"],
    ["EDiffSR", "EDiffSR"],
    ["SRLCM_e199", "LCMSR"],
    ["hr", "GT"]
]

def addborder(im, border_color, border_size):
    return ImageOps.expand(im, border=border_size, fill=tuple(border_color))


def draw_box_zoom(img, zoom_p, zoom_size, offset):
    rgb = np.array([[255, 0, 0], [0, 0, 255]])  # Define colors for the borders
    x_length, y_length = zoom_size
    width = 4
    ori_size = img.size

    draw = ImageDraw.Draw(img)
    for z_idx, (z_x, z_y) in enumerate(zoom_p):
        rgb_temp = tuple(rgb[z_idx % len(rgb)])
        # Draw rectangle borders
        draw.rectangle(
            [z_y, z_x, z_y + y_length, z_x + x_length], outline=rgb_temp, width=width
        )

    # Prepare image for zoom views with borders
    rimg = Image.new(
        "RGB", (ori_size[0], ori_size[1]), "white"
    )
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
            if method[0] == "LR":
                im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
            im_zoom = draw_box_zoom(im, zoom_p, zoom_size, offset)
            im_zoom.save(os.path.join(zoom_image_dir, f"{method[0]}.png"))

    if not flag:
        im_zoom = Image.new("RGB", ori_size, "gray")
        im_zoom = draw_box_zoom(im_zoom, zoom_p, zoom_size, offset)
        im_zoom.save(os.path.join(zoom_image_dir, f"{method[0]}.png"))

# Plotting the final figure
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 7))
plt.subplots_adjust(wspace=0.05, hspace=-0.25)  # 缩小子图之间的间距
plt.subplots_adjust(left=0, bottom=-0.05, top=1.12, right=1)  # 去除图像四周的空白
for i, method in enumerate(method_name):
    ax = axes[i // cols, i % cols]
    img_path = os.path.join(zoom_image_dir, f"{method[0]}.png")
    im = Image.open(img_path)
    ax.imshow(
        im.resize((int(im.width * rslt_img_scale), int(im.height * rslt_img_scale)))
    )
    # 设置标题在图像下方
    if isPrintAlphabet:
        title_text = f"{alphabet[i]}) {method[1]}"
    else:
        title_text = f"{method[1]}"
    ax.text(0.5, -0.05, title_text, fontsize=fontsize, ha='center', va='top', transform=ax.transAxes)
    ax.axis("off")

# plt.tight_layout()
plt.savefig(
    os.path.join(savepath, f'LCMSR_{setName}_{degrade.replace(".", "")}.png'),
    dpi=100,
)
shutil.rmtree(zoom_image_dir)
# plt.savefig(
#     os.path.join(savepath, f'mainrslt_{setName}_{degrade.replace(".", "")}.eps'),
#     format="eps",
# )