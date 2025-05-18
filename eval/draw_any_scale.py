import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from copy import deepcopy

# 设置全局参数
rootpath = r"logs/_results"
alphabet = "abcdefghijklmnopqrstuvwxyz"
fontsize = 14
box_border_width = 4
rows, cols = 2, 4
isPrintAlphabet = False
savepath = r"logs/_results_paper"

def draw(setName, upscale_num, img_name, zoom_p, zoom_size, save_eps=False):
    if setName == "AID":
        setNameDisp = "AID"

    # 绘制
    upscale = f"x{upscale_num:.1f}"
    method_name = [
        ["hr_", "Image from ..."],
        ["Bicubic", "Bicubic"],
        ["LIIF", "LIIF"],
        ["SADN", "SADN"],
        ["CiaoSR", "CiaoSR"],
        ["LMF", "LMF-LTE"],
        ["IDM", "IDM"],
        ["LDCSR_ours", "OURS"],
        ["hr", "GT"],
    ]

    # 路径设置
    hr_path = os.path.join(rootpath, setName, "HR")

    # 结果图像缩放比例
    rslt_img_scale = 150 / zoom_size[0]

    if not os.path.exists(os.path.join(savepath, "zoom_image", f"{setName}_{upscale}")):
        os.makedirs(os.path.join(savepath, "zoom_image", f"{setName}_{upscale}"))

    sa_x, sa_y = zoom_p
    x_length, y_length = zoom_size
    width = box_border_width
    rgb = [234, 112, 14]

    # 读取并处理HR图像
    im_hr = Image.open(os.path.join(hr_path, img_name))
    im_hr = im_hr.convert("RGB")
    im_hr_raw = deepcopy(im_hr)

    draw = ImageDraw.Draw(im_hr)
    for c in range(3):
        # down
        draw.line([(sa_y, sa_x), (sa_y + y_length, sa_x)], fill=tuple(rgb), width=width)
        # up
        draw.line(
            [(sa_y, sa_x + x_length), (sa_y + y_length, sa_x + x_length)],
            fill=tuple(rgb),
            width=width,
        )
        # left
        draw.line([(sa_y, sa_x), (sa_y, sa_x + x_length)], fill=tuple(rgb), width=width)
        # right
        draw.line(
            [(sa_y + y_length, sa_x), (sa_y + y_length, sa_x + x_length)],
            fill=tuple(rgb),
            width=width,
        )

    im_hr.save(os.path.join(savepath, "zoom_image", f"{setName}_{upscale}", "hr_.png"))

    # 处理并保存其他方法的结果图像
    for method in method_name:
        savepath_temp = os.path.join(savepath, "zoom_image", f"{setName}_{upscale}")
        if method[0] == "hr":
            im_zoom = im_hr_raw.crop((sa_y, sa_x, sa_y + y_length, sa_x + x_length))
            im_zoom.save(os.path.join(savepath_temp, "hr.png"))
        elif method[0] == "hr_":
            continue
        else:
            path = os.path.join(rootpath, setName, method[0], upscale)
            if not os.path.exists(path):
                im_zoom = Image.new("RGB", (zoom_size[1], zoom_size[0]), "gray")
                im_zoom.save(os.path.join(savepath_temp, f"{method[0]}.png"))
                continue
            image_list = [img for img in os.listdir(path) if img.endswith(".png")]
            flag = False
            for img_name_temp in image_list:
                if img_name_temp == img_name:
                    flag = True
                    img = Image.open(os.path.join(path, img_name_temp))
                    im_zoom = img.crop((sa_y, sa_x, sa_y + y_length, sa_x + x_length))
                    im_zoom.save(os.path.join(savepath_temp, f"{method[0]}.png"))
            if not flag:
                im_zoom = Image.new("RGB", (zoom_size[1], zoom_size[0]), "gray")
                im_zoom.save(os.path.join(savepath_temp, f"{method[0]}.png"))

    # 拼图处理
    small_img_x, small_img_y = zoom_size[1], zoom_size[0]
    padding_x = 10 / rslt_img_scale
    padding_y = 34 / rslt_img_scale
    big_img_x = rows * small_img_y + padding_y * (rows - 1)
    big_img_y = big_img_x

    totalwidth = int(
        (big_img_x + padding_x) + (small_img_x + padding_x) * cols - padding_x
    )
    totalheight = int(big_img_y + padding_y)

    fig = plt.figure(
        figsize=(totalwidth * rslt_img_scale / 100, totalheight * rslt_img_scale / 100)
    )
    gt = Image.open(
        os.path.join(savepath, "zoom_image", f"{setName}_{upscale}", "hr.png")
    )
    psnr_list = []
    ssim_list = []

    for i, method in enumerate(method_name):
        if method[0] == "hr_":
            x, y = 0, totalheight - padding_y
            w, h = big_img_x / totalwidth, big_img_y / totalheight
            # method[1] = f'"{img_name.split(".")[0]}" from {setNameDisp} ({upscale})'
            # method[1] = f"Image from {setName}:{img_name.split('.')[0]} ({upscale})"
            method[1] = f"Image from {setName} ({upscale})"
        else:
            offset_x = big_img_x + padding_x
            row = (i - 1) // cols + 1
            col = (i - 1) % cols + 1
            x = (col - 1) * (small_img_x + padding_x) + offset_x
            y = row * (small_img_y + padding_y) - padding_y
            w, h = small_img_x / totalwidth, small_img_y / totalheight

        x /= totalwidth
        y = 1 - y / totalheight

        ax = fig.add_axes([x, y, w, h])
        im_zoom_path = os.path.join(
            savepath, "zoom_image", f"{setName}_{upscale}", f"{method[0]}.png"
        )
        img = Image.open(im_zoom_path)
        if i == 0:
            padding = img.width - img.height
            if padding > 0 and zoom_p[1] * 2 > img.width:
                img = img.crop((padding, 0, padding + img.height, img.height))
            elif padding > 0 and zoom_p[1] * 2 <= img.width:
                img = img.crop((0, 0, img.width, img.height))
            elif padding < 0 and zoom_p[0] * 2 > img.height:
                img = img.crop((0, -padding, img.width, -padding + img.width))
            elif padding < 0 and zoom_p[0] * 2 <= img.height:
                img = img.crop((0, 0, img.width, img.width))
            ax.imshow(img)
        else:
            ax.imshow(
                img.resize(
                    (int(img.width * rslt_img_scale), int(img.height * rslt_img_scale))
                )
            )

        ax.axis("off")
        text_ax = fig.add_axes(
            [x, y - padding_y / totalheight, w, padding_y / totalheight]
        )  # 调整这个坐标来控制文字的位置
        text_ax.axis("off")

        if isPrintAlphabet:
            name = f"{alphabet[i]}) "
        else:
            name = ""

        text_ax.text(
            0.5,
            0.25,
            name + method[1],
            ha="center",
            fontsize=fontsize,
            # fontname="Times New Roman",
            weight="normal",
        )

    # 保存图像
    plt.savefig(
        os.path.join(savepath, f'cs_{setName}_{upscale.replace(".", "")}.png'),
        dpi=150,
    )
    if save_eps:
        plt.savefig(
            os.path.join(savepath, f'cs_{setName}_{upscale.replace(".", "")}.eps'),
            format="eps",
        )

if __name__ == "__main__":
    # 数据集设置
    configs = [
    {
        "setName": "AID",
        "upscale_num": 3.4,
        "img_name": "church_64.png",
        "zoom_p": [200,10],
        "zoom_size": [100, 180]
    },
    {
        "setName": "AID",
        "upscale_num": 6,
        "img_name": "mediumresidential_155.png",
        "zoom_p": [350,150],
        "zoom_size": [100, 180]
    },
    {
        "setName": "DOTA",
        "upscale_num": 2.6,
        "img_name": "P1971.png", #1971
        "zoom_p": [140,250],
        "zoom_size": [100, 180]
    },
    {
        "setName": "DOTA",
        "upscale_num": 4,
        "img_name": "P0106.png",
        "zoom_p": [240,220],
        "zoom_size": [150, 270]
    },
    {
        "setName": "DOTA",
        "upscale_num": 4,
        "img_name": "P2431.png",
        "zoom_p": [150,30],
        "zoom_size": [150, 270]
    },
    {
        "setName": "DIOR",
        "upscale_num": 2,
        "img_name": "14976.png",
        "zoom_p": [480,330],
        "zoom_size": [100, 180]
    },
    {
        "setName": "DIOR",
        "upscale_num": 8,
        "img_name": "14976.png",
        "zoom_p": [480,230],
        "zoom_size": [300, 540]
    },
    {
        "setName": "DIOR",
        "upscale_num": 4,
        "img_name": "13217.png",
        "zoom_p": [240,420],
        "zoom_size": [100, 180]
    },
    ]
    for row in configs:
        draw(**row)