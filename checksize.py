import os
from PIL import Image

def check_image_size(root_dir, target_width=600, target_height=600):
    # 存储不符合尺寸要求的图片文件路径
    mismatched_images = []
    
    # 遍历目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名，确认是否是图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    # 打开图片
                    with Image.open(file_path) as img:
                        width, height = img.size
                        # 检查图片尺寸
                        if width != target_width or height != target_height:
                            mismatched_images.append(file_path)
                except IOError:
                    print(f"Cannot open {file_path}")
    
    # 返回不符合尺寸要求的图片列表
    return mismatched_images

# 使用示例
directory = '/root/autodl-tmp/E2SR/logs/results/AID'  # 将此路径替换为你的目录路径
bad_images = check_image_size(directory)
if bad_images:
    print("以下图片尺寸不是600x600：")
    for img in bad_images:
        print(img)
else:
    print("所有图片都是600x600尺寸。")
