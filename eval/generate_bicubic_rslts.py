import os, sys

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)

from PIL import Image
import numpy as np
from data.transforms import resize_matlab


def bicubic_process(datasetName, scale):
    """
    处理给定路径中的所有图像文件，使用 imresize 函数先缩小再恢复，然后保存结果。
    input_path: 输入图像文件夹路径。
    output_path: 输出图像文件夹路径。
    scale: 缩放倍数。
    """
    input_path = f"load/benchmark/{datasetName}/HR"
    output_path = f"logs/_results/{datasetName}/Bicubic/x{scale:.1f}"
    # 检查输出路径是否存在，不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径中的所有文件
    for filename in os.listdir(input_path):
        # 构建完整的文件路径
        file_path = os.path.join(input_path, filename)

        # 只处理图像文件
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # 读取图像
            image = np.array(Image.open(file_path))

            # 使用 imresize 函数进行缩放和恢复
            processed_image = resize_matlab(image, 1.0 / scale)
            processed_image = resize_matlab(processed_image, size=image.shape[:2])

            # 构建输出文件路径
            output_file_path = os.path.join(output_path, filename)

            # 保存处理后的图像
            Image.fromarray(processed_image).save(output_file_path)
            print(f"Processed and saved: {output_file_path}")


if __name__ == "__main__":
    # 设置输入输出路径和缩放倍数
    datasetName = "AID"  # 替换为数据集路径
    scale = 4  # 缩放倍数，示例中缩小到原来的 50%

    # 执行图像处理
    bicubic_process(datasetName, scale)
