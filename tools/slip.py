import os
from PIL import Image
import numpy as np



import os
from PIL import Image


def slice_images(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 遍历输入文件夹中的所有文件
    for i in range(426):  # 从0到425，共426个文件
        file_name = f"{i}.tif"
        input_file_path = os.path.join(input_folder, file_name)

        # 检查文件是否存在
        if os.path.exists(input_file_path):
            # 打开图像文件
            with Image.open(input_file_path) as img:
                # 确保图像的尺寸是768*256
                if img.size == (768, 256):
                    # 截取左边2/3部分，即横坐标为[0:511]
                    sliced_img = img.crop((0, 0, 511, 256))

                    # 构建输出文件路径
                    output_file_path = os.path.join(output_folder, file_name)

                    # 保存切片后的图像
                    sliced_img.save(output_file_path)
                else:
                    print(f"警告: 图像 {file_name} 的尺寸不是768*256，跳过处理。")
        else:
            print(f"警告: 文件 {file_name} 不存在，跳过处理。")

        # 指定输入和输出文件夹路径

if __name__ == '__main__':

    input_folder = "/media/seven/WJH/pix2pix/val_smallview/UNet++"
    output_folder = "/media/seven/WJH/pix2pix/val_smallview/slipimages"

# 调用函数进行图像处理
    slice_images(input_folder, output_folder)