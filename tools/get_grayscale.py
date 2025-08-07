from PIL import Image
import numpy as np


def get_grayscale_histogram(image_path, output_txt_path):
    # 打开图像并转换为灰度图像
    image = Image.open(image_path).convert('L')

    # 将图像数据转换为NumPy数组
    image_array = np.array(image)

    # 计算灰度直方图
    histogram, bin_edges = np.histogram(image_array.flatten(),
                                        bins=range(257))  # bins=range(257) 表示从0到255的灰度级，加上一个边界256

    # 将灰度直方图数据保存到txt文件中
    with open(output_txt_path, 'w') as f:
        for i, count in enumerate(histogram):
            f.write(f"{bin_edges[i]}: {count}\n")

    print(f"灰度直方图数据已保存到 {output_txt_path}")


# 示例使用
image_path = '/media/seven/WJH/pix2pix/2048virtual/UNet/1.tif'  # 替换为你的图像路径
output_txt_path = '/media/seven/WJH/pix2pix/2048virtual/UNet/1_gray_UNet.txt'  # 替换为你想要保存的txt文件路径

get_grayscale_histogram(image_path, output_txt_path)