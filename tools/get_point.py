from PIL import Image
import numpy as np
import cv2


def read_image(image_path):
    """读取图像并返回PIL图像对象"""
    image = Image.open(image_path)
    image = image.resize((256, 256))  # 确保图像尺寸为256x256
    return image


def convert_to_grayscale(image):
    """将PIL图像转换为灰度图，并返回numpy数组"""
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    return gray_image


def get_line_pixels_with_coords(image, start, end):
    """获取指定直线上的像素值及其坐标"""
    height, width = image.shape
    x0, y0 = start
    x1, y1 = end

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    pixels_with_coords = []
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels_with_coords.append((y0, x0, image[y0, x0]))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return pixels_with_coords


def blackout_line_on_image(image, line_pixels_with_coords):
    """在原图像中将指定直线区域标黑"""
    image_np = np.array(image)
    for y, x, _ in line_pixels_with_coords:
        image_np[y, x, :] = [0, 0, 0]  # 将RGB值设置为黑色
    return Image.fromarray(image_np)


def save_image(image, output_file):
    """将图像保存到文件"""
    image.save(output_file)

def save_pixels_to_txt(pixels, output_file):

    """将像素值保存到txt文件中"""

    with open(output_file, 'w') as f:

        for pixel in pixels:

            f.write(str(pixel) + '\n')  # 灰度图的像素值不需要逗号分隔

def main():
    image_path = '/media/seven/WJH/pix2pix/7_332/an_horizontal.tif'  # 替换为你的图像路径
    start_point = (238, 95)  # 替换为你的起始点坐标
    end_point = (234, 130)  # 替换为你的终点坐标
    output_file_pixels = '/media/seven/WJH/pix2pix/7_332/an_+.txt'  # 输出文件路径（像素值）
    output_file_image = '/media/seven/WJH/pix2pix/7_332/an_+.tif'  # 输出文件路径（图像）

    image = read_image(image_path)
    print("已读")
    gray_image = convert_to_grayscale(image)
    line_pixels_with_coords = get_line_pixels_with_coords(gray_image, start_point, end_point)
    print("提取完成")

    # 可选：将像素值保存到txt文件中（如果需要）
    save_pixels_to_txt([pixel for _, _, pixel in line_pixels_with_coords], output_file_pixels)

    blackout_image = blackout_line_on_image(image, line_pixels_with_coords)
    save_image(blackout_image, output_file_image)

    print(f"像素值图像已保存到 {output_file_image}")


if __name__ == "__main__":
    main()
    print("end")
