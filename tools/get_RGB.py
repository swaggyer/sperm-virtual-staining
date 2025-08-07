from PIL import Image
import numpy as np

def read_image(image_path):
    """读取图像并返回PIL图像对象"""
    image = Image.open(image_path)
    image = image.resize((256, 256))  # 确保图像尺寸为256x256
    return image

def get_line_pixels_with_coords(image, start, end):
    """获取指定直线上的像素值及其坐标（RGB值）"""
    width, height = image.size
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
            # 注意：这里我们直接从PIL图像对象中获取RGB值
            r, g, b = image.getpixel((x0, y0))
            pixels_with_coords.append((r, g, b))

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

def save_pixels_to_txt(pixels, output_file):
    """将像素值（RGB）保存到txt文件中"""
    with open(output_file, 'w') as f:
        for (r, g, b) in pixels:

            f.write(f"{r},{g},{b}\n")

def main():
    image_path = '/media/seven/WJH/pix2pix/7_332/vs.tif'  # 替换为你的图像路径
    start_point = (0,0)  # 替换为你的起始点坐标
    end_point = (255, 255)  # 替换为你的终点坐标
    output_file_pixels = '/media/seven/WJH/pix2pix/7_332/VS_total_RGB.txt'  # 输出文件路径（像素值）

    image = read_image(image_path)
    print("已读")
    line_pixels_with_coords = get_line_pixels_with_coords(image, start_point, end_point)
    print("提取完成")

    # 将像素值保存到txt文件中
    save_pixels_to_txt(line_pixels_with_coords, output_file_pixels)

    print(f"像素值已保存到 {output_file_pixels}")

if __name__ == "__main__":
    main()
    print("end")