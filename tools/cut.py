from PIL import Image
import os


def slice_image(image_path, output_folder):
    # 图片尺寸和切割参数
    slice_size = 256
    step_size = 128

    # 加载图片
    img = Image.open(image_path)
    # print(image_path)
    img_name = image_path.split()
    # print(img_name[1])
    # print(img_name[1][:-4])
    image_width, image_height = img.size

    # 初始化切割起始点列表
    start_points_width = list(range(0, image_width, step_size))
    start_points_height = list(range(0, image_height, step_size))

    # 计算可切割的最大次数（包含最后一个可能的起始点）
    max_cuts_width = (image_width - slice_size) // step_size + 1
    max_cuts_height = (image_height - slice_size) // step_size + 1

    i=0

    # 执行切割并保存图片
    for idx, y in enumerate(start_points_height[:max_cuts_height]):
        for jdx, x in enumerate(start_points_width[:max_cuts_width]):
            # 计算结束坐标

            end_x = min(x + slice_size, image_width)
            end_y = min(y + slice_size, image_height)

            # 提取区域
            cropped_img = img.crop((x, y, end_x, end_y))

            # 构建新文件名，按切割顺序命名
            new_file_name = f"{img_name[1][:-4]}_{i}.tif"
            i=i+1
            output_path = os.path.join(output_folder, new_file_name)

            # 保存切割后的图片
            cropped_img.save(output_path)
            print(f"已保存: {output_path}")


# 指定文件夹路径
input_folder = r'/media/seven/WJH/pix2pix/store7'
output_folder = r'/media/seven/WJH/pix2pix/7image_he'

# 遍历文件夹中的.tif文件
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        image_path = os.path.join(input_folder, filename)
        print(f"正在处理: {image_path}")
        slice_image(image_path, output_folder)

