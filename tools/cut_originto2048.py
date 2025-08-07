import os
import cv2
import shutil

# 输入文件夹路径
input_folder = '/media/seven/WJH/pix2pix/ming1r'
# 输出文件夹路径
output_folder = '/media/seven/WJH/pix2pix/2048ming'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构造文件的完整路径
    file_path = os.path.join(input_folder, filename)

    # 检查文件是否为TIF格式
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
        # 读取图像
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        # 检查图像尺寸是否满足要求（3840x2560）

            # 裁剪图像，左上角部分，大小为2048x2048
        cropped_image = image[:2048, :2048]

            # 构造输出图像的完整路径
        output_file_path = os.path.join(output_folder, filename)

            # 存储裁剪后的图像
        cv2.imwrite(output_file_path, cropped_image)
        print(f"Processed and saved: {output_file_path}")


print("All images processed.")