import cv2

# 图像路径
img_path = '/media/seven/WJH/pix2pix/betterdata/SPPF777_DSUNet_styleloss/14.tif'

# 读取图像
img = cv2.imread(img_path)

# 检查图像是否成功读取
if img is None:
    print("Error: Could not read image from path.")
else:
    # 获取图像尺寸
    height, width, channels = img.shape

    # 确保图像宽度是768，高度是256（根据你的描述）
    assert width == 768 and height == 256, "Image dimensions do not match the expected size."

    # 计算每个区域的宽度
    region_width = width // 3

    # 切割图像
    dark_field = img[:, 0:region_width, :]  # 左边1/3
    bright_field = img[:, region_width:2 * region_width, :]  # 中间1/3
    virtual_field = img[:, 2 * region_width:3 * region_width, :]  # 右边1/3

    # 保存切割后的图像
    dark_field_path = '/media/seven/WJH/pix2pix/14data/dark_14.tif'
    bright_field_path = '/media/seven/WJH/pix2pix/14data/bright_14.tif'
    virtual_field_path = '/media/seven/WJH/pix2pix/14data/virtual_14.tif'

    cv2.imwrite(dark_field_path, dark_field)
    cv2.imwrite(bright_field_path, bright_field)
    cv2.imwrite(virtual_field_path, virtual_field)

    print("Images have been cut and saved successfully.")