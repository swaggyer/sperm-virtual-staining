import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from G_network.SPPF_UNet import *

# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



# 加载模型
generator_path = "/media/seven/WJH/pix2pix/saved_models/SPPF777_DSUNet/best_generator.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
generator = SPPF777_DSUNet(3)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.to(device)
generator.eval()

# 定义输入和输出文件夹
input_folder = "/media/seven/WJH/pix2pix/2048an/"
output_folder = "/media/seven/WJH/pix2pix/2048virtual/SPPF777_DSUNet"
results_file = "/2048virtual/SPPF777_DSUNet/checkpoint.txt"
img_size = [2048, 2048]

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

total_times = 0
process_times = 0

# 使用DataLoader加载图像
from torch.utils.data import DataLoader, Dataset
class ImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        filename = os.listdir(self.folder)[idx]
        img_path = os.path.join(self.folder, filename)
        image = Image.open(img_path).convert('RGB')
        return transform(image).to(device)

dataset = ImageDataset(input_folder)
dataloader = DataLoader(dataset, batch_size=1)

for tensor_img in dataloader:
    start_time = time.time()
    with torch.no_grad():
        fake_img = generator(tensor_img)
    color_time = time.time()
    fake_img = (fake_img + 1) / 2
    a_time = time.time()
    fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    end_time = time.time()

    process_time = end_time - a_time
    process_times += process_time

    total_time = process_time + (color_time - start_time)
    total_times += total_time

    output_path = os.path.join(output_folder, os.listdir(input_folder)[dataloader._index])
    Image.fromarray((fake_img * 255).astype('uint8')).save(output_path)

color_time = (total_times - process_times) / len(os.listdir(input_folder))
process_time = process_times / len(os.listdir(input_folder))
total_params = sum(p.numel() for p in generator.parameters())

print(f"Total trainable parameters: {total_params}")
print(generator_path)
print(f"单次染色推理需要 {color_time:.4f} 秒。")
print(f"其余图像操作需要 {process_time:.4f} 秒。")
print(f"处理完成！总共花费了 {total_times:.2f} 秒。")
print(f"网络参数量: {total_params}")
print("处理完成！")

with open(results_file, "a") as f:
    train_info = f"单次染色{img_size}尺寸的图片推理需要 {color_time:.4f} 秒。\n" \
                 f"其余图像操作需要 {process_time:.4f} 秒。\n" \
                 f"处理完成！总共花费了 {total_times:.2f} 秒。\n" \
                 f"生成器路径为{generator_path}。\n" \
                 f"参数量约为{total_params}。\n"
    f.write(train_info + "\n\n")