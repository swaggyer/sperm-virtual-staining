import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm
from torch.autograd import Variable
import argparse
from G_network.SPPF_UNet import *
# 命令行参数解析
parser = argparse.ArgumentParser(description='Virtual staining speed evaluation')
parser.add_argument("--generator_path", type=str, required=True, help="Path to generator model")
parser.add_argument("--input_folder", type=str, required=True, help="Input image directory")
parser.add_argument("--output_folder", type=str, required=True, help="Output directory")
parser.add_argument("--img_size", type=int, nargs=2, default=[2048, 2048], help="Image size [width, height]")
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_folder, exist_ok=True)
results_file = os.path.join(args.output_folder, "speed_results.txt")

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = SPPF777_DSUNet(3, 3)
generator.load_state_dict(torch.load(args.generator_path))
generator.to(device)
generator.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 模型预热
dummy_input = torch.rand(1, 3, 256, 256).to(device)
with torch.no_grad():
    for _ in range(100):
        _ = generator(dummy_input)

color_times = 0
processed_count = 0

# 处理所有图像
for filename in tqdm(os.listdir(args.input_folder), desc="Processing images"):
    if filename.endswith('.tif'):
        img_path = os.path.join(args.input_folder, filename)

        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            tensor_img = transform(image).unsqueeze(0).to(device)

            # 同步GPU并计时
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            # 虚拟染色
            with torch.no_grad():
                fake_img = generator(tensor_img)

            torch.cuda.synchronize(device)
            end_time = time.perf_counter()

            # 记录时间
            inference_time = end_time - start_time
            color_times += inference_time
            processed_count += 1

            # 后处理并保存
            fake_img = (fake_img + 1) / 2
            fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_path = os.path.join(args.output_folder, filename)
            Image.fromarray((fake_img * 255).astype('uint8')).save(output_path)

            print(f"Processed {filename}: {inference_time:.6f} seconds")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# 计算统计结果
if processed_count > 0:
    avg_time = color_times / processed_count
    total_params = sum(p.numel() for p in generator.parameters())

    # 输出结果
    print(f"\n{'=' * 50}")
    print(f"Total images processed: {processed_count}")
    print(f"Average inference time: {avg_time:.6f} seconds")
    print(f"Network parameters: {total_params}")
    print(f"Model path: {args.generator_path}")
    print(f"{'=' * 50}")

    # 保存结果到文件
    with open(results_file, "w") as f:
        f.write(f"Virtual Staining Speed Test Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Model: {args.generator_path}\n")
        f.write(f"Input folder: {args.input_folder}\n")
        f.write(f"Output folder: {args.output_folder}\n")
        f.write(f"Image size: {args.img_size[0]}x{args.img_size[1]}\n")
        f.write(f"Total images processed: {processed_count}\n")
        f.write(f"Average inference time: {avg_time:.6f} seconds\n")
        f.write(f"Network parameters: {total_params}\n")
        f.write(f"{'=' * 50}\n")
else:
    print("No valid images processed")