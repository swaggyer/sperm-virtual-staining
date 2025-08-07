import os
import time
from PIL import Image
import torchvision.transforms as transforms
from G_network.SPPF_UNet import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

torch.cuda.empty_cache()
generator_path = "/media/seven/WJH/pix2pix/saved_models/SPPF777_DSUNet_styleloss/best_generator.pth"
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
# print(device)
# generator = ResNetL3(3)# 确保类名正确
generator = SPPF777_DSUNet(3)
generator.load_state_dict(torch.load(generator_path))
generator.to(device)
# 打印输出generator的参数大小

generator.eval()  # 设置为评估模式

# 定义输入和输出文件夹
input_folder = "/media/seven/WJH/pix2pix/7image_an"
output_folder = "/media/seven/WJH/pix2pix/7image_virtual/SPPF7_DSUNet_styleloss"
results_file = "/media/seven/WJH/pix2pix/7image_virtual/SPPF7_DSUNet_styleloss/checkpoint.txt"
img_size = [256,256]
# 图像预处理（根据生成器的输入需求进行调整）
# 这里假设生成器需要归一化到[-1, 1]的tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

total_times = 0
process_times = 0

for filename in os.listdir(input_folder):
     if filename.endswith('.tif'):
         # 加载图像
         img_path = os.path.join(input_folder, filename)
         image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
         start_time = time.time()
         # 应用预处理
         # print(torch.cuda.memory_allocated()/(1024*1024))
         tensor_img = transform(image).unsqueeze(0).to(device)  # 增加一个批次维度
         # print(torch.cuda.memory_allocated()/(1024*1024))
         # print(tensor_img.shape)
        # 使用生成器处理图像
         with torch.no_grad():  # 不计算梯度
             fake_img = generator(tensor_img)
         # Tnch = nn.Tanh()
         # fake_img = Tnch(fake_img)
         color_time = time.time()
         color_times =color_time-start_time
         # 假设生成器输出也是在[-1, 1]之间，我们需要将其转换回[0, 1]并保存为图像
         fake_img = (fake_img + 1) / 2  # 转换回[0, 1]
         # fake_img = fake_img.clamp(0, 1)  # 确保值在有效范围内
         a_time = time.time()
         fake_img = fake_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转换回CHW -> HWC并移除批次维度
         end_time = time.time()

         process_time = end_time - a_time
         process_times +=process_time


         total_time = process_time+color_times
         total_times +=total_time

         # 保存图像
         output_path = os.path.join(output_folder, filename)
         Image.fromarray((fake_img * 255).astype('uint8')).save(output_path)
         # print( Image.fromarray((fake_img * 255).astype('uint8')))

 #end_time = time.time()
 #total_time = end_time - start_time
color_time = (total_times-process_times)/551.0
process_time = process_times/551.0
total_params = sum(p.numel() for p in generator.parameters())
print(f"Total trainable parameters: {total_params}")
print(generator_path)
print(f"单次染色推理需要 {color_times:4f} 秒。")
print(f"其余图像操作需要 {process_time:4f} 秒。")
print(f"处理完成！总共花费了 {total_times:.2f} 秒。")
print(f"网络参数量: {total_params}")
print("处理完成！")
# print(generator)

with open(results_file, "a") as f:
    train_info = f"单次染色{img_size}尺寸的图片推理需要 {color_times:4f} 秒。\n" \
                 f"其余图像操作需要 {process_time:4f} 秒。\n" \
                 f"处理完成！总共花费了 {total_times:.2f} 秒。\n" \
                 f"生成器路径为{generator_path}。\n"\
                 f"参数量约为{total_params}。\n"

    f.write(train_info + "\n\n")



