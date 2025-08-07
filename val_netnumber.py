import torch
from G_network.SPPF_UNet import *



results_file = "output/superunet2/checkpoint.txt"
generator_path = r"superunet2/best_generator.pth"
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = ResNetL3(3)# 确保类名正确
generator = SPPF777_DSUNet(3, 3)
generator.load_state_dict(torch.load(generator_path))
# 直接加载整个模型


# 确保model是一个nn.Module的实例
# 如果不是，你可能需要稍微调整加载方式或模型保存方式
assert isinstance(generator, torch.nn.Module), "Loaded object is not a torch.nn.Module"

# 计算参数量
total_params = sum(p.numel() for p in generator.parameters())
print(f"Total trainable parameters: {total_params}")


with open(results_file, "a") as f:
    train_info = f"参数量约为{total_params}。\n"
    f.write(train_info + "\n\n")

print("处理完成")