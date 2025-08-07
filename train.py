import argparse
import time
import datetime
import numpy as np
from tqdm.autonotebook import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from G_network.SPPF_UNet import SPPF777_DSUNet
from dataset import ImageDataset
from tools.StyleLoss import StyleLoss
import torchvision.transforms as transforms
from D_network import *
import os

# --------------------------
# 参数解析配置
# --------------------------
parser = argparse.ArgumentParser(description='图像转换模型训练脚本')
parser.add_argument("--epoch", type=int, default=0, help="起始训练轮次")
parser.add_argument("--n_epochs", type=int, default=200, help="总训练轮次")
parser.add_argument("--dataset_name", type=str, default="SPPF_avg579_DS", help="数据集名称")
parser.add_argument("--data_root", type=str, default="data/sperm", help="数据集根目录")
parser.add_argument("--batch_size", type=int, default=20, help="批次大小")
parser.add_argument("--lr", type=float, default=0.0002, help="Adam学习率")
parser.add_argument("--b1", type=float, default=0.5, help="Adam一阶动量衰减")
parser.add_argument("--b2", type=float, default=0.999, help="Adam二阶动量衰减")
parser.add_argument("--decay_epoch", type=int, default=20, help="学习率衰减起始轮次")
parser.add_argument("--n_cpu", type=int, default=8, help="数据生成CPU线程数")
parser.add_argument("--img_height", type=int, default=256, help="图像高度")
parser.add_argument("--img_width", type=int, default=256, help="图像宽度")
parser.add_argument("--channels", type=int, default=3, help="图像通道数")
parser.add_argument("--sample_interval", type=int, default=100, help="样本生成间隔")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="模型保存间隔")
parser.add_argument("--save_best", default=True, type=bool, help="仅保存最佳模型")
opt = parser.parse_args()

# --------------------------
# 目录初始化
# --------------------------
os.makedirs(f"images/{opt.dataset_name}", exist_ok=True)
os.makedirs(f"saved_models/{opt.dataset_name}", exist_ok=True)


# --------------------------
# 工具函数定义
# --------------------------
def create_lr_scheduler(optimizer, num_step: int, epochs: int,
                        warmup: bool = True, warmup_epochs: int = 1, warmup_factor: float = 1e-3):
    """创建学习率调度器（带热身）"""
    assert num_step > 0 and epochs > 0

    def lr_func(x):
        if warmup and x <= warmup_epochs * num_step:
            alpha = x / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return ((1 - (x - warmup_epochs * num_step) /
                     ((epochs - warmup_epochs) * num_step))) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


def sample_images(batches_done):
    """保存验证集生成样本"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))

    with torch.no_grad():
        fake_B = generator(real_A)

    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, f"images/{opt.dataset_name}/{batches_done}.tif",
               nrow=5, normalize=True)


def update_best_model(batches_done, current_loss):
    """更新最佳模型（基于像素损失）"""
    global best_loss_pixel
    if current_loss < best_loss_pixel:
        print(f"保存最佳模型 | 当前像素损失: {current_loss:.4f}")
        torch.save(generator.state_dict(),
                   f"saved_models/{opt.dataset_name}/best_generator.pth")
        best_loss_pixel = current_loss


# --------------------------
# 模型与损失初始化
# --------------------------
# 初始化生成器和判别器
generator = SPPF777_DSUNet(3, 3)
discriminator = Discriminator()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = device.type == "cuda"

# 损失函数
criterion_GAN = nn.MSELoss().to(device)
criterion_pixelwise = nn.L1Loss().to(device)
style_loss = StyleLoss().to(device)  # 风格损失（备用）

# 超参数
lambda_pixel = 100
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 模型加载（若指定起始轮次）
if opt.epoch != 0:
    generator.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/generator_{opt.epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/discriminator_{opt.epoch}.pth"))

# --------------------------
# 数据加载器配置
# --------------------------
# 数据增强配置
transformer = [
    transforms.Resize((opt.img_height, opt.img_width), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# 训练集（随机打乱）
train_dataset = ImageDataset(opt.data_root, transform=transformer, mode="train")
dataloader = DataLoader(train_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.n_cpu)

# 验证集（固定顺序）
val_dataset = ImageDataset(opt.data_root, transform=transformer, mode="test")
val_dataloader = DataLoader(val_dataset,
                            batch_size=10,
                            shuffle=False,
                            num_workers=1)

# 测试集（单样本）
valdata_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)

# --------------------------
# 训练变量初始化
# --------------------------
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
best_loss_pixel = float('inf')
G_Loss, D_Loss, Pixel_loss, Gan_loss = [], [], [], []
epoch_times = []

# --------------------------
# 主训练循环
# --------------------------
prev_time = time.time()
results_file = f"G_network/Checkpoint/checkpoint_{opt.dataset_name}.txt"

for epoch in range(opt.epoch, opt.n_epochs):
    start_time = time.time()
    loop = tqdm(dataloader, leave=True, desc=f"Epoch [{epoch}/{opt.n_epochs}]")
    checkpoint = []  # 存储本轮损失

    for i, batch in enumerate(loop):
        # 数据加载（暗场->real_A，明场->real_B）
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # 对抗标签（真实=1，伪造=0）
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ----------------------
        # 生成器训练
        # ----------------------
        optimizer_G.zero_grad()

        # 生成伪造图像并计算对抗损失
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # 计算像素级重建损失
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # 总生成器损失
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        # 反向传播与优化
        loss_G.backward()
        optimizer_G.step()

        # ----------------------
        # 判别器训练
        # ----------------------
        optimizer_D.zero_grad()

        # 真实图像判别损失
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # 伪造图像判别损失（使用生成器的输出）
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # 总判别器损失
        loss_D = 0.5 * (loss_real + loss_fake)

        # 反向传播与优化
        loss_D.backward()
        optimizer_D.step()

        # 存储本轮损失
        checkpoint.extend([loss_D.item(), loss_G.item(),
                           loss_GAN.item(), loss_pixel.item()])

        # ----------------------
        # 进度条更新
        # ----------------------
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        loop.set_postfix(
            D_loss=f"{loss_D.item():.4f}",
            G_loss=f"{loss_G.item():.4f}",
            loss_GAN=f"{loss_GAN.item():.4f}",
            loss_pixel=f"{loss_pixel.item():.4f}",
            time_left=str(time_left),
            best_pixelloss=f"{best_loss_pixel:.4f}"
        )

        # ----------------------
        # 样本生成（间隔控制）
        # ----------------------
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # ----------------------
    # 轮次结束处理
    # ----------------------
    # 存储本轮平均损失
    D_Loss.append(checkpoint[-4])
    G_Loss.append(checkpoint[-3])
    Gan_loss.append(checkpoint[-2])
    Pixel_loss.append(checkpoint[-1])

    # 记录轮次耗时
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    # 更新最佳模型
    if opt.save_best:
        update_best_model(batches_done, Pixel_loss[-1])

    # 日志记录
    with open(results_file, "a") as f:
        log_info = (
            f"Epoch {epoch + 1:03d} | "
            f"D_loss: {loss_D.item():.6f} | "
            f"G_loss: {loss_G.item():.6f} | "
            f"Pixel_loss: {loss_pixel.item():.4f} | "
            f"Best_Pixel: {best_loss_pixel:.4f}\n"
        )
        f.write(log_info)

    # 模型保存（间隔控制）
    if epoch % 5 == 0:
        torch.save(generator.state_dict(),
                   f"saved_models/{opt.dataset_name}/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(),
                   f"saved_models/{opt.dataset_name}/discriminator_{epoch}.pth")

    # 输出轮次耗时
    print(f"Epoch {epoch + 1} 训练耗时: {epoch_time:.4f}s")

# --------------------------
# 训练总结
# --------------------------
total_time = np.sum(epoch_times)
print(f"\n训练完成！总耗时: {total_time:.4f}s")