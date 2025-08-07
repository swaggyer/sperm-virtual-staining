from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from G_network.SPPF_UNet import SPPF_DSUNet, SPPF777_atten_DSUNet
from localutils.SSIM_Value import *
from torch.autograd import Variable
from dataset import *
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from G_network.U2Net import  *
import time
# from Utils.output import process_time

if __name__ == '__main__':

    img_dataroot = "/media/seven/WJH/pix2pix/data/sperm"

    transformer = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    val_dataloader = DataLoader(
        ImageDataset(img_dataroot, transform=transformer, mode="test"),
        batch_size=1,
        num_workers=1,
    )

    generator_path = r"/media/seven/WJH/pix2pix/saved_models/U2Net/best_generator.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    generator = u2net_full(3)

    generator.load_state_dict(torch.load(generator_path))
    generator.to(device)
    generator.eval()

    output_folder = r"/media/seven/WJH/pix2pix/betterdata/U2Net"
    results_file = r"/media/seven/WJH/pix2pix/betterdata/U2Net/checkpoint.txt"

    all_psnrs = []
    all_ssims = []
    all_mses = []
    process_times = 0
    num = 0


    for i,batch in enumerate(val_dataloader):
    # real_A是暗场显微，real_B是明场染色
        if batch is None or "A" not in batch or "B" not in batch:
            print(f"警告：在第 {i} 个批次中未找到有效数据")

        real_A = Variable(batch["B"].type(Tensor))  # 四维张量
        real_B = Variable(batch["A"].type(Tensor))
        num = num+1
        start_time = time.time()
        with torch.no_grad():
            fake_B = generator(real_A)
        color_time = time.time()
        process_times +=(color_time-start_time)

        img_savepath = os.path.join(output_folder, "%s.tif" % i)
        img_sample = torch.cat((real_A.data, real_B.data, fake_B.data), -1)
        save_image(img_sample, img_savepath,normalize=True)

        SSIM = ssim(real_B, fake_B, window_size=2, size_average=True).cpu().numpy()


        real_B=np.array(real_B[0].permute(1, 2, 0).cpu().numpy())
        fake_B=np.array(fake_B[0].permute(1, 2, 0).cpu().numpy())

        MSE = compare_mse(real_B, fake_B)
        PSNR = compare_psnr(real_B, fake_B)
        # SSIM = compare_ssim(real_B, fake_B, channel_axis=2, data_range=1)

        with open(results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr
            train_info = f"第 {i}组图片的评估值为：  " \
                     f"MSE:{MSE:.4f}  " \
                     f"PSNR: {PSNR:.4f}  " \
                     f"SSIM: {SSIM:.4f}  "
            f.write(train_info + "\n")

        all_mses.append(MSE)
        all_ssims.append(SSIM)
        all_psnrs.append(PSNR)
        print(f"第 {i}组图片的MSE为{MSE:.4f},PSNR为{PSNR:.4f},SSIM为{SSIM:.4f}" )
    total_params = sum(p.numel() for p in generator.parameters())
    process_time = process_times / num
    with open(results_file, "a") as f:
        train_info = f"上述图片组的综合评估值为:" \
                     f"Average MSE:{np.mean(all_mses):.4f}  " \
                     f"Average PSNR: {np.mean(all_psnrs):.4f}  " \
                     f"Average SSIM: {np.mean(all_ssims):.4f}  " \
                     f"网络参数总量: {total_params:.4f}  " \
                     f"加工染色时间: {process_time:.4f}  " \

        f.write(train_info + "\n")

    print(f"Average MSE: {np.mean(all_mses):.4f}")
    print(f"Average PSNR: {np.mean(all_psnrs):.4f}")
    print(f"Average SSIM: {np.mean(all_ssims):.4f}")
    print(f"网络参数总量: {total_params:.4f}  ")
    print(f"加工染色时间: {process_time:.4f}  ")
    print(f"{num}")



