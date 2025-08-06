import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import matplotlib.pyplot as plt


def main(args):
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "quality_results.txt")

    # 获取真实图像和生成图像列表
    real_images = sorted([f for f in os.listdir(args.real_dir) if f.endswith(('.tif', '.png', '.jpg'))])
    fake_images = sorted([f for f in os.listdir(args.fake_dir) if f.endswith(('.tif', '.png', '.jpg'))])

    if len(real_images) != len(fake_images):
        print(f"警告: 真实图像数量({len(real_images)})与生成图像数量({len(fake_images)})不匹配")
        min_count = min(len(real_images), len(fake_images))
        real_images = real_images[:min_count]
        fake_images = fake_images[:min_count]

    all_psnrs = []
    all_ssims = []
    all_mses = []
    per_image_results = []

    print(f"开始评估 {len(real_images)} 对图像的质量...")

    for i, (real_name, fake_name) in enumerate(zip(real_images, fake_images)):
        try:
            # 加载图像
            real_img = np.array(Image.open(os.path.join(args.real_dir, real_name)))
            fake_img = np.array(Image.open(os.path.join(args.fake_dir, fake_name)))

            # 确保图像尺寸匹配
            if real_img.shape != fake_img.shape:
                print(f"图像尺寸不匹配: {real_name}({real_img.shape}) vs {fake_name}({fake_img.shape})")
                continue

            # 计算质量指标
            MSE = compare_mse(real_img, fake_img)
            PSNR = compare_psnr(real_img, fake_img)
            SSIM = compare_ssim(real_img, fake_img, channel_axis=2)

            # 记录结果
            all_mses.append(MSE)
            all_ssims.append(SSIM)
            all_psnrs.append(PSNR)
            per_image_results.append((real_name, fake_name, MSE, PSNR, SSIM))

            # 记录到文件
            with open(results_file, "a") as f:
                train_info = (f"第 {i + 1}组图片评估: "
                              f"真实图像: {real_name}, "
                              f"生成图像: {fake_name}, "
                              f"MSE: {MSE:.6f}, "
                              f"PSNR: {PSNR:.6f}, "
                              f"SSIM: {SSIM:.6f}\n")
                f.write(train_info)

            print(f"处理完成: {real_name} vs {fake_name} | MSE: {MSE:.4f}, PSNR: {PSNR:.2f}, SSIM: {SSIM:.4f}")

        except Exception as e:
            print(f"处理图像对 {real_name} 和 {fake_name} 时出错: {str(e)}")

    # 计算平均指标
    avg_mse = np.mean(all_mses) if all_mses else 0
    avg_psnr = np.mean(all_psnrs) if all_psnrs else 0
    avg_ssim = np.mean(all_ssims) if all_ssims else 0

    # 输出综合结果
    with open(results_file, "a") as f:
        summary = (f"\n综合评估结果:\n"
                   f"处理图像对数: {len(per_image_results)}\n"
                   f"平均 MSE: {avg_mse:.6f} ± {np.std(all_mses):.6f}\n"
                   f"平均 PSNR: {avg_psnr:.6f} ± {np.std(all_psnrs):.6f}\n"
                   f"平均 SSIM: {avg_ssim:.6f} ± {np.std(all_ssims):.6f}\n")
        f.write(summary)

    # 控制台输出
    print("\n" + "=" * 50)
    print("图像质量评估完成")
    print("=" * 50)
    print(f"处理图像对数: {len(per_image_results)}")
    print(f"平均 MSE: {avg_mse:.6f} ± {np.std(all_mses):.6f}")
    print(f"平均 PSNR: {avg_psnr:.6f} ± {np.std(all_psnrs):.6f}")
    print(f"平均 SSIM: {avg_ssim:.6f} ± {np.std(all_ssims):.6f}")
    print("=" * 50)

    # 生成可视化报告
    generate_visual_report(per_image_results, args.output_dir)

    return {
        "image_pairs": len(per_image_results),
        "avg_mse": avg_mse,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "mse_std": np.std(all_mses),
        "psnr_std": np.std(all_psnrs),
        "ssim_std": np.std(all_ssims)
    }


def generate_visual_report(results, output_dir):
    """生成可视化质量报告"""
    try:
        # 提取指标数据
        indices = list(range(1, len(results) + 1))
        mses = [r[2] for r in results]
        psnrs = [r[3] for r in results]
        ssims = [r[4] for r in results]

        # 创建图表
        plt.figure(figsize=(15, 10))

        # MSE图表
        plt.subplot(3, 1, 1)
        plt.plot(indices, mses, 'bo-', label='MSE')
        plt.axhline(y=np.mean(mses), color='r', linestyle='--', label='平均值')
        plt.xlabel('图像对编号')
        plt.ylabel('MSE')
        plt.title('均方误差(MSE)分布')
        plt.legend()
        plt.grid(True)

        # PSNR图表
        plt.subplot(3, 1, 2)
        plt.plot(indices, psnrs, 'go-', label='PSNR')
        plt.axhline(y=np.mean(psnrs), color='r', linestyle='--', label='平均值')
        plt.xlabel('图像对编号')
        plt.ylabel('PSNR (dB)')
        plt.title('峰值信噪比(PSNR)分布')
        plt.legend()
        plt.grid(True)

        # SSIM图表
        plt.subplot(3, 1, 3)
        plt.plot(indices, ssims, 'mo-', label='SSIM')
        plt.axhline(y=np.mean(ssims), color='r', linestyle='--', label='平均值')
        plt.xlabel('图像对编号')
        plt.ylabel('SSIM')
        plt.title('结构相似性(SSIM)分布')
        plt.legend()
        plt.grid(True)

        # 保存图表
        plt.tight_layout()
        report_path = os.path.join(output_dir, "quality_report.png")
        plt.savefig(report_path)
        plt.close()

        print(f"已生成可视化报告: {report_path}")

    except Exception as e:
        print(f"生成可视化报告时出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='虚拟染色图像质量评估工具')
    parser.add_argument('--real_dir', type=str, required=True,
                        help='真实染色图像目录路径')
    parser.add_argument('--fake_dir', type=str, required=True,
                        help='虚拟染色图像目录路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录 (默认: results)')

    args = parser.parse_args()
    results = main(args)