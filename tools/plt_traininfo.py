import pandas as pd
import matplotlib.pyplot as plt


def plot_training_metrics(file_path):
    # 使用pandas读取文件
    # 假设文件是空格分隔的，且每行格式固定
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['epoch', 'D_loss', 'G_loss', 'loss_pixel', 'lr'])

    # 设置索引为epoch
    data.set_index('epoch', inplace=True)

    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制D_loss
    plt.plot(data.index, data['D_loss'], label='D_loss')

    # 绘制G_loss
    plt.plot(data.index, data['G_loss'], label='G_loss')

    # 绘制loss_pixel
    plt.plot(data.index, data['loss_pixel'], label='loss_pixel')

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Training Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 显示图形
    plt.grid(True)  # 显示网格
    plt.show()

if __name__ == '__main__':
    txtfil = 'G:\pix2pix\G_network\checkpoint_DS_LocalBranch.txt'
    plot_training_metrics(txtfil)
