import os.path
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, pylab
from matplotlib.lines import Line2D
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from skimage.color import label2rgb
from skimage.morphology import dilation, square
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.Utils import to_networkx
from PIL import ImageDraw
# from GCN import get_graph_list
# from get_fullGraph import create_adjacency_matrix
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.measure import label, regionprops
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import mayavi.mlab as mlab
from PIL import Image
import matplotlib.lines as mlines


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.previous_best_score = 0
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, model, result_save_address, time_dir):  # call方法：使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用，按此参数列表进行传参
        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(val_acc, model, result_save_address, time_dir)
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.previous_best_score = self.best_score
            self.best_score = val_acc
            self.save_checkpoint(val_acc, model, result_save_address, time_dir)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, result_save_address, time_dir):
        if self.verbose:
            print(
                f'Validation accuracy increased ({self.previous_best_score * 100:.4f}% --> {self.best_score * 100:.4f}%).  Saving model ...')
        # torch.save(model.state_dict(), os.path.join(result_save_address, time_dir, 'best_model.pt'))
        torch.save(model, os.path.join(result_save_address, time_dir, 'best_model.pt'))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # 计算 softmax 概率
        pt = torch.exp(-ce_loss)

        # 计算 focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 权重 alpha
        if self.alpha is not None:
            alpha_weight = torch.gather(self.alpha, 0, target.view(-1))
            focal_loss = alpha_weight * focal_loss

        # 根据 reduction 选项进行汇总
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option: {}".format(self.reduction))


# 钩子函数获取特定层输出
def get_layer_output(model, layer_name, input_data, device):
    # 将模型移动到指定设备上
    model.to(device)
    model.eval()

    # 定义钩子函数来获取特定层的输出
    outputs = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # 获取模型中的特定层
    layer = dict([*model.named_modules()])[layer_name]
    # 注册钩子函数
    hook = layer.register_forward_hook(hook_fn)

    # 输入数据并获取输出
    with torch.no_grad():
        _ = model(input_data.to(device))

    # 获取钩子函数捕获的输出并取消注册
    layer_output = outputs[0].cpu().numpy()
    hook.remove()

    return layer_output


def visualize_tsne(layer_output):
    # 进行t-SNE降维
    tsne = TSNE(n_components=2, learning_rate=200, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(layer_output.reshape(layer_output.shape[0], -1))

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.title('t-SNE Visualization of Layer Output')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def calculate_alpha(labels):
    # 计算每个类别的样本数量
    class_counts = np.bincount(labels.astype(int))
    # 计算每个类别的权重
    class_weights = 1. / class_counts
    # 归一化权重，使它们的总和为1
    class_weights = class_weights / np.sum(class_weights)
    return class_weights.tolist()


def plot_confusion(labels, predicted, save_add, tim_dir):
    # 绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    # 真实标签和预测标签计算混淆矩阵
    cm = confusion_matrix(labels, predicted)
    # 将cm转换为百分比
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    # 设置类别标签
    class_names = ['Normal', 'LGSC', 'HGSC']
    # 创建图表
    fig, ax = plt.subplots()
    # 绘制混淆矩阵的热力图
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names, ax=ax)
    # 添加坐标轴标签
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    # 设置图表标题
    ax.set_title('Confusion Matrix')
    # 自动调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(save_add, tim_dir, "Confusion Matrix.png"))
    # 显示图表
    plt.show()

    # 绘制不归一化的混淆矩阵
    cm2 = confusion_matrix(labels, predicted)
    class_names = ['Normal', 'LGSC', 'HGSC']
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm2, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted label')
    ax2.set_ylabel('True label')
    ax2.set_title('No_normal_Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_add, tim_dir, "No_normal_Confusion Matrix.png"))
    plt.show()


# 保存绘图数据为CSV文件
def save_as_csv(save_add, tim_dir, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
    df1 = pd.DataFrame({'epoch': epoch})
    df2 = pd.DataFrame({'train_loss': train_loss})
    df3 = pd.DataFrame({'val_loss': val_loss})
    df4 = pd.DataFrame({'train_accuracy': train_accuracy})
    df5 = pd.DataFrame({'val_accuracy': val_accuracy})
    df_combined = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df_combined.to_csv(os.path.join(save_add, tim_dir, 'Acc_loss_curve.csv'), index=False)


# def plt_curve(path: str, epoch:int,args, label, title):
#     plt.figure(figsize=(10,5))
#     plt.plot(range(1, epoch + 1), args, label=label)
#     plt.xlabel('epochs')
#     plt.ylabel(label)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(path, title+".png"))
#     plt.show()


def plt_curve(path: str, epoch: int, args: list, labels: list, title: str):
    # 确保args和labels的长度相同
    if len(args) != len(labels):
        raise ValueError("The length of 'args' and 'labels' must be the same.")

    plt.figure(figsize=(10, 5))

    # 遍历args和labels，绘制每条曲线
    for data, label in zip(args, labels):
        if len(data) != epoch:
            raise ValueError(f"Data length must match the epoch number. Got {len(data)} for epoch {epoch}")
        plt.plot(range(1, epoch + 1), data, label=label)

    plt.xlabel('epochs')
    plt.ylabel('Values')  # 或者可以根据具体需求调整ylabel
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, title + ".png"))
    plt.show()


def save_args_to_txt(args, filename):
    with open(filename, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')


def output_metric(tar, pre):
    # 先判断数据所在设备
    if torch.is_tensor(tar):
        tar_cpu = tar.cpu().numpy()
        pre_cpu = pre.cpu().numpy()
    else:
        tar_cpu = tar
        pre_cpu = pre
    matrix = confusion_matrix(tar_cpu, pre_cpu)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def plot_tsne(z, y):  # z is model prediction, y is ground truth
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    z_tsne = tsne.fit_transform(z)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_tsne[y == 0, 0], z_tsne[y == 0, 1], c='r', label='normal', s=10)
    plt.scatter(z_tsne[y == 1, 0], z_tsne[y == 1, 1], c='g', label='low', s=10)
    plt.scatter(z_tsne[y == 2, 0], z_tsne[y == 2, 1], c='b', label='high', s=10)
    plt.legend()
    plt.show()


# 利用nx库对图进行可视化
def graph_nx_vis(graph, file_name):  # graph is a torch_geometric.data.Data object
    # initialize Figure
    plt.figure(num=None, figsize=(40, 40), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    graph = to_networkx(graph)
    pos = nx.spring_layout(graph)  # pos是一个字典，字典的键是节点的编号，字典的值是节点的坐标
    nx.draw_networkx_nodes(graph, pos)
    # 打印输出节点的个数
    print("number of nodes: ", graph.number_of_nodes())
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.show()
    # plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig


# 原图片叠加图可视化
def superpixel2graph_visualization(img, segments) -> (Image.Image, Image.Image):
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.where(segments == i), axis=1) for i in segments_ids])
    adj_matrix = create_adjacency_matrix(segments)

    # 在原图像上显示带有边界的图像
    boundaries_image = mark_boundaries(img, segments, color=(0, 0, 1))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(boundaries_image)

    # 保存原图像带边界的图像
    g_image = Image.fromarray((boundaries_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(g_image)

    # 绘制连接线
    for i in range(adj_matrix.shape[1]):
        y0, x0 = centers[adj_matrix[0, i], :]
        y1, x1 = centers[adj_matrix[1, i], :]
        line = mlines.Line2D([x0, x1], [y0, y1], color='yellow', linewidth=1.5, alpha=0.6)
        ax.add_line(line)
        draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill='yellow', width=2)

    # 绘制红色点
    radius = 5
    for coord in centers:
        y, x = coord
        upper_left = (x - radius, y - radius)
        lower_right = (x + radius, y + radius)
        draw.ellipse([upper_left, lower_right], fill=(45, 128, 167))

    # 展示并保存原图像
    plt.show()
    g_image.show()
    # g_image.save('superpixel2graph_original.png')

    # 创建白色背景的图像
    width, height = img.shape[1], img.shape[0]
    white_bg_image = Image.new("RGB", (width, height), "white")
    draw_white_bg = ImageDraw.Draw(white_bg_image)

    # 在白色背景图像上绘制连接线
    for i in range(adj_matrix.shape[1]):
        y0, x0 = centers[adj_matrix[0, i], :]
        y1, x1 = centers[adj_matrix[1, i], :]
        draw_white_bg.line([(int(x0), int(y0)), (int(x1), int(y1))], fill='yellow', width=2)

    # 在白色背景图像上绘制红色点
    for coord in centers:
        y, x = coord
        upper_left = (x - radius, y - radius)
        lower_right = (x + radius, y + radius)
        draw_white_bg.ellipse([upper_left, lower_right], fill=(45, 128, 167))

    # 展示并保存白色背景图像
    white_bg_image.show()
    # white_bg_image.save('superpixel2graph_white_bg.png')

    return g_image, white_bg_image  # 返回两张图像


def tensorboard_vis(log_folder, model, dummy_input, device):
    model_name = str(model)  # 或者其他获取字符串表示的方法
    log_folder = os.path.join(log_folder, "./log")
    with SummaryWriter(log_folder) as writer:
        dummy_input = dummy_input.to(device)
        writer.add_graph(model, input_to_model=dummy_input)  # 使用tensorboard进行模型可视化
        # writer.add_image('acc', )


def get_one_superpixel(rgb, segments, label_img):  # 说明：展示图上需要截取标号100的超像素时，输入的index应该是98
    # 使用skimage库的mark_boundaries函数在RGB图像上叠加超像素边界
    # segments = segments - 1
    boundary_marked_image = mark_boundaries(rgb, segments, color=(0, 1, 1))

    # 使用 mark_boundaries 函数获取单像素宽度的边界
    # boundary_image = mark_boundaries(rgb, segments)
    #
    # # 将边界部分提取出来
    # boundary_mask = find_boundaries(segments, mode='thick')
    #
    # # 膨胀操作增加边界线的粗细
    # thick_boundary_mask = dilation(boundary_mask, square(5))  # 使用 line_width 参数调整边界线粗细
    #
    # # 创建一个副本用于显示结果
    # boundary_marked_image = rgb.copy()
    #
    # # 将膨胀后的边界线涂成黄色
    # boundary_marked_image[thick_boundary_mask] = [0, 255, 255]  # 黄色 RGB 值为 (255, 255, 0)

    # 将 boundary_marked_image 转换为 PIL 图像对象
    image = Image.fromarray((boundary_marked_image * 255).astype(np.uint8))
    image.save(r'C:\\Users\\417服务器\\Desktop\\segments.png')
    draw = ImageDraw.Draw(image)

    # 计算每个segment的中心位置
    labeled_segments = label(segments)
    regions = regionprops(labeled_segments)
    centers = [region.centroid for region in regions]

    # 在每个中心位置上绘制文本
    for i, center in enumerate(centers):
        draw.text((center[1], center[0]), str(i + 1), fill='red')  # 调整 i+1 以匹配 segment 标签

    # 显示图像
    image.show()

    # 接收用户输入
    index = int(input("Enter a segment index: "))

    # 提取对应的超像素区域
    mask = np.where(segments == index, 1, 0)  # 不再需要 index+1
    # 打印输出该超像素包含的像素数量
    print("Number of pixels in the segment:", np.sum(mask))

    # 找到指定超像素区域的边界
    boundaries = find_boundaries(segments == index, mode='inner')  # 同样不需要 index+1
    # 膨胀边界以增加其厚度
    thick_boundaries = binary_dilation(boundaries, iterations=1)

    # 复制 boundary_marked_image 并将厚边界标记为红色
    specific_boundary_marked_image = boundary_marked_image.copy()
    specific_boundary_marked_image[thick_boundaries] = [1, 0, 0]

    # 将 specific_boundary_marked_image 转换为 PIL 图像对象
    updated_image = Image.fromarray((specific_boundary_marked_image * 255).astype(np.uint8))
    updated_image.save(r'C:\\Users\\417服务器\\Desktop\\se1.png')

    # 创建一个新的图像对象
    segment_image = Image.fromarray((rgb * mask[:, :, None]).astype('uint8'))
    segment_image.show()
    # 将 mask 区域之外的所有像素设置为白色
    segment_image = Image.fromarray(np.where(mask[:, :, None], segment_image, 255))

    # 找到 mask 区域的边界
    region = regions[index + 4]  # 调整为 index-1，因为 regions 列表是从 0 开始
    minr, minc, maxr, maxc = region.bbox

    # 使用边界裁剪图像
    cropped_image = segment_image.crop((minc, minr, maxc, maxr))
    # cropped_image = image.crop((minc, minr, maxc, maxr))
    # 在 cropped_image 基础生成一个 mask 区域值为 1，其他区域值为 0 的二维矩阵
    mask_mat = np.where(mask[minr:maxr, minc:maxc][:, :, None], 1, 0)

    # 显示图像
    cropped_image.show()
    # 保存图像
    cropped_image.save(r'C:\\Users\\417服务器\\Desktop\\superpixel.png')
    return mask_mat

    # white_img.save('superpixel.png')
    # sp_graph = get_graph_list(rgb, segments, label_img)
    # graph_nx_vis(sp_graph[53], 'superpixel_graph.png')


def visualize_matrix(matrix):
    # 计算矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0])

    # 设置节点直径和连接线宽度
    node_diameter = 15
    node_spacing = 10
    line_width = 3

    # 计算画布的宽度和高度
    width = cols * (node_diameter + node_spacing)
    height = rows * (node_diameter + node_spacing)

    # 创建白色背景的画布
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # 遍历矩阵，绘制节点和连接线
    for i in range(rows):
        for j in range(cols):
            # 当前像素值非零，则绘制节点
            if matrix[i][j] != 0:
                # 计算节点的中心坐标
                center_x = j * (node_diameter + node_spacing) + node_diameter // 2
                center_y = i * (node_diameter + node_spacing) + node_diameter // 2
                # 绘制节点
                draw.ellipse((center_x - node_diameter // 2, center_y - node_diameter // 2,
                              center_x + node_diameter // 2, center_y + node_diameter // 2),
                             fill=(45, 128, 167), outline=(45, 128, 167))

                # 绘制相邻节点间的连接线
                if i > 0 and matrix[i - 1][j] != 0:
                    prev_center_y = (i - 1) * (node_diameter + node_spacing) + node_diameter // 2
                    draw.line([(center_x, center_y - node_diameter // 2),
                               (center_x, prev_center_y + node_diameter // 2)],
                              fill="black", width=line_width)
                if j > 0 and matrix[i][j - 1] != 0:
                    prev_center_x = (j - 1) * (node_diameter + node_spacing) + node_diameter // 2
                    draw.line([(center_x - node_diameter // 2, center_y),
                               (prev_center_x + node_diameter // 2, center_y)],
                              fill="black", width=line_width)

    # 展示画布
    image.show()
    image.save(r'C:\Users\417服务器\Desktop\graph.png')


def display_masked_images(folder_path, filename_mapping_path):
    # 读取文件名对照文件
    with open(filename_mapping_path, 'r') as f:
        filename_mapping = {line.strip().split()[1]: line.strip().split()[0] for line in f}

    for k in range(1, 77):
        if k >= 40:
            # 构造文件名
            filename = os.path.join(folder_path, str(k) + '.mat')
            # 加载mat文件
            data = loadmat(filename)
            # 获取rgb和segments的值
            rgb = data['rgb']
            segments = data['segments']
            label = data['label']
            # 将超像素分割边界segments掩膜在rgb上
            masked_rgb = mark_boundaries(rgb, segments, color=(0, 1, 1))
            # 获取原始文件名
            original_filename = filename_mapping.get(str(k), 'Unknown')

            # 将标签叠加在 RGB 图像上
            colors = [(0, 0.6, 1),
                      (0, 0, 0)]  # 标签值为 2 的颜色，蓝色

            overlay_image = label2rgb(label, image=masked_rgb, colors=colors, alpha=0.3, bg_label=0, bg_color=None,
                                      kind='overlay', saturation=1)

            # 将 NumPy 数组转换为 PIL 图像
            pil_image = Image.fromarray((overlay_image * 255).astype(np.uint8))

            # 展示图像
            pil_image.show()
            pil_image.save(os.path.join(folder_path, f'{k}.png'))
            input(f'Press Enter to continue to the next image (Current filename: {original_filename})')
