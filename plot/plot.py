import matplotlib.pyplot as plt

# 初始化存储历史数据的列表
loss_history = []
psnr_history = []
ssim_history = []

def plot_metrics(data_dict, epoch, max_epochs, save_path):
    global loss_history, psnr_history, ssim_history

    # 从字典中获取数据并添加到历史数据列表中
    loss_history.append(data_dict['loss'])
    psnr_history.append(data_dict['PSNR'])
    ssim_history.append(data_dict['SSIM'])

    # 清除当前图表
    plt.clf()

    def plot_subgraph(data, index, label, ylabel, color):
        plt.subplot(2, 2, index)
        plt.plot(data, label=label, color=color)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        for i, value in enumerate(data):
            if (i + 1) % 5 == 0:
                plt.text(i, value, f'{value:.2f}')

    # 绘制 Loss 子图
    plot_subgraph(loss_history, 1, 'Loss', 'Loss', 'red')

    # 绘制 PSNR 子图
    plot_subgraph(psnr_history, 2, 'PSNR', 'PSNR', 'blue')

    # 绘制 SSIM 子图
    plot_subgraph(ssim_history, 3, 'SSIM', 'SSIM', 'green')

    # 调整子图间距
    plt.tight_layout()

    # 如果是最后一个epoch，保存图像
    if epoch == max_epochs:
        plt.savefig(save_path)

    plt.pause(0.01)