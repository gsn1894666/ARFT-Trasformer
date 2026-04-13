import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from data_loader import TableDataset
import os


def plot_sample_distribution(root_path, file_names):
    """
    绘制多个文件的样本分布图，每个样本为一个点，负类蓝点更分散，正类不变，仅第一张图有图例，标题在下方以 (a) filename 形式，无网格，保存到目录。

    Args:
        root_path (str): 数据文件所在目录
        file_names (list): 文件名列表，例如 ['mysql.csv', 'linux.csv', 'httpd.csv']
    """
    # 设置子图布局，保持原图形尺寸
    fig, axes = plt.subplots(1, len(file_names), figsize=(5 * len(file_names), 5))
    if len(file_names) == 1:
        axes = [axes]  # 确保单文件时 axes 是列表

    # 用于生成 (a), (b), (c) 的标签
    title_labels = ['(a)', '(b)', '(c)']

    for idx, file_name in enumerate(file_names):
        # 加载数据集
        dataset = TableDataset(root_path + file_name, global_mean=None, global_std=None)
        features = dataset.features
        labels = dataset.labels.astype(int)

        # 使用 PCA 降维到 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # 标准化 PCA 输出
        mean_2d = np.mean(features_2d, axis=0)
        std_2d = np.std(features_2d, axis=0) + 1e-8  # 避免除零
        features_2d = (features_2d - mean_2d) / std_2d

        # 分离正负类
        neg_indices = labels == 0
        pos_indices = labels == 1

        # 仅对负类（蓝点）应用缩放以更分散
        scale_factor = 2.0  # 负类坐标放大 2 倍
        features_2d_neg = features_2d[neg_indices] * scale_factor
        features_2d_pos = features_2d[pos_indices]  # 正类保持不变

        # 调试：打印坐标范围
        print(
            f"{file_name} Blue points X range: {features_2d_neg[:, 0].min():.2f} to {features_2d_neg[:, 0].max():.2f}")
        print(
            f"{file_name} Blue points Y range: {features_2d_neg[:, 1].min():.2f} to {features_2d_neg[:, 1].max():.2f}")
        print(f"{file_name} Red points X range: {features_2d_pos[:, 0].min():.2f} to {features_2d_pos[:, 0].max():.2f}")
        print(f"{file_name} Red points Y range: {features_2d_pos[:, 1].min():.2f} to {features_2d_pos[:, 1].max():.2f}")

        # 绘制散点图
        ax = axes[idx]
        ax.scatter(
            features_2d_neg[:, 0],
            features_2d_neg[:, 1],
            c='cornflowerblue',
            label='Negative (0)' if idx == 0 else None,  # 仅第一张图有标签
            alpha=0.5,
            s=15
        )
        ax.scatter(
            features_2d_pos[:, 0],
            features_2d_pos[:, 1],
            c='tomato',
            label='Positive (1)' if idx == 0 else None,  # 仅第一张图有标签
            alpha=0.5,
            s=20
        )

        # 设置标题（去掉扩展名，格式为 (a) filename）
        base_name = os.path.splitext(file_name)[0]
        ax.set_title(f'{title_labels[idx]} {base_name}', y=-0.15, fontsize=10)

        # 设置轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 仅第一张图显示图例
        if idx == 0:
            ax.legend()

        # 固定轴范围
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    # 调整布局，增加底部空间以适应下方标题
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # 保存图形到当前目录
    output_path = 'sample_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"分布图已保存到 {output_path}")

    plt.show()