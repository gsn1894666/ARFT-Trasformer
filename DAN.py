from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import mmd
import data_loader
from torch.utils import model_zoo
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from visualization import plot_evaluation_metrics, plot_train_loss, plot_test_loss
from Focal_Loss import FocalLoss
from FT_MLP import ARFTransformer
import pandas as pd  # 导入 pandas 用于表格保存
from plot_distribution import plot_sample_distribution


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 训练参数
batch_size =32
iteration = 1000
lr = 0.001
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10                                                    #训练日志的迭代间隔
l2_decay = 5e-4

# 数据集路径（修改为 CSV 文件名）
root_path = "./danyuanDS/"
src_name = "Linux.csv"  # 源域 CSV 文件
tgt_name = "MySQL.csv"    # 目标域 CSV 文件
el_name = "Httpd.csv"   #剩余文件
cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# 计算全局均值和标准差
global_mean, global_std = data_loader.compute_global_stats(root_path, src_name, tgt_name)
# 加载数据（使用 CSV 文件）
src_loader = data_loader.load_training_with_ros(root_path, src_name, batch_size, kwargs, global_mean, global_std)
tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs, global_mean, global_std)
# 添加未过采样的源域加载器
raw_src_loader = data_loader.load_raw_training(root_path, src_name, batch_size, kwargs, global_mean, global_std)

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_test_loader)
raw_src_dataset_len = len(raw_src_loader.dataset)

# 训练函数
def train(model):
    tgt_test_data = None  # 先赋默认值，防止未定义错误

    # 获取 tgt_test_loader 的数据
    for data in tgt_test_loader:
        tgt_test_data = data[0]  # 输入特征
        tgt_test_label = data[1]  # 真实标签
        print("tgt_test_data:", tgt_test_data.shape)
        print("tgt_test_label:", tgt_test_label.shape)
        break  # 只取第一个 batch，避免 DataLoader 过度消耗

    # 检查 tgt_test_data 是否成功赋值
    if tgt_test_data is None:
        print("❌ 错误: tgt_test_data 仍未赋值！请检查 tgt_test_loader 是否为空！")
        return  # 终止训练

    print("Checking if tgt_test_label exists before assignment:", 'tgt_test_label' in locals())
    print("✅ 开始训练...")

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_test_loader)                                 #创建数据加载器的迭代器，方便批量获取数据
    correct = 0      #初始化正确预测的次数

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_decay)

    # 为了提高效率，在循环外定义
    focal_loss_fn = FocalLoss(gamma=2, alpha=None)  # Adjust gamma and alpha as needed
    # 添加历史记录列表
    precision_history = []
    recall_history = []        #recall默认是少数类（label=1）
    f1_history = []
    balance_history = []
    iteration_history = []
    pf_history = []

    # 损失历史记录
    train_loss_history = []  # 每 100 次迭代的平均训练损失，共 10 次
    test_loss_history = []  # 每次测试的损失，共 10 次
    temp_train_losses = []  # 临时存储每 100 次的训练损失
    iteration_points = []  # 记录每 100 次迭代的点，例如 [100, 200, ..., 1000]

#循环训练
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10* (i - 1) / iteration), 0.75)        #更新学习率：使用学习率衰减策略逐步降低学习率
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

        if (i - 1) % 100== 0:                                                                    #每一百次循环打印一次学习率
            print(f'🔹 learning rate: {LEARNING_RATE:.4f}')

        # 获取源域数据
        try:
            src_data, src_label = next(src_iter)                                                  #从 src_iter（源域数据加载器的迭代器）获取 src_data（数据）和 src_label（标签）
        except StopIteration:                                                                     #当 src_iter 遍历完 src_loader 里的所有数据后，会抛出 StopIteration。
            src_iter = iter(src_loader)                                                           #处理STop异常：重新创建一个新的迭代器 src_iter = iter(src_loader)，然后再获取 src_data, src_label
            src_data, src_label = next(src_iter)

        # 获取目标域训练数据
        try:
            tgt_data, _ = next(tgt_iter)                                                          #目标域数据通常没有标签（_ 占位）
        except StopIteration:
            tgt_iter = iter(tgt_test_loader)
            tgt_data, _ = next(tgt_iter)

        # 对齐 batch size
        min_batch_size = min(src_data.size(0), tgt_data.size(0))
        src_data = src_data[:min_batch_size]
        src_label = src_label[:min_batch_size]
        tgt_data = tgt_data[:min_batch_size]

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()                               #将数据移动到 GPU，加快计算速度
            tgt_data = tgt_data.cuda()

            # **转换标签为 long 类型**
        src_label = src_label.long()

        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)

        # **计算分类损失**
        cls_loss = focal_loss_fn(src_pred, src_label)

        # 计算一个动态权重（lambd），用于调整分类损失和 MMD 损失的相对重要性：逐渐引入目标域对源域模型的影响
        # 在训练初期，lambd 会较小，意味着 MMD 损失对总损失的贡献较少；随着迭代的进行，lambd 的值增大，MMD 损失的权重逐渐增加。
        lambd = 0.5 / (1 + math.exp(-5 * i / iteration))  # 范围 (0, 0.5)
        loss = cls_loss + lambd * mmd_loss

        # **反向传播**这行代码执行反向传播，根据之前计算的总loss，计算每个参数的梯度
        loss.backward()
        optimizer.step()                                                          #根据计算出的梯度，执行优化步骤。通过这个步骤，模型的参数会更新，以减少损失。

        # 收集每次的训练损失
        temp_train_losses.append(loss.item())

        # **日志输出**每 10次迭代输出一次
        if i % (log_interval *1) == 0:
            print(f'🟢 Train iter: {i} [{100. * i / iteration:.0f}%] '
                  f'Loss: {loss.item():.6f} '                             #总损失
                  f'soft_Loss: {cls_loss.item():.6f} '                    #分类损失
                  f'mmd_Loss: {mmd_loss.item():.6f}')                     #MMD损失

        # **测试模型**每 100次迭代进行一次测试
        if i % (log_interval *10)== 0:
            # 计算并记录 100 次迭代的平均训练损失
            avg_train_loss = np.mean(temp_train_losses)
            train_loss_history.append(avg_train_loss)
            temp_train_losses = []  # 清空临时列表

            # 记录迭代点
            iteration_points.append(i)
            t_correct, iteration_list, precision_list, recall_list, f1_list, balance_list, pf_list, test_loss= test(model)    #t_correct 表示测试集上的正确预测数

            # 测试并记录测试损失
            test_loss_history.append(test_loss)
            # 记录所有指标的平均值
            precision_history.append(np.mean(precision_list))
            recall_history.append(np.mean(recall_list))
            f1_history.append(np.mean(f1_list))
            balance_history.append(np.mean(balance_list))
            pf_history.append(np.mean(pf_list))
            iteration_history.append(i)

            src_precision, src_recall, src_balance = test_source(model)
            print(f'🟡 在源域的测试结果: Precision:{src_precision:.4f}, Recall: {src_recall:.4f}, Balance: {src_balance:.4f}')
            # 训练结束后绘制图表
    # 绘制训练损失和测试集损失曲线(测试过了是十个点)
    plot_train_loss(iteration_points, train_loss_history)
    plot_test_loss(iteration_points,test_loss_history)
        # 绘制所有指标
    plot_evaluation_metrics(iteration_history, precision_history, recall_history, f1_history, balance_history, pf_history)
    # 返回历史记录
    return {
        'precision_history': precision_history,
        'recall_history': recall_history,
        'f1_history': f1_history,
        'balance_history': balance_history,
        'pf_history': pf_history,
    }

def test_source(model):
    """验证模型在源域上的表现"""
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for src_data, src_label in raw_src_loader:         # 使用未过采样数据
            if cuda:
                src_data, src_label = src_data.cuda(), src_label.cuda()
            src_label = src_label.long()

            # 前向传播
            src_pred, _ = model(src_data, None)  # MMD 损失在此不重要
            pred = src_pred.argmax(dim=1)

            # 计算正确预测数
            correct += pred.eq(src_label).sum().item()

            # 记录预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(src_label.cpu().numpy())

    # 计算指标
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)
    # 计算 Balance 指标
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    pf = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误报率
    balance = 1 - (((1 - recall) ** 2 + pf ** 2) ** 0.5) / (2 ** 0.5)  # 使用 PF 计算 balance，与 test 函数一致

    print("源域预测分布:", np.bincount(all_preds))
    return precision, recall, balance


# 测试函数
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    # 存储每个批次的指标，用来计算最后的平均值
    precision_list = []
    recall_list = []
    f1_list = []
    balance_list = []
    pf_list = []  # 误报概率
    iteration_list = []

    #为了提高效率，在循环外定义
    focal_loss_fn = FocalLoss(gamma=2, alpha=None, reduction='sum')
    with torch.no_grad():

        for i, (tgt_test_data, tgt_test_label) in enumerate(tgt_test_loader):
            iteration_list.append(i)  # ✅ 记录当前迭代次数

            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)                         #将数据包装成 Variable，高版本可以不用，不用管

            # **转换标签为 long 类型**
            tgt_test_label = tgt_test_label.long()

            tgt_pred, _ = model(tgt_test_data, None)  # 修改为 None，因为 test 不需要 MMD

            #损失（测试中没有MMD损失）
            batch_loss = focal_loss_fn(tgt_pred, tgt_test_label).item()
            test_loss += batch_loss  # 累加每个 batch 的损失

            # **计算预测正确的数量**
            pred = tgt_pred.argmax(dim=1)  # 获取最大概率的类别
            correct += pred.eq(tgt_test_label).sum().item()  # 计算匹配的个数

            # 将当前批次的预测结果和真实标签存入列表
            all_preds.extend(pred.cpu().numpy())  # .cpu() 将数据移到 CPU，.numpy() 转换为 NumPy 数组
            all_labels.extend(tgt_test_label.cpu().numpy())

            # **计算每个批次的 Precision, Recall, F1, PF, Balance**
            precision = precision_score(tgt_test_label.cpu().numpy(), pred.cpu().numpy(), average='binary',
                                        zero_division=1)
            recall = recall_score(tgt_test_label.cpu().numpy(), pred.cpu().numpy(), average='binary', zero_division=1)
            f1 = f1_score(tgt_test_label.cpu().numpy(), pred.cpu().numpy(), average='binary', zero_division=1)

            # **修正 confusion_matrix 计算**
            cm = confusion_matrix(tgt_test_label.cpu().numpy(), pred.cpu().numpy(), labels=[0, 1])
            # ✅ 调试用    print("Confusion Matrix:\n", cm)
            if cm.shape == (2, 2):  # 确保是二分类
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0  # 避免代码崩溃
            pf = fp / (fp + tn) if (fp + tn) > 0 else 0
            balance = 1 - (((1 - recall) ** 2 + pf ** 2) ** 0.5) / (2 ** 0.5)  # 使用 PF 计算 balance，与 test 函数一致

            # **将批次指标存入列表**
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            balance_list.append(balance)
            pf_list.append(pf)


    # **计算整个测试集的平均损失**
    avg_test_loss = test_loss / len(tgt_test_loader)  # 计算所有 batch 的平均 test loss

    # **计算整个测试集的平均 Precision、Recall、F1、PF、Balance**
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_balance = np.mean(balance_list)
    avg_pf = np.mean(pf_list)

    print("测试集标签分布:", np.bincount(all_preds))
    # **计算平均损失和准确率**

    print(f'\n{tgt_name} 测试集平均结果:  loss: {avg_test_loss:.4f}, '
          f'Precision: {avg_precision:.6f}， Recall: {avg_recall:.6f},  '
          f'Balance: {avg_balance:.6f},  PF: {avg_pf:.6f}, F1: {avg_f1:.6f}\n')

    #返回测试指标
    return correct, iteration_list, avg_precision, avg_recall, avg_f1, avg_balance, avg_pf, avg_test_loss
# 运行主程序
if __name__ == '__main__':
    print("Starting the script...")

    # 绘制样本分布图
    file_names = ['MySQL.csv', 'Linux.csv', 'Httpd.csv']
    plot_sample_distribution(root_path, file_names)
    model = ARFTransformer(
        input_dim=52,
        d_token=128,
        n_layers=3,
        n_heads=8,
        d_ffn=256,
        num_classes=2
    )
    print(model)

    #为了确保只打印一次，所以放main里
    # 目标数据的训练集和测试集没有正确拆分！！！！！！可能对可能不对，有待考察！！！！！！！
    src_train_dataset = src_loader.dataset
    tgt_test_dataset = tgt_test_loader.dataset
    raw_src_dataset = raw_src_loader.dataset
    print(f"src训练集样本数: {len(src_train_dataset)}")
    print(f"tgt测试集样本数: {len(tgt_test_dataset)}")
    print(f"未过采样源域数据样本数: {len(raw_src_dataset)}")
    print(f"未过采样源域数据分布: {np.bincount(raw_src_dataset.labels.astype(int))}")
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()

        # 调用训练函数并获取结果
        results = train(model)

    # 创建表格
    df = pd.DataFrame({
        'Recall': results['recall_history'],
        'Balance': results['balance_history'],
        'PF': results['pf_history'],
        'Precision': results['precision_history'],
        'F1 Score': results['f1_history'],
        })

    # 计算平均值并添加到表格
    avg_row = pd.DataFrame({
        'Recall': [df['Recall'].mean()],
        'Balance': [df['Balance'].mean()],
        'PF': [df['PF'].mean()],
        'Precision': [df['Precision'].mean()],
        'F1 Score': [df['F1 Score'].mean()]
    })

    # 将平均值行追加到表格
    df = pd.concat([df, avg_row], ignore_index=True)

    # 保存为 CSV 文件
    output_file = 'test_results.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"测试结果已保存到 {output_file}")
