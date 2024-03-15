# @Author   : ChaoQiezi
# @Time     : 2024/1/3  16:54
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 构建lstm模型并训练
"""

import random
import glob
import os.path
import numpy as np
import pandas as pd
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from VEG.utils.utils import H5DatasetDecoder, cal_r2
from VEG.utils.models import LSTMModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 准备
train_path = r'E:\FeaturesTargets\train.h5'
eval_path = r'E:\FeaturesTargets\eval.h5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_model_dir = r'E:\Models'
dynamic_features_name = [
    'LST_MAX',
    'PRCP',
    'ET',
    'Qs',
    'Qsb',
    'TWSC'
]
static_feature_name = [
    'Slope',
    'DEM'
]
# 创建LSTM模型实例并移至GPU
model = LSTMModel(6, 256, 4, 12).to('cuda' if torch.cuda.is_available() else 'cpu')
summary(model, input_data=[(12, 6), (2,)])
batch_size = 256

# generator = torch.Generator().manual_seed(42)  # 指定随机种子
# train_dataset, eval_dataset, sample_dataset = random_split(dataset, (0.8, 0.195, 0.005), generator=generator)
# train_dataset, eval_dataset = random_split(dataset, (0.8, 0.2), generator=generator)
# 创建数据加载器
train_dataset = H5DatasetDecoder(train_path)  # 创建自定义数据集实例
eval_dataset = H5DatasetDecoder(eval_path)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
# 训练参数
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # 初始学习率设置为0.001
epochs_num = 30
model.train()  # 切换为训练模式


def model_train(data_loader, feature_ix: int = None, epochs_num: int = 25, dynamic: bool = True,
                save_path: str = None, device='cuda'):
    # 创建新的模型实例
    model = LSTMModel(6, 256, 4, 12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 初始学习率设置为0.001
    epochs_loss = []
    for epoch in range(epochs_num):
        train_loss = []
        for dynamic_inputs, static_inputs, targets in data_loader:
            # if feature_ix is not None:
            #     if dynamic:
            #         batch_size, _, _ = dynamic_inputs.shape
            #         shuffled_indices = torch.randperm(batch_size)
            #         # dynamic_inputs[:, :, feature_ix] = torch.tensor(np.random.permutation(dynamic_inputs[:, :, feature_ix]))
            #         dynamic_inputs[:, :, feature_ix] = torch.tensor(dynamic_inputs[shuffled_indices, :, feature_ix])
            #     else:
            #         batch_size, _ = static_inputs.shape
            #         shuffled_indices = torch.randperm(batch_size)
            #         # static_inputs[:, feature_ix] = torch.tensor(np.random.permutation(static_inputs[shuffled_indices, feature_ix]))
            #         static_inputs[:, feature_ix] = torch.tensor(static_inputs[shuffled_indices, feature_ix])
            dynamic_inputs, static_inputs, targets = dynamic_inputs.to(device), static_inputs.to(device), targets.to(
                device)

            """正常"""
            # 前向传播
            outputs = model(dynamic_inputs, static_inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            # scheduler.step()  # 更新学习率

            optimizer.zero_grad()  # 清除梯度
            train_loss.append(loss.item())
        print(f'Epoch {epoch + 1}/{epochs_num}, Loss: {np.mean(train_loss)}')
        epochs_loss.append(np.mean(train_loss))

    if save_path:
        torch.save(model.state_dict(), save_path)

    return epochs_loss


def model_eval_whole(model_path: str, data_loader, device='cuda'):
    # 加载模型
    model = LSTMModel(6, 256, 4, 12).to(device)
    model.load_state_dict(torch.load(model_path))

    # 评估
    model.eval()  # 评估模式
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for dynamic_inputs, static_inputs, targets in data_loader:
            dynamic_inputs, static_inputs, targets = dynamic_inputs.to(device), static_inputs.to(device), targets.to(
                device)
            outputs = model(dynamic_inputs, static_inputs)
            all_outputs.append(outputs.cpu())  # outputs/targets: (batch_size, time_steps)
            all_targets.append(targets.cpu())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # mse_per_step = []
    # mae_per_step = []
    # r2_per_step = []
    # rmse_per_step = []
    # for time_step in range(12):
    #     mse_step = mean_squared_error(all_targets[:, time_step], all_outputs[:, time_step])
    #     mae_step = mean_absolute_error(all_targets[:, time_step], all_outputs[:, time_step])
    #     r2_step = r2_score(all_targets[:, time_step], all_outputs[:, time_step])
    #     rmse_step = np.sqrt(mse_step)
    #
    #     mse_per_step.append(mse_step)
    #     mae_per_step.append(mae_step)
    #     r2_per_step.append(r2_step)
    #     rmse_per_step.append(rmse_step)

    # mse = np.mean(mse_per_step)
    # mae = np.mean(mae_per_step)
    # r2 = np.mean(r2_per_step)
    # rmse = np.mean(rmse_per_step)

    # 不区分月份求取指标(视为整体)
    mse_step = mean_squared_error(all_targets.reshape(-1), all_outputs.reshape(-1))
    mae_step = mean_absolute_error(all_targets.reshape(-1), all_outputs.reshape(-1))
    r2_step = r2_score(all_targets.reshape(-1), all_outputs.reshape(-1))
    rmse_step = np.sqrt(mse_step)
    return mse_step, mae_step, r2_step, rmse_step

    # return mse_per_step, mae_per_step, r2_per_step, rmse_per_step, all_outputs, all_targets



def model_eval(model_path: str, data_loader, device='cuda'):
    # 加载模型
    model = LSTMModel(6, 256, 4, 12).to(device)
    model.load_state_dict(torch.load(model_path))

    # 评估
    model.eval()  # 评估模式
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for dynamic_inputs, static_inputs, targets in data_loader:
            dynamic_inputs, static_inputs, targets = dynamic_inputs.to(device), static_inputs.to(device), targets.to(
                device)
            outputs = model(dynamic_inputs, static_inputs)
            all_outputs.append(outputs.cpu())  # outputs/targets: (batch_size, time_steps)
            all_targets.append(targets.cpu())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse_per_step = []
    mae_per_step = []
    r2_per_step = []
    rmse_per_step = []
    for time_step in range(12):
        mse_step = mean_squared_error(all_targets[:, time_step], all_outputs[:, time_step])
        mae_step = mean_absolute_error(all_targets[:, time_step], all_outputs[:, time_step])
        r2_step = r2_score(all_targets[:, time_step], all_outputs[:, time_step])
        rmse_step = np.sqrt(mse_step)

        mse_per_step.append(mse_step)
        mae_per_step.append(mae_step)
        r2_per_step.append(r2_step)
        rmse_per_step.append(rmse_step)

    return mse_per_step, mae_per_step, r2_per_step, rmse_per_step, all_outputs, all_targets

if __name__ == '__main__':
    # df = pd.DataFrame()
    # # 常规训练
    # df['normal_epochs_loss'] = model_train(train_data_loader, save_path=os.path.join(out_model_dir, 'normal_model.pth'))
    # print('>>> 常规训练结束')
    # # 特征重要性训练
    # # 动态特征
    # for feature_ix in range(6):
    #     train_dataset = H5DatasetDecoder(train_path, shuffle_feature_ix=feature_ix, dynamic=True)  # 创建自定义数据集实例
    #     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     cur_feature_name = dynamic_features_name[feature_ix]
    #     save_path = os.path.join(out_model_dir, cur_feature_name + '_model.pth')
    #     df[cur_feature_name + '_epochs_loss'] = \
    #         model_train(train_data_loader, feature_ix, dynamic=True, save_path=save_path)
    #     print('>>> {}乱序排列 训练结束'.format(cur_feature_name))
    # # 静态特征
    # for feature_ix in range(2):
    #     train_dataset = H5DatasetDecoder(train_path, shuffle_feature_ix=feature_ix, dynamic=False)  # 创建自定义数据集实例
    #     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     cur_feature_name = static_feature_name[feature_ix]
    #     save_path = os.path.join(out_model_dir, cur_feature_name + '_model.pth')
    #     df[cur_feature_name + '_epochs_loss'] = \
    #         model_train(train_data_loader, feature_ix, dynamic=False, save_path=save_path)
    #     print('>>> {}乱序排列 训练结束'.format(cur_feature_name))
    # df.to_excel(r'E:\Models\training_eval_results\training_loss.xlsx')

    # 评估
    indicator_whole = pd.DataFrame()
    indicator = pd.DataFrame()
    model_paths = glob.glob(os.path.join(out_model_dir, '*.pth'))
    for model_path in model_paths:
        cur_model_name = os.path.basename(model_path).rsplit('_model')[0]
        mse_step, mae_step, r2_step, rmse_step = model_eval_whole(model_path, eval_data_loader)
        indicator_whole[cur_model_name + '_evaluate_mse'] = [mse_step]
        indicator_whole[cur_model_name + '_evaluate_mae'] = [mae_step]
        indicator_whole[cur_model_name + '_evaluate_r2'] = [r2_step]
        indicator_whole[cur_model_name + '_evaluate_rmse'] = [rmse_step]

        mse_per_step, mae_per_step, r2_per_step, rmse_per_step, all_outputs, all_targets = model_eval(model_path, eval_data_loader)

        all_outputs_targets = np.concatenate((all_outputs, all_targets), axis=1)
        columns = [*['outputs_{:02}'.format(month) for month in range(1, 13)], *['targets_{:02}'.format(month) for month in range(1, 13)]]
        outputs_targets = pd.DataFrame(all_outputs_targets, columns=columns)
        indicator[cur_model_name + '_evaluate_mse'] = mse_per_step
        indicator[cur_model_name + '_evaluate_mae'] = mae_per_step
        indicator[cur_model_name + '_evaluate_r2'] = r2_per_step
        indicator[cur_model_name + '_evaluate_rmse'] = rmse_per_step
        outputs_targets.to_excel(r'E:\Models\training_eval_results\{}_outputs_targets.xlsx'.format(cur_model_name))
        print('>>> {} 重要性评估完毕'.format(cur_model_name))
    indicator.loc['均值指标'] = np.mean(indicator, axis=0)
    indicator.to_excel(r'E:\Models\training_eval_results\eval_indicators_整体.xlsx')
    indicator_whole.to_excel(r'E:\Models\training_eval_results\eval_indicators_整体.xlsx')
    # model.eval()
    # eval_loss = []
    # with torch.no_grad():
    #     for dynamic_inputs, static_inputs, targets in data_loader:
    #         dynamic_inputs = dynamic_inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    #         static_inputs = static_inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
    #         targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')
    #         # 前向传播
    #         outputs = model(dynamic_inputs, static_inputs)
    #         # 计算损失
    #         loss = criterion(outputs, targets)
    #         r2 = cal_r2(outputs, targets)
    #         print('预测项:', outputs)
    #         print('目标项:', targets)
    #         print(f'MSE Loss: {loss.item()}')
    #         break
    #         eval_loss.append(loss.item())
    # print(f'Loss: {np.mean(eval_loss)}')
    # print(f'R2:', r2)



# # 取
# with h5py.File(r'E:\FeaturesTargets\features_targets.h5', 'r') as h5:
#     features = np.transpose(h5['2003/features1'][:], (1, 0, 2))  # shape=(样本数, 时间步, 特征项)
#     targets = np.transpose(h5['2003/targets'][:], (1, 0))  # shape=(样本数, 时间步)
#     static_features = np.column_stack((h5['2003/features2'][:], h5['dem'][:]))
#     mask1 = ~np.any(np.isnan(features), axis=(1, 2))
#     mask2 = ~np.any(np.isnan(targets), axis=(1,))
#     mask3 = ~np.any(np.isnan(static_features), axis=(1, ))
#     mask = (mask1 & mask2 & mask3)
#     features = features[mask, :, :]
#     targets = targets[mask, :]
#     static_features = static_features[mask, :]
#     print(features.shape)
#     print(targets.shape)
# for ix in range(6):
#     feature = features[:, :, ix]
#     features[:, :, ix] = (feature - feature.mean()) / feature.std()
#     if ix <= 1:
#         feature = static_features[:, ix]
#         static_features[:, ix] = (feature - feature.mean()) / feature.std()
#
# features_tensor = torch.tensor(features, dtype=torch.float32)
# targets_tensor = torch.tensor(targets, dtype=torch.float32)
# static_features_tensor = torch.tensor(static_features, dtype=torch.float32)
#
# # 创建包含动态特征、静态特征和目标的数据集
# dataset = TensorDataset(features_tensor, static_features_tensor, targets_tensor)
# train_dataset, eval_dataset = random_split(dataset, [8000, 10238 - 8000])