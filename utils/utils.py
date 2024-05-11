# @Author   : ChaoQiezi
# @Time     : 2023/12/14  6:33
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 存放常用工具
"""

import h5py
import torch
from torch.utils.data import Dataset

torch.manual_seed(42)  # 固定种子

class H5DatasetDecoder(Dataset):
    """
    对存储特征项和目标项的HDF5文件进行解析，用于后续的数据集加载训练
    """
    def __init__(self, file_path, shuffle_feature_ix=None, dynamic=True):
        self.file_path = file_path
        self.shuffle_feature_ix = shuffle_feature_ix
        self.dynamic = dynamic

        # 获取数据集样本数
        with h5py.File(file_path, mode='r') as h5:
            self.length = h5['static_features1'].shape[0]
            self.targets = h5['targets'][:]  #  (12, 138488)
            self.dynamic_features = h5['dynamic_features'][:]  #  (12, 138488, 6)
            self.static_features1 = h5['static_features1'][:]  #  (138488,)
            self.static_features2 = h5['static_features2'][:]  #  (138488,)
            self.static_features3 = h5['static_features3'][:]  #  (138488,)
            self.static_features4 = h5['static_features4'][:]  #  (138488,)

            if self.shuffle_feature_ix is not None:
                shuffled_indices = torch.randperm(self.length)
                if self.dynamic:
                    # 乱序索引
                    self.dynamic_features[:, :, self.shuffle_feature_ix] =\
                        self.dynamic_features[:, shuffled_indices, self.shuffle_feature_ix]
                elif self.shuffle_feature_ix == 0:  # 静态的
                    self.static_features1 = self.static_features1[shuffled_indices]
                elif self.shuffle_feature_ix == 1:
                    self.static_features2 = self.static_features2[shuffled_indices]
                elif self.shuffle_feature_ix == 2:
                    self.static_features3 = self.static_features3[shuffled_indices]
                elif self.shuffle_feature_ix == 3:
                    self.static_features4 = self.static_features4[shuffled_indices]


    def __len__(self):
        """
        返回数据集的总样本数
        :return:
        """

        return self.length

    def __getitem__(self, index):
        """
        依据索引索引返回一个样本
        :param index:
        :return:
        """

        dynamic_feature = self.dynamic_features[:, index, :]
        static_features1 = self.static_features1[index]
        static_features2 = self.static_features2[index]
        static_features3 = self.static_features3[index]
        static_features4 = self.static_features4[index]
        target = self.targets[:, index]

        static_feature = (static_features1, static_features2, static_features3, static_features4)
        return torch.tensor(dynamic_feature, dtype=torch.float32), \
            torch.tensor(static_feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def cal_r2(outputs, targets):
    """
    计算R2决定系数
    :param outputs:
    :param targets:
    :return:
    """
    mean_predictions = torch.mean(outputs, dim=0, keepdim=True)
    mean_targets = torch.mean(targets, dim=0, keepdim=True)
    predictions_centered = outputs - mean_predictions
    targets_centered = targets - mean_targets
    corr = torch.sum(predictions_centered * targets_centered, dim=0) / \
           (torch.sqrt(torch.sum(predictions_centered ** 2, dim=0)) * torch.sqrt(torch.sum(targets_centered ** 2, dim=0)))

    return torch.mean(corr)