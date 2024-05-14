# @Author   : ChaoQiezi
# @Time     : 2024/1/19  3:12
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 包括数据集的整合以支持输入到模型中训练，以及特征工程

各个数据集的时间范围:

Landuse: 2001 - 2020
LST(MEAN/MIN/MAX): 200002 - 202210
NDVI(MEAN/MIN/MAX): 200002 - 202010
ET: 200204 - 202309
PRCP: 200204 - 202309
Qs: 200204 - 202309
Qsb: 200204 - 202309
TWSC: 200204 - 202309
dem: single

输出的nc文件的数据格式:
- group(year)
    - features1 -> (None, time_step, features_count) , eg. (184, 139, 12 or other, 6)
        7: LST, PRCP, ET, Qs, Qsb, TWSC
    - features2 -> (None, ), Landuse, (184 * 139)
    - targets-> (Noner, time_step), NDVI, (184 * 139, 12)
- features3 -> dem

2024/5/11 新增关于Rs地表太阳辐射和经纬度数据集的添加
由于Rs时间范围为1983-2017年6月份, 因此此处公共部分的使用日期缩短到2016年12月份.

2024/5/11 仅使用LST,PRCP,ET,RS四个变量进行特征构建
"""

from datetime import datetime
import os
import re
from glob import glob

import netCDF4 as nc
import numpy as np
from osgeo import gdal
import h5py
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale


def read_img(img_path):
    """
    读取栅格文件的波段数据集
    :param img_path: 待读取栅格文件的路径
    :return: 波段数据集
    """
    img = gdal.Open(img_path)
    band = np.float32(img.GetRasterBand(1).ReadAsArray())
    no_data_value = img.GetRasterBand(1).GetNoDataValue()
    band[band == no_data_value] = np.nan

    return band


# 准备
in_dir = r'E:\FeaturesTargets\uniform'
h5_path = r'E:\FeaturesTargets\features_targets.h5'
dem_path = r'E:\FeaturesTargets\uniform\dem.tiff'
slope_path = r'E:\FeaturesTargets\uniform\slope.tif'
lon_path = r'E:\FeaturesTargets\uniform\Lat.tiff'
lat_path = r'E:\FeaturesTargets\uniform\Lat.tiff'
start_date = datetime(2003, 1, 1)
end_date = datetime(2016, 12, 1)
features1_params = {
    'LST_MAX': 'LST_MAX_',
    # 'LST_MIN': 'LST_MIN_',
    # 'LST_MEAN': 'LST_MEAN_',
    'PRCP': 'GLDAS_PRCP_',
    'ET': 'GLDAS_ET_',
    # 'Qs': 'GLDAS_Qs_',
    # 'Qsb': 'GLDAS_Qsb_',
    # 'TWSC': 'GLDAS_TWSC_',
    'Rs': 'Rs_'
}
rows = 132
cols = 193
features1_size = len(features1_params)

# 特征处理和写入
h5 = h5py.File(h5_path, mode='w')
for year in range(start_date.year, end_date.year + 1):
    start_month = start_date.month if year == start_date.year else 1
    end_month = end_date.month if year == end_date.year else 12

    features1 = []  # 存储动态特征
    targets = []
    cur_group = h5.create_group(str(year))
    for month in range(start_month, end_month + 1):
        # 当前月份特征项的读取
        cur_features = np.empty((rows, cols, features1_size))
        for ix, (parent_folder_name, feature_wildcard) in enumerate(features1_params.items()):
            cur_in_dir = os.path.join(in_dir, parent_folder_name)
            pattern = re.compile(feature_wildcard + r'{:04}_?{:02}\.tiff'.format(year, month))
            feature_paths = [_path for _path in os.listdir(cur_in_dir) if pattern.match(_path)]
            if len(feature_paths) != 1:
                raise NameError('文件名错误, 文件不存在或者指定文件存在多个')
            feature_path = os.path.join(cur_in_dir, feature_paths[0])
            cur_features[:, :, ix] = read_img(feature_path)
        features1.append(cur_features.reshape(-1, features1_size))
        # 当前月份目标项的读取
        ndvi_paths = glob(os.path.join(in_dir, 'NDVI_MAX', 'NDVI_MAX_{:04}_{:02}.tiff'.format(year, month)))
        if len(ndvi_paths) != 1:
            raise NameError('文件名错误, 文件不存在或者指定文件存在多个')
        ndvi_path = ndvi_paths[0]
        cur_ndvi = read_img(ndvi_path)
        targets.append(cur_ndvi.reshape(-1))
    features1 = np.array(features1)
    targets = np.array(targets)

    """这里不使用土地利用数据，改用slope数据"""
    # landuse_paths = glob(os.path.join(in_dir, 'Landuse', 'Landuse_{}.tiff'.format(year)))
    # if len(landuse_paths) != 1:
    #     raise NameError('文件名错误, 文件不存在或者指定文件存在多个')
    # landuse_path = landuse_paths[0]
    # features2 = read_img(landuse_path).reshape(-1)

    cur_group['features1'] = features1
    # cur_group['features2'] = features2
    cur_group['targets'] = targets
    print('目前已处理: {}'.format(year))

h5['dem'] = read_img(dem_path).reshape(-1)
h5['slope'] = read_img(slope_path).reshape(-1)  # 添加slope数据作为特征项
h5['lon'] = read_img(lon_path).reshape(-1)
h5['lat'] = read_img(lat_path).reshape(-1)
if np.isnan(h5['lon']).any() or np.isnan(h5['lat']).any():
    raise RuntimeWarning("Lon/Lat 存在无效值!")
h5.flush()
h5.close()
h5 = None

# 进一步处理，混合所有年份的数据(无需分组)
with h5py.File(h5_path, mode='a') as h5:
    year_dem = h5['dem']
    year_slope = h5['slope']
    year_lon = h5['lon']
    year_lat = h5['lat']
    for year in range(start_date.year, end_date.year + 1):
        year_features1 = h5[r'{}/features1'.format(year)]  # 这里导致的重大错误: year_features1 = h5[r'2003/features1']
        # year_features2 = h5[r'2003/features2']
        year_targets = h5[r'{}/targets'.format(year)]  # Here too

        mask = np.all(~np.isnan(year_features1), axis=(0, 2)) & \
               ~np.isnan(year_slope) & \
               np.all(~np.isnan(year_targets), axis=0) & \
               ~np.isnan(year_dem)
        h5['{}/mask'.format(year)] = mask
        if year == 2003:
            features1 = year_features1[:, mask, :]
            slope = year_slope[mask]
            targets = year_targets[:, mask]
            dem = year_dem[mask]
            lon = year_lon[mask]
            lat = year_lat[mask]
        else:
            features1 = np.concatenate((features1, year_features1[:, mask, :]), axis=1)
            slope = np.concatenate((slope, year_slope[mask]), axis=0)
            targets = np.concatenate((targets, year_targets[:, mask]), axis=1)
            dem = np.concatenate((dem, year_dem[mask]), axis=0)
            lon = np.concatenate((lon, year_lon[mask]), axis=0)
            lat = np.concatenate((dem, year_lat[mask]), axis=0)

    # 归一化
    scaler = StandardScaler()
    for month in range(12):
        features1[month, :, :] = scaler.fit_transform(features1[month, :, :])
    dem = scaler.fit_transform(dem.reshape(-1, 1)).ravel()
    slope = scaler.fit_transform(slope.reshape(-1, 1)).ravel()
    lon = scaler.fit_transform(lon.reshape(-1, 1)).ravel()
    lat = scaler.fit_transform(lat.reshape(-1, 1)).ravel()

    sample_size = dem.shape[0]
    train_amount = int(sample_size * 0.8)
    eval_amount = sample_size - train_amount
# 创建数据集并存储训练数据
with h5py.File(r'E:\FeaturesTargets\train.h5', mode='w') as h5:
    h5.create_dataset('dynamic_features', data=features1[:, :train_amount, :])
    h5.create_dataset('static_features1', data=slope[:train_amount])  # 静态变量
    h5.create_dataset('static_features2', data=dem[:train_amount])  # 静态变量
    h5.create_dataset('static_features3', data=lon[:train_amount])  # 静态变量
    h5.create_dataset('static_features4', data=lat[:train_amount])  # 静态变量
    h5.create_dataset('targets', data=targets[:, :train_amount])
with h5py.File(r'E:\FeaturesTargets\eval.h5', mode='w') as h5:
    # # # 创建数据集并存储评估数据
    h5.create_dataset('dynamic_features', data=features1[:, train_amount:, :])
    h5.create_dataset('static_features1', data=slope[train_amount:])  # 静态变量
    h5.create_dataset('static_features2', data=dem[train_amount:])  # 静态变量
    h5.create_dataset('static_features3', data=lon[train_amount:])  # 静态变量
    h5.create_dataset('static_features4', data=lat[train_amount:])  # 静态变量
    h5.create_dataset('targets', data=targets[:, train_amount:])
