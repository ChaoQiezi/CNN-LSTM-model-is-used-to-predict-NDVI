# @Author   : ChaoQiezi
# @Time     : 2024/3/11  18:58
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 是用来绘图滴，主要是箱线图和核密度散点图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from osgeo import gdal
from matplotlib.colors import LinearSegmentedColormap

# 准备
in_path = r'H:\Datasets\Objects\Veg\Plot\cor_by_st.csv'
dem_path = r'H:\Datasets\Objects\Veg\DEM\dem_1km.tif'
out_dir =r'H:\Datasets\Objects\Veg\Plot'
sns.set_style('darkgrid')  # 设置风格
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 允许负号正常显示

# 加载数据
df = pd.read_csv(in_path)
dem = gdal.Open(dem_path)
dem_raster = dem.GetRasterBand(1).ReadAsArray()  # 获取dem栅格矩阵
dem_nodata_value = dem.GetRasterBand(1).GetNoDataValue()  # 获取无效值
lon_ul, lon_res, _, lat_ul, _, lat_res_negative = dem.GetGeoTransform()  # [左上角经度, 经度分辨率, 旋转角度, 左上角纬度, 旋转角度, -纬度分辨率]
lat_res = -lat_res_negative
# 删除TWSC列, 将TWSC_SH列标签换为TWSC
df.drop(['TWSC', 'TWSC_1', 'TWSC_2', 'TWSC_3'], axis=1, inplace=True)
df.rename(columns={'TWSC_SH': 'TWSC', 'TWSC_SH_1': 'TWSC_1', 'TWSC_SH_2': 'TWSC_2', 'TWSC_SH_3': 'TWSC_3'}, inplace=True)
iter_columns_name = df.columns[4:]
# 色带
colors = ['#ff0000', '#ff6f00', '#fbb700', '#cdff00', '#a1ff6e', '#52ffc7', '#00ffff', '#15acff', '#4261ff', '#3100fe']
colors.reverse()
cm = LinearSegmentedColormap.from_list('common', colors, 100)

# 添加DEM列
cols = np.floor((df['Lon'] - lon_ul) / lon_res).astype(int)
rows = np.floor((lat_ul - df['Lat']) / lat_res).astype(int)
df['DEM'] = dem_raster[rows, cols]
df[df['DEM'] == dem_nodata_value] = np.nan
# 绘制散点核密度图
for column_name in iter_columns_name:
    plt.figure(dpi=200)
    cur_ds = df[['DEM', column_name]].dropna(how='any')
    cur_ds['Density'] = gaussian_kde(cur_ds[column_name])(cur_ds[column_name])

    scatter = plt.scatter(x='DEM', y=column_name, c='Density', cmap=cm, linewidth=0, data=cur_ds, s=20)
    clb = plt.colorbar(scatter)
    clb.ax.set_title('Density', fontsize=8)  # 为色带添加标题
    # sns.kdeplot(x='DEM', y=column_name, fill=True, data=cur_ds, alpha=0.6)
    sns.kdeplot(x='DEM', y=column_name, fill=False, color='gray', data=cur_ds, alpha=0.6)
    title_name = 'Scatter kernel density map of $R^2$ \n between NDVI and {} under DEM'.format(column_name)
    plt.title(title_name, fontsize=16)
    plt.xlabel('DEM(m)', fontsize=14)
    plt.ylabel('$R^2$ between NDVI and {}'.format(column_name), fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # 设置XY轴起始值
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(out_dir, 'R2_{}.png'.format(column_name)), dpi=200)
    # plt.show()
    print('处理: {}'.format(column_name))
# 绘制箱线图
meanprops = {"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"}
fig, axs = plt.subplots(4, 1, figsize=(13, 18), dpi=432)
axs = axs.flatten()
fig.suptitle('Box plot of NDVI and correlation coefficients of each variable', fontsize=30, va='top')
for ix, ax in enumerate(axs):
    # print(iter_columns_name[(ix * 9):((ix + 1) * 9)])
    # ax.figure(figsize=(26, 9), dpi=321)
    df_melt = pd.melt(df, value_vars=iter_columns_name[(ix * 8):((ix + 1) * 8)]).dropna(how='any')
    sns.boxplot(data=df_melt, x='variable', y='value', palette=cm(np.linspace(0, 1, 9)), ax=ax, linewidth=3,
                showmeans=True, meanprops=meanprops)
    ax.set_xlabel('', fontsize=25)
    ax.set_ylabel('$R^2$', fontsize=25)
    ax.tick_params(axis='x', labelsize=18)  # x轴标签旋转90度
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True)
plt.tight_layout(pad=2)
fig.savefig(os.path.join(out_dir, 'Box_R2.png'))
# plt.show()


# 用于看
fig, axs = plt.subplots(4, 1, figsize=(13, 18), dpi=432)
axs = axs.flatten()
fig.suptitle('Box plot of NDVI and correlation coefficients of each variable', fontsize=30, va='top')

for ix, ax in enumerate(axs):
    columns_slice = iter_columns_name[(ix * 9):((ix + 1) * 9)]
    df_melt = pd.melt(df, value_vars=columns_slice).dropna(how='any')
    # sns.boxplot(data=df_melt, x='variable', y='value', palette=cm(np.linspace(0, 1, 9)), ax=ax, linewidth=3,
    #             showmeans=True, meanprops=meanprops)
    sns.boxplot(data=df_melt, x='variable', y='value', palette=cm(np.linspace(0, 1, 9)), ax=ax, linewidth=3)

    # 循环每个变量，计算最大值、最小值和平均值，然后在图上标注
    for i, column in enumerate(columns_slice):
        subset = df[column].dropna()
        max_val = subset.max()
        min_val = subset.min()
        mean_val = subset.mean()

        # 标注最大值、最小值和平均值
        ax.text(i, max_val, f'{max_val:.2f}', ha='center', va='bottom', fontsize=16, rotation=45)
        ax.text(i, min_val, f'{min_val:.2f}', ha='center', va='top', fontsize=16, rotation=45)
        ax.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='center', fontsize=16, color='white', rotation=45)

    ax.set_xlabel('', fontsize=25)
    ax.set_ylabel('$R^2$', fontsize=25)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True)

plt.tight_layout(pad=2)
fig.savefig(os.path.join(out_dir, 'Box_R2_quick.png'))
# plt.show()