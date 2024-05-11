# @Author   : ChaoQiezi
# @Time     : 2024/3/29  9:47
# @FileName : line_bar_distribution_plot.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 绘制折线图、柱状图、插值分布图

EWTC: 包括人类活动导致的(地表水地下使用加工运输到别的地方等), 自然变化的(蒸腾蒸发降水等)引起的储水量变化
TWSC: 指单独自然变化导致的储水量变化
AWC: (EWTC - TWSC即可得到)人类活动导致引发的储水量变化
"""

import glob
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal

# 准备
ewtc_path = r'H:\Datasets\Objects\Veg\LXB_plot\Data\sw_EWTC_T.csv'
twsc_path = r'H:\Datasets\Objects\Veg\LXB_plot\Data\sw_TWSC_SH_T.csv'
ndvi_path = r'H:\Datasets\Objects\Veg\LXB_plot\Data\sw_NDVI_T.csv'
in_img_dir= r'E:\FeaturesTargets\uniform'
out_dir = r'H:\Datasets\Objects\Veg\LXB_plot'
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 新罗马字体


# 绘制
ewtc = pd.read_csv(ewtc_path)
twsc = pd.read_csv(twsc_path)
ndvi = pd.read_csv(ndvi_path)
awc = pd.merge(ewtc, twsc, left_on='ProvinceNa', right_on='ProvinceNa', suffixes=('_ewtc', '_twsc'))
awc['AWC'] = awc['EWTC'] - awc['TWSC_SH']

# # 年月均值(ewtc twsc ndvi)
# var_str = ['EWTC', 'TWSC_SH', 'AWC']
# for ix, var in enumerate([ewtc, twsc, awc]):
#     var_name = var_str[ix]
#     if var_name == 'AWC':
#         var_monthly = var[['Year_ewtc', 'Month_ewtc', var_name]].groupby(['Year_ewtc', 'Month_ewtc']).mean()
#         var_monthly.to_csv(os.path.join(out_dir, '{}.csv'.format(var_name)))
#     else:
#         var_monthly = var[['Year', 'Month', var_name]].groupby(['Year', 'Month']).mean()
#     var_monthly['Date'] = var_monthly.reset_index().apply(lambda x: '{:04.0f}/{:02.0f}'.format(x.iloc[0], x.iloc[1]), axis=1).values
#     var_monthly.to_csv(os.path.join(out_dir, '{}.csv'.format(var_name)))
# # 月均值
# var_str = ['EWTC', 'TWSC_SH', 'AWC']
# for ix, var in enumerate([ewtc, twsc, awc]):
#     var_name = var_str[ix]
#     if var_name == 'AWC':
#         var_monthly = var[['Month_ewtc', var_name]].groupby(['Month_ewtc']).mean()
#         # var_monthly.to_csv(os.path.join(out_dir, '{}_monthly.csv'.format(var_name)))
#     else:
#         var_monthly = var[['Month', var_name]].groupby(['Month']).mean()
#     var_monthly['Date'] = var_monthly.index
#     # var_monthly['Date'] = var_monthly.reset_index().apply(lambda x: '{:04.0f}/{:02.0f}'.format(x.iloc[0], x.iloc[1]), axis=1).values
#     var_monthly.to_csv(os.path.join(out_dir, '{}_monthly.csv'.format(var_name)), index=False)
# 年均值
var_str = ['EWTC', 'TWSC_SH', 'AWC']
for ix, var in enumerate([ewtc, twsc, awc]):
    var_name = var_str[ix]
    if var_name == 'AWC':
        var_yearly = var[['Year_ewtc', var_name]].groupby(['Year_ewtc']).mean()
        # var_monthly.to_csv(os.path.join(out_dir, '{}_monthly.csv'.format(var_name)))
    else:
        var_yearly = var[['Year', var_name]].groupby(['Year']).mean()
    var_yearly['Date'] = var_yearly.index
    # var_monthly['Date'] = var_monthly.reset_index().apply(lambda x: '{:04.0f}/{:02.0f}'.format(x.iloc[0], x.iloc[1]), axis=1).values
    var_yearly.to_csv(os.path.join(out_dir, '{}_yearly.csv'.format(var_name)), index=False)

# PRCP, Qs, Qsb, ET年均值的计算
prcp_month_path = r'H:\Datasets\Objects\Veg\LXB_plot\PRCP\PRCP.xlsx'
et_month_path = r'H:\Datasets\Objects\Veg\LXB_plot\ET\ET.xlsx'
qs_month_path = r'H:\Datasets\Objects\Veg\LXB_plot\Qs\Qs.xlsx'
qsb_month_path = r'H:\Datasets\Objects\Veg\LXB_plot\Qsb\Qsb.xlsx'
prcp = pd.read_excel(prcp_month_path)
et = pd.read_excel(et_month_path)
qs = pd.read_excel(qs_month_path)
qsb = pd.read_excel(qsb_month_path)
prcp['Year'] = prcp.date.apply(lambda x: x.year)
prcp_yearly = prcp[['Year', 'PRCP']].groupby(['Year']).mean().reset_index(drop=False)
prcp_yearly.to_excel(os.path.join('H:\Datasets\Objects\Veg\LXB_plot\PRCP', 'prcp_yearly.xlsx'), index=False)

et['Year'] = et.date.apply(lambda x: x.year)
et_yearly = et[['Year', 'ET']].groupby(['Year']).mean().reset_index(drop=False)
et_yearly.to_excel(os.path.join('H:\Datasets\Objects\Veg\LXB_plot\ET', 'et_yearly.xlsx'), index=False)

qs['Year'] = qs.date.apply(lambda x: x.year)
qs_yearly = qs[['Year', 'Qs']].groupby(['Year']).mean().reset_index(drop=False)
qs_yearly.to_excel(os.path.join('H:\Datasets\Objects\Veg\LXB_plot\Qs', 'Qs_yearly.xlsx'), index=False)

qsb['Year'] = qsb.date.apply(lambda x: x.year)
qsb_yearly = qsb[['Year', 'Qsb']].groupby(['Year']).mean().reset_index(drop=False)
qsb_yearly.to_excel(os.path.join('H:\Datasets\Objects\Veg\LXB_plot\Qsb', 'Qsb_yearly.xlsx'), index=False)



# 绘制NDVI年变化折线图和月变化柱状图
ndvi_monthly = ndvi[['Month', 'NDVI']].groupby('Month').mean()
ndvi_yearly = ndvi[['Year', 'NDVI']].groupby('Year').mean()
ndvi_yearly['Year'] = ndvi_yearly.index
ndvi_monthly['Month'] = ['{:02}'.format(_x) for _x in ndvi_monthly.index]
ndvi_monthly.to_csv(os.path.join(out_dir, 'ndvi_monthly.csv'), index=False)
ndvi_yearly.to_csv(os.path.join(out_dir, 'ndvi_yearly.csv'), index=False)
# # 绘制折线图
# plt.figure(figsize=(13, 9), dpi=222)
# sns.lineplot(data=ndvi_yearly, x='Year', y='NDVI', linestyle='-', color='#1f77b4', linewidth=7, legend=True)
# plt.scatter(ndvi_yearly['Year'], ndvi_yearly['NDVI'], s=100, facecolors='none', edgecolors='#bcbd22', linewidths=5, zorder=5)
# plt.xlabel('Year', size=26)  # 设置x轴标签
# plt.ylabel('NDVI', size=26)  # 设置y轴标签
# plt.xticks(ndvi_yearly['Year'], rotation=45, fontsize=18)
# plt.yticks(fontsize=22)
# plt.savefig(os.path.join(out_dir, 'ndvi_line_yearly.png'))
# plt.show()
# # 绘制柱状图
# plt.figure(figsize=(13, 9), dpi=222)
# sns.barplot(data=ndvi_monthly, x='Month', y='NDVI', linestyle='-', color='#1f77b4')
# plt.xlabel('Month', size=26)  # 设置x轴标签
# plt.ylabel('NDVI', size=26)  # 设置y轴标签
# x_labels = ndvi_monthly['Month'].apply(lambda x: '{:02}'.format(x))
# plt.xticks(ticks=range(len(x_labels)), labels=x_labels, fontsize=18)
# plt.yticks(fontsize=22)
# plt.savefig(os.path.join(out_dir, 'ndvi_line_monthly.png'))
# plt.show()

# 提取降水、蒸散、地表和地下径流
station = ndvi.drop_duplicates(['Lon', 'Lat'])[['Lon', 'Lat']]
var_names = ['PRCP', 'ET', 'Qs', 'Qsb']
for var_name in var_names:
    var = []
    cur_dir = os.path.join(in_img_dir, var_name)
    var_paths = glob.glob(os.path.join(cur_dir, 'GLDAS_{}*.tiff'.format(var_name)))
    for var_path in var_paths:
        ds = gdal.Open(var_path)
        lon_min, lon_res, _, lat_max, _, lat_res_negative = ds.GetGeoTransform()
        ds_band = np.float32(ds.GetRasterBand(1).ReadAsArray())
        nodata_value = ds.GetRasterBand(1).GetNoDataValue()
        ds_band[ds_band == nodata_value] = np.nan
        station['row'] = np.floor((lat_max - station['Lat']) / (-lat_res_negative)).astype(int)
        station['col'] = np.floor((station['Lon'] - lon_min) / lon_res).astype(int)
        station[var_name] = ds_band[station['row'], station['col']]
        station['date'] = os.path.basename(var_path).split('_')[2][:6]
        var.append(station.copy())
    var = pd.concat(var, ignore_index=True)
    var['date'] = var['date'].apply(lambda x: x[:4] + '/' + x[4:])
    var = var[['date', var_name]].groupby(['date']).mean()
    out_path = os.path.join(out_dir, '{}.csv'.format(var_name))
    var.to_csv(out_path)

# TWSC/EWTC/AWC均值计算
ewtc_by_station = ewtc[['Lon', 'Lat', 'EWTC']].groupby(['Lon', 'Lat']).mean().reset_index()
twsc_by_station = twsc[['Lon', 'Lat', 'TWSC_SH']].groupby(['Lon', 'Lat']).mean().reset_index()
awc_by_station = awc[['Lat_ewtc', 'Lon_ewtc', 'AWC']].groupby(['Lat_ewtc', 'Lon_ewtc']).mean().reset_index()
ndvi_by_station = ndvi[['Lon', 'Lat', 'NDVI']].groupby(['Lon', 'Lat']).mean().reset_index()
for var in [ewtc_by_station, twsc_by_station, awc_by_station, ndvi_by_station]:
    out_path = os.path.join(out_dir, 'distribution_{}.csv'.format(var.columns[-1]))
    var.to_csv(out_path, index=False)