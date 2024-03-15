# @Author   : ChaoQiezi
# @Time     : 2023/12/7  15:07
# @FileName : check_datasets.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 用于检查数据完整性, 包括MCD12Q1、MOD11A2、MOD13A2
-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-
拓展: MYD\MOD\MCD
MOD标识Terra卫星
MYD标识Aqua卫星
MCD标识Terra和Aqua卫星的结合
-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-·-
拓展: MCD12Q1\MOD11A2\MOD13A2
MCD12Q1为土地利用数据
MOD11A2为地表温度数据
MOD13A2为植被指数数据(包括NDVI\EVI)
"""

import os.path
import glob
from datetime import datetime, timedelta

# 准备
in_dir = r'F:\Cy_modis'
searching_ds_wildcard = ['MCD12Q1', 'MOD11A2', 'MOD13A2']

# 检查MCD12Q1数据集
error_txt = os.path.join(in_dir, 'MCD12Q1_check_error.txt')
ds_name_wildcard = 'MCD12Q1*'
region_wildcard = ['h26v05', 'h26v06', 'h27v05', 'h27v06']
with open(error_txt, 'w+') as f:
    for year in range(2001, 2021):
        for region in region_wildcard:
            cur_ds_name_wildcard = ds_name_wildcard + 'A{}*'.format(year) + region + '*.hdf'
            ds_path_wildcard = os.path.join(in_dir, '**', cur_ds_name_wildcard)
            hdf_paths = glob.glob(ds_path_wildcard, recursive=True)
            if len(hdf_paths) != 1:
                f.write('{}: 文件数目(为: {})不正常\n'.format(cur_ds_name_wildcard, len(hdf_paths)))
    if not f.read():
        f.write('MCD12Q1数据集文件数正常')

# 检查MOD11A2数据集
error_txt = os.path.join(in_dir, 'MOD11A2_check_error.txt')
ds_name_wildcard = 'MOD11A2*'
region_wildcard = ['h26v05', 'h26v06', 'h27v05', 'h27v06']
start_date = datetime(2000, 1, 1) + timedelta(days=48)
end_date = datetime(2022, 1, 1) + timedelta(days=296)
with open(error_txt, 'w+') as f:
    cur_date = start_date
    while cur_date <= end_date:
        cur_date_str = cur_date.strftime('%Y%j')
        for region in region_wildcard:
            cur_ds_name_wildcard = ds_name_wildcard + 'A{}*'.format(cur_date_str) + region + '*.hdf'
            ds_path_wildcard = os.path.join(in_dir, '**', cur_ds_name_wildcard)
            hdf_paths = glob.glob(ds_path_wildcard, recursive=True)
            if len(hdf_paths) != 1:
                f.write('{}: 文件数目(为: {})不正常\n'.format(cur_ds_name_wildcard, len(hdf_paths)))
        if (cur_date + timedelta(days=8)).year != cur_date.year:
            cur_date = datetime(cur_date.year + 1, 1, 1)
        else:
            cur_date += timedelta(days=8)
    if not f.read():
        f.write('MOD11A2数据集文件数正常')

# 检查MOD13A2数据集
error_txt = os.path.join(in_dir, 'MOD13A2_check_error.txt')
ds_name_wildcard = 'MOD13A2*'
region_wildcard = ['h26v05', 'h26v06', 'h27v05', 'h27v06']
start_date = datetime(2000, 1, 1) + timedelta(days=48)
end_date = datetime(2020, 1, 1) + timedelta(days=352)
with open(error_txt, 'w+') as f:
    cur_date = start_date
    while cur_date <= end_date:
        cur_date_str = cur_date.strftime('%Y%j')
        for region in region_wildcard:
            cur_ds_name_wildcard = ds_name_wildcard + 'A{}*'.format(cur_date_str) + region + '*.hdf'
            ds_path_wildcard = os.path.join(in_dir, '**', cur_ds_name_wildcard)
            hdf_paths = glob.glob(ds_path_wildcard, recursive=True)
            if len(hdf_paths) != 1:
                f.write('{}: 文件数目(为: {})不正常\n'.format(cur_ds_name_wildcard, len(hdf_paths)))
        if (cur_date + timedelta(days=16)).year != cur_date.year:
            cur_date = datetime(cur_date.year + 1, 1, 1)
        else:
            cur_date += timedelta(days=16)
    if not f.read():
        f.write('MOD13A2数据集文件数正常')



