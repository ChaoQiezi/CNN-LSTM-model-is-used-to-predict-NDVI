# @Author   : ChaoQiezi
# @Time     : 2024/5/9  19:25
# @FileName : process_Rs.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 处理RS地表太阳辐射并提取经纬度数据集，通过裁剪、掩膜、重采样等处理输出为tiff文件
"""

import os.path
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from osgeo import gdal, osr
import pandas as pd

# 准备
Rs_path = r'H:\Datasets\Objects\Veg\GWRHXG_Rs1.nc'
out_Rs_dir = r'E:\FeaturesTargets\uniform\Rs'
out_dir = r'E:\FeaturesTargets\uniform'
mask_path = r'E:\Basic\Region\sw5f\sw5_mask.shp'
out_res = 0.1  # 度(°)
if not os.path.exists(out_Rs_dir): os.makedirs(out_Rs_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)

# 读取
with nc.Dataset(Rs_path) as f:
    lon, lat = f['longitude'][:].filled(-9999), f['latitude'][:].filled(-9999)
    Rs = f['Rs'][:].filled(-9999)
    years, months = f['year'][:].filled(np.nan), f['month'][:].filled(np.nan)
    for ix, (year, month) in enumerate(zip(years, months)):
        cur_Rs = Rs[ix, :, :]  # 当前时间点的Rs地表太阳辐射
        lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
        lon_res = (lon_max - lon_min) / len(lon)
        lat_res = (lat_max - lat_min) / len(lat)
        geo_transform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]

        # 输出
        out_file_name = 'Rs_{:4.0f}{:02.0f}.tiff'.format(year, month)
        out_path = os.path.join(out_Rs_dir, out_file_name)
        mem_driver = gdal.GetDriverByName('MEM')
        mem_ds = mem_driver.Create('', len(lon), len(lat), 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        mem_ds.SetProjection(srs.ExportToWkt())
        mem_ds.SetGeoTransform(geo_transform)
        mem_ds.GetRasterBand(1).WriteArray(cur_Rs)
        mem_ds.GetRasterBand(1).SetNoDataValue(-9999)  # 设置无效值
        out_ds = gdal.Warp(out_path, mem_ds, cropToCutline=True, cutlineDSName=mask_path, xRes=out_res, yRes=out_res,
                  resampleAlg=gdal.GRA_Cubic, srcNodata=-9999, dstNodata=-9999)
        mem_ds.FlushCache()
        print('processing: {}'.format(out_file_name))

    masked_geo_transform = out_ds.GetGeoTransform()
    rows, cols = out_ds.RasterYSize, out_ds.RasterXSize
    lat = np.array([masked_geo_transform[3] + _ix * masked_geo_transform[-1] + masked_geo_transform[-1] / 2 for _ix in range(rows)])
    lon = np.array([masked_geo_transform[0] + _ix * masked_geo_transform[1] + masked_geo_transform[1] / 2 for _ix in range(cols)])
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    driver = gdal.GetDriverByName('GTiff')
    lon_ds = driver.Create(os.path.join(out_dir, 'Lon.tiff'), len(lon), len(lat), 1, gdal.GDT_Float32)
    lat_ds = driver.Create(os.path.join(out_dir, 'Lat.tiff'), len(lon), len(lat), 1, gdal.GDT_Float32)
    srs.ImportFromEPSG(4326)
    lon_ds.SetProjection(srs.ExportToWkt())
    lon_ds.SetGeoTransform(masked_geo_transform)
    lon_ds.GetRasterBand(1).WriteArray(lon_2d)
    lon_ds.GetRasterBand(1).SetNoDataValue(-9999)  # 设置无效值
    lat_ds.SetProjection(srs.ExportToWkt())
    lat_ds.SetGeoTransform(masked_geo_transform)
    lat_ds.GetRasterBand(1).WriteArray(lat_2d)
    lat_ds.GetRasterBand(1).SetNoDataValue(-9999)  # 设置无效值
    lon_ds.FlushCache()
    lat_ds.FlushCache()

print('Done!')