# @Author   : ChaoQiezi
# @Time     : 2024/1/17  12:41
# @FileName : process_gldas.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 预处理global gldas数据集

说明:
    为确保简洁性和便捷性, 今后读取HDF5文件和NC文件均使用xarray模块而非h5py和NetCDF4模块
数据集介绍:
    TWSC = 降水量(PRCP) - 蒸散发量(ET) - 径流量(即表面径流量Qs + 地下径流量Qsb)    ==> 给定时间间隔内, 例如月
    在gldas数据集中:
        Rainf_f_tavg表示降水通量，即单位时间单位面积上的降水量(本数据集单位为kg/m2/s)
        Evap_tavg表示蒸散发通量，即单位时间单位面积上的水蒸发量(本数据集单位为kg/m2/s)
        Qs_acc表示表面径流量，即一定时间内通过地表流动进入河流、湖泊和水库的水量(本数据集单位为kg/m2)
        Qsb_acc表示地下径流量，即一定时间内通过土壤层流动的水量，最终进入河流的水量，最终进入河流的水量(本数据集单位为kg/m2)
        TWSC计算了由降水和蒸发引起的净水量变化，再减去地表和地下径流，其评估给定时间段内区域水资源变化的重要指标

存疑:
    01 对于Qs和Qsb的计算, 由于数据集单位未包含/s， 是否已经是月累加值？ --2024/01/18(已解决)
    ==> 由gldas_tws_eg.py知是: numbers of 3 hours in a month,
        另外nc文件全局属性也提及:
            :tavg_definision: = "past 3-hour average";
            :acc_definision: = "past 3-hour accumulation";

"""

import os.path
from glob import glob
from calendar import monthrange
from datetime import datetime

import numpy as np
import xarray as xr
from osgeo import gdal, osr

# 准备
in_dir = r'E:\Global GLDAS'  # 检索该文件夹及迭代其所有子文件夹满足要求的文件
out_dir = r'E:\FeaturesTargets\non_uniform'
target_names = ['Rainf_f_tavg', 'Evap_tavg', 'Qs_acc', 'Qsb_acc']
out_names = ['PRCP', 'ET', 'Qs', 'Qsb', 'TWSC']
out_res = 0.1  # default: 0.25°, base on default res of gldas
no_data_value = -65535.0  # 缺失值或者无效值的设置
# 预准备
[os.makedirs(os.path.join(out_dir, _name)) for _name in out_names if not os.path.exists(os.path.join(out_dir, _name))]

# 检索和循环
nc_paths = glob(os.path.join(in_dir, '**', 'GLDAS_NOAH025_M*.nc4'), recursive=True)
for nc_path in nc_paths:
    # 获取当前月天数
    cur_time = datetime.strptime(nc_path.split('.')[1], 'A%Y%m')  # eg. 200204
    _, cur_month_days = monthrange(cur_time.year, cur_time.month)

    ds = xr.open_dataset(nc_path)
    # 读取经纬度数据集和地理参数
    lon = ds['lon'].values  # (1440, )
    lat = ds['lat'].values  # (600, )
    lon_res = ds.attrs['DX']
    lat_res = ds.attrs['DY']
    lon_min = min(lon) - lon_res / 2.0
    lon_max = max(lon) + lon_res / 2.0
    lat_min = min(lat) - lat_res / 2.0
    lat_max = max(lat) + lat_res / 2.0
    """
    注意: 经纬度数据集中的所有值均指代对应地理位置的像元的中心处的经纬度, 因此经纬度范围需要往外扩充0.5个分辨率
    """
    geo_transform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]  # gdal要求样式
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84

    fluxs = {}
    # 获取Rain_f_tavg, Evap_tavg, Qs_acc, Qsb_acc四个数据集
    for target_name, out_name in zip(target_names, out_names):  # 仅循环前四次
        # 计算月累加值
        flux = ds[target_name].values
        vmin = ds[target_name].attrs['vmin']
        vmax = ds[target_name].attrs['vmax']
        flux[(flux < vmin) | (flux > vmax)] = np.nan  # 将不在规定范围内的值设置为nan
        flux = np.squeeze(flux)  # 去掉多余维度
        flux = np.flipud(flux) # 南北极颠倒(使之正常: 北极在上)
        if target_name.endswith('acc'):  # :acc_definision: = "past 3-hour accumulation";
            flux *= cur_month_days * 8
        elif target_name.endswith('tavg'):  # :tavg_definision: = "past 3-hour average";
            flux *= cur_month_days * 24 * 3600
        fluxs[out_name] = flux

    fluxs['TWSC'] = fluxs['PRCP'] - fluxs['ET'] - (fluxs['Qs'] + fluxs['Qsb'])  # 计算TWSC
    for out_name, flux in fluxs.items():
        # 输出路径
        cur_out_name = 'GLDAS_{}_{:04}{:02}.tiff'.format(out_name, cur_time.year, cur_time.month)
        cur_out_path = os.path.join(out_dir, out_name, cur_out_name)

        driver = gdal.GetDriverByName('MEM')  # 在内存/TIFF中创建
        temp_img = driver.Create('', flux.shape[1], flux.shape[0], 1, gdal.GDT_Float32)
        temp_img.SetProjection(srs.ExportToWkt())  # 设置坐标系
        temp_img.SetGeoTransform(geo_transform)  # 设置仿射参数
        flux = np.nan_to_num(flux, nan=no_data_value)
        temp_img.GetRasterBand(1).WriteArray(flux)  # 写入数据集
        temp_img.GetRasterBand(1).SetNoDataValue(no_data_value)  # 设置无效值
        resample_img = gdal.Warp(cur_out_path, temp_img, xRes=out_res, yRes=out_res, resampleAlg=gdal.GRA_Cubic)  # 重采样
        # 去除由于重采样造成的数据集不符合实际意义例如降水为负值等情况
        vmin = np.nanmin(flux)
        vmax = np.nanmax(flux)
        flux = resample_img.GetRasterBand(1).ReadAsArray()
        resample_img_srs = resample_img.GetProjection()
        resample_img_transform = resample_img.GetGeoTransform()
        temp_img, resample_img = None, None  # 释放资源
        flux[flux < vmin] = vmin
        flux[flux > vmax] = vmax
        driver = gdal.GetDriverByName('GTiff')
        final_img = driver.Create(cur_out_path, flux.shape[1], flux.shape[0], 1, gdal.GDT_Float32)
        final_img.SetProjection(resample_img_srs)
        final_img.SetGeoTransform(resample_img_transform)
        final_img.GetRasterBand(1).WriteArray(flux)
        final_img.GetRasterBand(1).SetNoDataValue(no_data_value)
        final_img.FlushCache()
        temp_img, final_img = None, None

        print('当前处理: {}-{}'.format(out_name, cur_time.strftime('%Y%m')))

    ds.close()  # 关闭当前nc文件，释放资源
print('处理完成')