# @Author   : ChaoQiezi
# @Time     : 2024/1/3  16:51
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to 对各个数据集进行统一,例如空间范围()

主要包括: 对modis(土地利用、ndvi、地表温度)、geo(DEM等)、gldas数据集进行重采样, 范围限定(裁剪至掩膜形状)
"""

import os.path
from glob import glob
from concurrent.futures import ThreadPoolExecutor  # 线程池

from osgeo import gdal

# 准备
in_dir = r'E:\FeaturesTargets\non_uniform'
out_dir = r'E:\FeaturesTargets\uniform'
shp_path = r'E:\Basic\Region\sw5f\sw5_mask.shp'
dem_path = r'E:\GEO\cndem01.tif'
out_res = 0.1


def resample_clip_mask(in_dir: str, out_dir: str, shp_path: str, wildcard: str, out_res: float = 0.1,
                       resampleAlg=gdal.GRA_Cubic):
    """
    该函数用于对指定文件夹内的影像进行批量重采样和裁剪、掩膜
    :param in_dir: 待处理文件所在文件夹目录
    :param out_dir: 输出文件的文件夹目录
    :param shp_path: 掩膜裁剪的shp文件
    :param wildcard: 检索输入文件夹内指定文件的通配符
    :param out_res: 输出分辨率
    :param resampleAlg: 重采样方法
    :return: None
    """

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    target_paths = glob(os.path.join(in_dir, wildcard))
    for target_path in target_paths:
        out_path = os.path.join(out_dir, os.path.basename(target_path))

        img = gdal.Warp(
            out_path,  # 输出位置
            target_path,  # 源文件位置
            cutlineDSName=shp_path,  # 掩膜裁剪所需文件
            cropToCutline=True,  # 裁剪至掩膜形状
            xRes=out_res,  # X方向分辨率
            yRes=out_res,  # Y方向分辨率
            resampleAlg=resampleAlg  # 重采样方法
        )
        img = None

        print('目前已处理: {}'.format(os.path.splitext(os.path.basename(target_path))[0]))


# # 处理土地利用数据集
# in_landuse_dir = os.path.join(in_dir, 'Landuse')
# out_landuse_dir = os.path.join(out_dir, 'Landuse')
# resample_clip_mask(in_landuse_dir, out_landuse_dir, shp_path, 'Landuse*.tiff', resampleAlg=gdal.GRA_NearestNeighbour)
# # 处理地表温度数据集
# in_lst_dir = os.path.join(in_dir, 'LST')
# out_lst_dir = os.path.join(out_dir, 'LST')
# resample_clip_mask(in_lst_dir, out_lst_dir, shp_path, 'LST*.tiff')
# # 处理NDVI数据集
# in_ndvi_dir = os.path.join(in_dir, 'NDVI')
# out_ndvi_dir = os.path.join(out_dir, 'NDVI')
# resample_clip_mask(in_ndvi_dir, out_ndvi_dir, shp_path, 'NDVI*.tiff')
# # 处理ET(蒸散发量)数据集
# in_et_dir = os.path.join(in_dir, 'ET')
# out_et_dir = os.path.join(out_dir, 'ET')
# resample_clip_mask(in_et_dir, out_et_dir, shp_path, 'GLDAS_ET*.tiff')
# # 处理降水数据集
# in_prcp_dir = os.path.join(in_dir, 'PRCP')
# out_prcp_dir = os.path.join(out_dir, 'PRCP')
# resample_clip_mask(in_prcp_dir, out_prcp_dir, shp_path, 'GLDAS_PRCP*.tiff')
# # 处理Qs(表面径流量)数据集
# in_qs_dir = os.path.join(in_dir, 'Qs')
# out_qs_dir = os.path.join(out_dir, 'Qs')
# resample_clip_mask(in_qs_dir, out_qs_dir, shp_path, 'GLDAS_Qs*.tiff')
# # 处理Qsb(地下径流量)数据集
# in_qsb_dir = os.path.join(in_dir, 'Qsb')
# out_qsb_dir = os.path.join(out_dir, 'Qsb')
# resample_clip_mask(in_qsb_dir, out_qsb_dir, shp_path, 'GLDAS_Qsb*.tiff')
# # 处理TWSC数据集
# in_twsc_dir = os.path.join(in_dir, 'TWSC')
# out_twsc_dir = os.path.join(out_dir, 'TWSC')
# resample_clip_mask(in_twsc_dir, out_twsc_dir, shp_path, 'GLDAS_TWSC*.tiff')
# 处理DEM数据集
# out_dem_path = os.path.join(out_dir, 'dem.tiff')
# img = gdal.Warp(
#     out_dem_path,
#     dem_path,
#     cutlineDSName=shp_path,
#     cropToCutline=True,
#     xRes=out_res,
#     yRes=out_res,
#     resampleAlg=gdal.GRA_Cubic
# )
# img = None

# 并行处理(加快处理速度)
datasets_param = {
    'Landuse': 'Landuse*.tiff',
    'LST_MEAN': 'LST_MEAN*.tiff',
    'LST_MAX': 'LST_MAX*.tiff',
    'LST_MIN': 'LST_MIN*.tiff',
    'NDVI_MEAN': 'NDVI_MEAN*.tiff',
    'NDVI_MAX': 'NDVI_MAX*.tiff',
    'NDVI_MIN': 'NDVI_MIN*.tiff',
    'ET': 'GLDAS_ET*.tiff',
    'PRCP': 'GLDAS_PRCP*.tiff',
    'Qs': 'GLDAS_Qs*.tiff',
    'Qsb': 'GLDAS_Qsb*.tiff',
    'TWSC': 'GLDAS_TWSC*.tiff',

}

if __name__ == '__main__':
    with ThreadPoolExecutor() as executor:
        futures = []
        for dataset_name, wildcard in datasets_param.items():
            in_dataset_dir = os.path.join(in_dir, dataset_name)
            out_dataset_dir = os.path.join(out_dir, dataset_name)
            resampleAlg = gdal.GRA_NearestNeighbour if dataset_name == 'Landuse' else gdal.GRA_Cubic
            futures.append(executor.submit(resample_clip_mask, in_dataset_dir, out_dataset_dir, shp_path,
                                           wildcard, resampleAlg=resampleAlg))
        # 处理DEM
        out_dem_path = os.path.join(out_dir, 'dem.tiff')
        futures.append(executor.submit(gdal.Warp, out_dem_path, dem_path, cutlineDSName=shp_path,
                                       cropToCutline=True, xRes=out_res, yRes=out_res, resampleAlg=gdal.GRA_Cubic))
        # 等待所有数据集处理完成
        for future in futures:
            future.result()

# 处理DEM数据集
"""
下述代码比较冗余, 简化为resample_clip_mask函数
----------------------------------------------------------------------
# 处理地表温度数据
lst_paths = glob(os.path.join(lst_dir, 'LST*.tiff'))
out_lst_dir = os.path.join(out_dir, lst_dir.split('\\')[-1])
if not os.path.exists(out_lst_dir): os.makedirs(out_lst_dir)
for lst_path in lst_paths:
    out_path = os.path.join(out_lst_dir, os.path.basename(lst_path))

    # 重采样、掩膜和裁剪
    gdal.Warp(
        out_path,
        lst_path,
        xRes=out_res,
        yRes=out_res,
        cutlineDSName=shp_path,  # 设置掩膜 shp文件
        cropToCutline=True,  # 裁剪至掩膜形状
        resampleAlg=gdal.GRA_Cubic  # 重采样方法: 三次卷积
    )
    print('目前已处理: {}'.format(os.path.splitext(os.path.basename(lst_path))[0]))

# 处理ndvi数据集
ndvi_paths = glob(os.path.join(ndvi_dir, 'NDVI*.tiff'))
out_ndvi_dir = os.path.join(out_dir, ndvi_dir.split('\\')[-1])
if not os.path.exists(out_ndvi_dir): os.makedirs(out_ndvi_dir)
for ndvi_path in ndvi_paths:
    out_path = os.path.join(out_ndvi_dir, os.path.basename(ndvi_path))
    out_path = os.path.join(out_ndvi_dir, 'NDVI_temp.tiff')
    gdal.Warp(
        out_path,
        ndvi_path,
        cutlineDSName=shp_path,  # 设置掩膜 shp文件
        cropToCutline=True,  # 是否裁剪至掩膜形状
        xRes=out_res,
        yRes=out_res,
        resampleAlg=gdal.GRA_Cubic  # 重采样方法: 三次卷积
    )
"""
