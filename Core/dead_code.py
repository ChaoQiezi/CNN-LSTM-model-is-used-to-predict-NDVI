# @Author   : ChaoQiezi
# @Time     : 2023/12/30  20:35
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""
import os

# 指定要更改的目录
path = r'H:\Datasets\Objects\Veg\LST_Max'

# 遍历目录下的所有文件
for filename in os.listdir(path):
    # 检查文件名是否包含"max"
    if "Max" in filename:
        # 创建新的文件名，将"max"替换为"MAX"
        new_filename = filename.replace("Max", "MAX")

        # 获取文件的原始路径和新路径
        old_path = os.path.join(path, filename)
        new_path = os.path.join(path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
