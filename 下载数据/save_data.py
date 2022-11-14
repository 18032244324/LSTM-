#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 10:26 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : save_data.py
# @Software: PyCharm
# @remarks : 无
import glob
import os
import uuid

import numpy as np
from netCDF4 import Dataset


def get_data(file_name):
    # 解析文件获取数据
    # nc_obj = Dataset('../数据/202203/01/H08_20220301_1400_1HARP031_FLDK.02401_02401.nc')

    # nc_obj = Dataset(file_name)
    try:
        nc_obj = Dataset(file_name)
    except:
        print(file_name,"解析出错")
        return 0
    # nc_obj = Dataset('../NC格式数据/L2/NC_H08_20220905_0900_L2ARP030_FLDK.02401_02401.nc')
    # print(nc_obj)
    # 查看nc文件中的变量
    # print(nc_obj.variables.keys())
    # print(nc_obj.variables["AOT_Merged"])
    # 获取气溶胶数据信息  小时均值
    AOT = nc_obj.variables["AOT_L2_Mean"]
    # AOT_arr = AOT[:]
    #转换成二维数组
    AOT_arr = np.asarray(AOT)
    # 获取冠县对应的网格数据
    lng = nc_obj.variables['longitude'][706:717]
    lat = nc_obj.variables['latitude'][466:474]
    AOT_arr = AOT_arr[466:474, 706:717]
    '''均值'''

    # 对缺省值归0处理
    AOT_arr[np.where(AOT_arr == -32768)] = 0.0
    mean = np.mean(AOT_arr)
    #存储网格数据
    List_data = []
    # 对日期字符串切分处理
    time = nc_obj.id[4:17]
    # '20220906_0400'
    date = time[0:4]+'-'+time[4:6]+'-'+time[6:8]
    # date = time[0:8]
    time = int(time[9:11])
    #转换为北京时间
    time = (time + 8)%24
    if time<10:
        time = '0'+str(time)
    time = str(time)+':'+'00:00'

    # if mean>0:
    data = (str(uuid.uuid1()),date + " " + str(time), 'hour',mean)
    return data
    # else:
    #     return 0
    # data = (
    # str(uuid.uuid1()), lng[j], lat[i], date + " " + str(time), 'AOT', AOT_arr[i][j], 'hour', str(i) + '-' + str(j))
# 扫描本地已经下载的文件
def ScanFile(path, scanSubdirectory=True, _usrCall=True):
    global _file_list_
    if _usrCall: _file_list_ = []
    files = os.scandir(path)
    for v in files:
        if v.is_dir():
            if scanSubdirectory: ScanFile(v.path, scanSubdirectory, False)
        else:
            _file_list_.append(v.path)
    for i in range(len(_file_list_)):
        # _file_list_[i] = _file_list_[i][21:]
        _file_list_[i] = _file_list_[i][:]
    return _file_list_
# from 下载数据.connect_FTP import ScanFile
from 下载数据.save_气溶胶数据 import save_data

for i in range(32):
    i=1+i
    if i <10:
        i = "0"+str(i)

    local_file_list = ScanFile(r'../数据/202206/'+str(i), True)

    for file in local_file_list:

        # f = open(file, 'r')
        try:
            List_data = get_data(file)
            if len(List_data) > 0:
                save_data(List_data)
                print(file, "保存成功")

        except:
            print(file,"解析失败")