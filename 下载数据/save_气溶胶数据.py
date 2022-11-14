#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 4:54 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : save_气溶胶数据.py
# @Software: PyCharm
# @remarks : 无
import uuid

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from 代码.Connect_DataBase import BaseSet
#创建数据库对象
database = BaseSet()
# 数据保存到数据库
def save_data(List_data):
    # print(List_data)
    # if len(List_data)>0:
    #     print(List_data)
    #存数据
    database = BaseSet()
    # sql = "insert into Aerosol(time,grid,lng,lat,aerosol) values(%s,%s,%s,%s,%s)"
    sql = []
    sql.append( "insert into hb_bis_himawari_AOD(id,date,type,spacing) values (%s,%s,%s,%s)")

    # sql = "select * from Aerosol"
    # parameter = ('2002-02-02',2,3,4)
    #批量插入到数据库
    result = database.Save_Satellite_Aerosol(sql, List_data)
    # print(result)


def get_data(file_name):
    # 解析文件获取数据
    # nc_obj = Dataset('../NC格式数据/L3/H08_20220906_0400_1HARP031_FLDK.02401_02401.nc')
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

    # print(lon.shape)
    # print(lat.shape)
    # res = pd.DataFrame(AOT_arr, index=lat, columns=lon)
    # print(res)
    # 存储到文件
    # file_csv = "data_csv/20220906_0400_all.csv"


def Get_Aerosol_PM(List_data):
    result_data = []
    for i in range(len(List_data)):
        lon = List_data[i][1]
        lat = List_data[i][2]
        datatime = List_data[i][3]
        # print(datatime)
        # '20220907110000'->'2020-09-07 11:00:00'
        # datatime = datatime[0:4]+'-'+datatime[4:6]+'-'+datatime[6:8]+' '+datatime[8:10]+':'+datatime[10:12]+':'+datatime[12:14]
        sql = []
        sql.append( "SELECT round( avg( k.pm10 ), 2 ) AS pm10,round( avg( k.pm25 ), 2 ) AS pm25 FROM(SELECT avg( i.pm10 ) AS pm10,avg( i.pm25 ) AS pm25 FROM aircheck_2022 i " \
              "WHERE substring_index( i.lonlat, ',', 1 ) BETWEEN (%s) AND (%s) " \
              "AND substring_index( i.lonlat, ',', - 1 ) BETWEEN (%s) AND (%s) and i.timepoint = (%s) " \
              "UNION ALL SELECT avg( s.pm10 ) AS pm10,avg( s.pm25 ) AS pm25 FROM sites118_2022 s " \
              "WHERE s.lon BETWEEN (%s) AND (%s) AND s.lat BETWEEN (%s)  AND (%s) and s.qcDatetime = (%s) " \
              "UNION ALL SELECT avg( w.pm10 ) AS pm10,avg( w.pm25 ) AS pm25 FROM wsite27_2022 w left join wsite te on w.stationcode = te.bh " \
              "WHERE te.lon BETWEEN (%s) AND (%s) AND te.lat BETWEEN (%s) AND (%s) and receivetime = (%s)) k ")
        # parameter = [lon,float(lon)+float(0.03956),float(lat) - float(0.03956),lat,datatime,lon,lon,lat,lat,datatime,lon,lon,lat,lat,datatime]
        lon2 = str(lon + 0.03956)
        lat2 = str(lat - 0.03956)
        parameter = [str(lon), lon2, lat2, str(lat), datatime, str(lon), lon2, lat2, str(lat), datatime, str(lon), lon2, lat2, str(lat), datatime]
        result = database.Get_Aerosol_PM(sql,parameter)
        # if result != None:
        if result[0][0] == None:
            lon2 = str(float(lon2)+0.15)
            lat2 = str(float(lat2)-0.15)
            parameter = [lon, lon2, lat2, lat, datatime, lon, lon2, lat2, lat, datatime, lon, lon2, lat2, lat,
                         datatime, ]
            result = database.Get_Aerosol_PM(sql, parameter)
        # print(result)
        # if result[0][0]!=None:
        #     print(result)
        List_data[i] = List_data[i] + (result[0][0],result[0][1])
    return List_data


# 每次最新气溶胶数据入库的时候  获取这些数据 然后进行预测
# def get_data_need_pre():
#     sql = []
#     sql.append("SELECT * from hb_bis_himawari_aot_copy2 where pm25 is null")

#
# List_data = get_data("../NC格式数据/L3/H08_20220906_0400_1HARP031_FLDK.02401_02401.nc")
# List_data = get_data("../NC格式数据/L3/H08_20220323_0200_1HARP031_FLDK.02401_02401.nc")
# result = Get_Aerosol_PM(List_data)
# save_data(List_data)
# print(List_data)



