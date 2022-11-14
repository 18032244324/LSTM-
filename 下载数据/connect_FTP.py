#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 2:13 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : connect_FTP.py
# @Software: PyCharm
# @remarks : 无
import datetime
from ftplib import FTP
import os
import schedule
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from apscheduler.schedulers.blocking import BlockingScheduler
from save_气溶胶数据 import get_data,save_data
# 创建数据库对象
from 代码.Connect_DataBase import BaseSet
database = BaseSet()
class FTP_OP:
    def __init__(self, host, username, password, port, passive):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.pasv = passive

    def ftp_connect(self):
        ftp = FTP()
        ftp.encoding = 'gbk'
        ftp.set_debuglevel(0)# 不开启调试模式
        ftp.connect(self.host, self.port) # 连接ftp
        ftp.login(self.username, self.password)# 登录ftp
        ftp.set_pasv(self.pasv)#ftp有主动 被动模式 需要调整
        return ftp

    def download_file(self, ftp_file_path, dst_file_path,local_file_list,ftp):
        buffer_size = 8192
        # ftp = self.ftp_connect()
        print(ftp.getwelcome())
        try:
            file_list = ftp.nlst(ftp_file_path)
        except:
            print("没有该目录",ftp_file_path)
            return 0
        for file_name in file_list:
            ftp_file = os.path.join(ftp_file_path, file_name)
            a = os.path.basename(file_name)+'\n' #为ftp 需求目录下的所有文件
            if os.path.basename(file_name) not in local_file_list:
            # if os.path.basename(file_name)+'\n' not in local_file_list:
                write_file = dst_file_path + os.path.basename(file_name)
                # print(os.path.basename(file_name))
                # print(write_file)
                with open(write_file, "wb") as f:
                    ftp.retrbinary('RETR %s' % ftp_file, f.write, buffer_size)

                #下载完成之后对文件进行解析，数据保存到数据库
                List_data = get_data(write_file)
                if len(List_data)>0:
                    save_data(List_data)

                    # local_file_list.append()
                    print(os.path.basename(file_name),"ftp文件下载成功！")
                else:
                    print("文件解析失败！或者没有有效数据")
            else:
                print("此文件已经存在")
        # ftp.quit()

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
        _file_list_[i] = _file_list_[i][9:]
    return _file_list_


# 用不到了10.13
def download():
    host = "ftp.ptree.jaxa.jp"
    username = '983852772_qq.com'
    password = 'SP+wari8'
    port = 21
    ftp_filefolder = "/pub/himawari/L3/ARP/031/"  # FTP目录文件
    # ftp_filefolder += str(year) + str(month) + '/' + str(day)  # '/PUB/HIMAWARI/L3/ARP/031/202209/15'
    # ftp_filefolder = "/pub/himawari/L3/ARP/031/202210/01"
    dst_filefolder = "../数据/L3/"  # 本地存储文件目录
    ftp = FTP_OP(host, username, password, port, passive=True)
    ftp1 = ftp.ftp_connect()
    # 扫描获取本地已经下载的文件防止重复下载
    for i in range(1):
        i = i+5
        if i < 10:
            i = "0" + str(i)
        for j in range(32):
            j = j + 1
            if j < 10:
                j = "0" + str(j)
            ftp_filefolder = "/pub/himawari/L3/ARP/031/2022" + str(i) + "/" + str(j) # 小时平均
            local_file_list = ScanFile(r'../数据/L3/', True)

            ftp.download_file(ftp_filefolder, dst_filefolder, local_file_list,ftp1)
            print(ftp_filefolder)
            print("once over")


if __name__ == '__main__':
    # 下载日平均数据
    download()







