#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 9:00 上午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : 历史天气4.py
# @Software: PyCharm
# @remarks : 无
import requests
import re
import time
from bs4 import BeautifulSoup
import pandas as pd

#url = 'http://lishi.tianqi.com/mianyang/201905.html'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
            'Cookie':'lianjia_uuid=9d3277d3-58e4-440e-bade-5069cb5203a4; UM_distinctid=16ba37f7160390-05f17711c11c3e-454c0b2b-100200-16ba37f716618b; _smt_uid=5d176c66.5119839a; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2216ba37f7a942a6-0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22%24device_id%22%3A%2216ba37f7a942a6-0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; _ga=GA1.2.1772719071.1561816174; Hm_lvt_9152f8221cb6243a53c83b956842be8a=1561822858; _jzqa=1.2532744094467475000.1561816167.1561822858.1561870561.3; CNZZDATA1253477573=987273979-1561811144-%7C1561865554; CNZZDATA1254525948=879163647-1561815364-%7C1561869382; CNZZDATA1255633284=1986996647-1561812900-%7C1561866923; CNZZDATA1255604082=891570058-1561813905-%7C1561866148; _qzja=1.1577983579.1561816168942.1561822857520.1561870561449.1561870561449.1561870847908.0.0.0.7.3; select_city=110000; lianjia_ssid=4e1fa281-1ebf-e1c1-ac56-32b3ec83f7ca; srcid=eyJ0Ijoie1wiZGF0YVwiOlwiMzQ2MDU5ZTQ0OWY4N2RiOTE4NjQ5YmQ0ZGRlMDAyZmFhODZmNjI1ZDQyNWU0OGQ3MjE3Yzk5NzFiYTY4ODM4ZThiZDNhZjliNGU4ODM4M2M3ODZhNDNiNjM1NzMzNjQ4ODY3MWVhMWFmNzFjMDVmMDY4NWMyMTM3MjIxYjBmYzhkYWE1MzIyNzFlOGMyOWFiYmQwZjBjYjcyNmIwOWEwYTNlMTY2MDI1NjkyOTBkNjQ1ZDkwNGM5ZDhkYTIyODU0ZmQzZjhjODhlNGQ1NGRkZTA0ZTBlZDFiNmIxOTE2YmU1NTIxNzhhMGQ3Yzk0ZjQ4NDBlZWI0YjlhYzFiYmJlZjJlNDQ5MDdlNzcxMzAwMmM1ODBlZDJkNmIwZmY0NDAwYmQxNjNjZDlhNmJkNDk3NGMzOTQxNTdkYjZlMjJkYjAxYjIzNjdmYzhiNzMxZDA1MGJlNjBmNzQxMTZjNDIzNFwiLFwia2V5X2lkXCI6XCIxXCIsXCJzaWduXCI6XCIzMGJlNDJiN1wifSIsInIiOiJodHRwczovL2JqLmxpYW5qaWEuY29tL3p1ZmFuZy9yY28zMS8iLCJvcyI6IndlYiIsInYiOiIwLjEifQ=='
            }

def set_link(year):
    #year参数为需要爬取数据的年份
    link = []
    for i in range(1,13):
        #一年有12个月份
        if i < 10:
            url='http://lishi.tianqi.com/guanxian/{}0{}.html'.format(year,i)
        else:
            url='http://lishi.tianqi.com/guanxian/{}{}.html'.format(year,i)
        link.append(url)
    return link

def get_page(url,headers):
    html = requests.get(url,headers=headers)
    if html.status_code == 200:
        html.encoding = html.apparent_encoding
        return html.text
    else:
        return None

date_box = []
max_temp = []
min_temp = []
weh = []
wind = []
week_box = []

def get_data():
    link = set_link(2022)
    for url in link:
        html = get_page(url,headers)
        bs = BeautifulSoup(html,'html.parser')

        data = bs.find_all(class_='thrui')
        date = re.compile('class="th200">(.*?)</')
        tem = re.compile('class="th140">(.*?)</')
        time = re.findall(date,str(data))
        print(time)
        print(len(time))
        for item in time:
            week = item[10:]
            week_box.append(week)
            date_box.append(item[:10])
        temp = re.findall(tem, str(data))
        for i in range(len(time)):
            #之前因为自身需要的只是19年6月的天气信息，没有考虑到每个月的天数不一样，现在修改后就没有问题了
            max_temp.append(temp[i*4+0])
            min_temp.append(temp[i*4+1])
            weh.append(temp[i*4+2])
            wind.append(temp[i*4+3])
get_data()
datas = pd.DataFrame({'日期':date_box,'星期':week_box,'最高温度':max_temp,'最低温度':min_temp,'天气':weh,'风向':wind})
# datas.to_csv('天气数据.csv',encoding='utf_8_sig')
print(datas['星期'][0])
from 代码.Connect_DataBase import BaseSet
# from connect_database import BaseSet
baseset = BaseSet()
sql =[]
parameter = []
# sql.append("insert into weather2(date,weather,wind,temp_max,temp_min) values(%s,%s,%s,%s,%s)")
sql.append("update hb_bis_himawari_AOD set speed = (%s) where substring_index(date,' ',1)  =(%s)")
# sql.append("update hb_bis_himawari_AOD set speed = (%s) where id =(%s)")

# sql = "insert into weather2(date,weather,wind,temp_max,temp_min) values(%s,%s,%s,%s,%s)"
# sql = "select * from weather2"

# parameter[0]=datas['日期']
# parameter[1]=datas['天气']
# parameter[2]=datas['风向']
# parameter[3]=datas['最高温度']
# parameter[4]=datas['最低温度']
parameter.append(datas['日期'])
parameter.append(datas['天气'])
parameter.append(datas['风向'])
parameter.append(datas['最高温度'])
parameter.append(datas['最低温度'])


data = []
for i in range(len(date_box)):
    speed = "".join(list(filter(str.isdigit, wind[i])))
    # data.append([date_box[i],weh[i],wind[i],max_temp[i],min_temp[i],speed])
    data.append((date_box[i],speed))
    date = date_box[i]
    baseset.save_weather(sql, [speed,date])
    # 返回12345

print(data[0])
print(data[1])
print(data[2])
print(data[3])
# baseset.save_weather(sql[0],data)




