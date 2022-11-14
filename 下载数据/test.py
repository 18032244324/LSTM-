#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 3:56 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : test.py
# @Software: PyCharm
# @remarks : 无
from matplotlib import pyplot as plt
import numpy as np
x= np.linspace(0,2*np.pi,500)

#分开绘制，生成一个figure对象供操作
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x,label="aaaa")
ax1.set_title('source image')
plt.legend()
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x,label="bbbb")
ax2.set_title('source image')
plt.legend()
plt.show()

