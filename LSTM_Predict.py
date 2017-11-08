

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Gilbert Sun
# email : gilbertsun@itri.org.tw
# data  : 2017/11/01

"""
Tensorflow 中采用 LSTM to分类 Battery Status
"""
# 1) 载入Data
# From目录：http://60.251.xxx.xxx/home/iris/battery/ ... \
# download dataset到当前目录下
# one_hot=1, 例如将label data 转化 4 status[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
