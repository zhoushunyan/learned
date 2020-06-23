# -*- coding: utf-8 -*-
import jieba
import numpy as np
import pandas as pd

# pn = pd.read_csv('task1/train.csv')
# cw = lambda x: list(jieba.cut(x))  # 定义分词函数 ，lambda的作用是用来构造函数，lambda 参数：表达式。
# pn['words'] = pn['text'].apply(cw)
# print(pn)
# pn['lant']=3
# #x= pn[2]
# print(pn)
# x = np.random.uniform(0.0, 1.0, (200))
# x = [1, 2, 3,4]
# y = np.expand_dims(x[:3], 1)
# print(x)
# print(y)

# pn = pd.read_csv('task1/train.csv')  # 合并语料
#
# q=[[1,2,2],[1,3,4]]
# print(q)
# t=[2,1]
# q = np.c_[q,t]
# 将二维数组转化为一维数组。
# c= []
# for i in range(10):
#     c.append(i)
# print(c)
# a =[[1],[2],[3]]
# print(a[0][0])
# b=[]
# for i in a:
#     b=b+i
# print(b)
# print()

#
# a = pd.Series([1,2,3])
#
# print(a)
pn = pd.read_csv('submit.csv')

p=pn.iloc[:,[1,3]]  #截取第二列和第四列的数据
print(pn)
p.to_csv('submit2.csv')