# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import jieba
from keras.layers import Bidirectional
from keras.layers.core import Flatten
from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras import metrics

neg = pd.read_excel('neg.xls', header=None, index=None)
pos = pd.read_excel('pos.xls', header=None, index=None)  # 读取训练语料完毕
pos['mark'] = 1
neg['mark'] = 0  # 给训练语料贴上标签
pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
neglen = len(neg)
poslen = len(pos)  # 计算语料数目

cw = lambda x: list(jieba.cut(x))  # 定义分词函数 ，lambda的作用是用来构造函数，lambda 参数：表达式。
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('sum.xls')  # 读入评论内容
# comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw)  # 将非空评论进行分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)  # 联合

w = []  # 将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计每个词和每个符号的出现次数
del w, d2v_train
dict['id'] = list(range(1, len(dict) + 1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)  # 速度太慢   # 将一句话中的每一个分词 对应 相应的数字。

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen)) #将sent列设置为固定长度maxlen，若小于maxlen，则添加0，若大于maxlen，则截取maxlen长度。

x = np.array(list(pn['sent']))[::2]  # 训练集 从0开始 间隔为2 的选取数值。
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]  # 测试集 从1开始 间隔为2 选取数值
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))  # 全集
ya = np.array(list(pn['mark']))

print('Build model...')
# LSTM模型
model = Sequential()
model.add(Embedding(len(dict) + 1, 256))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# Bi-LSTM模型
# model = Sequential()
# model.add(Embedding(len(dict)+1,256))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(128),merge_mode='concat'))
# model.add(Dropout(0.5))
# #model.add(Flatten())
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=16, nb_epoch=5, callbacks=[TensorBoard(log_dir='mytensorboard')])  # 训练时间为若干个小时

scores = model.evaluate(xt, yt)
print('LSTM:test_loss:%f,accuracy:%f' % (scores[0], scores[1]))
# classes = model.predict_classes(xt)
# print(classes)
# acc = metrics.binary_accuracy(yt,classes)
# print('Test accuracy:', acc)
