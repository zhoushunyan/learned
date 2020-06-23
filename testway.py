import jieba
import pandas as pd
from keras.preprocessing import sequence
import numpy as np

pos = pd.read_excel('pos.xls',header=None,index=None)
neg = pd.read_excel('neg.xls', header=None, index=None)
pos['mark'] = 1
neg['mark'] = 0

neglen = len(neg)
poslen = len(pos)  # 计算语料数目

pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
#print(pn)
cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)


comment = pd.read_excel('sum.xls')
comment = comment[comment['rateContent'].notnull()]
comment['words'] = comment['rateContent'].apply(cw)

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)
w=[]
for i in d2v_train:
    w.extend(i)
dict = pd.DataFrame(pd.Series(w).value_counts())

dict['id'] = list(range(1, len(dict) + 1))
get_sent = lambda x: list(dict['id'][x]) # 将一句话中的每一个分词 对应 相应的数字。

pn['sent'] = pn['words'].apply(get_sent)
maxlen = 50
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2]
print(x)