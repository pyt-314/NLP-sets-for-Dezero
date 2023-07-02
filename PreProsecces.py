# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 11:00:31 2023

@author: luigi
"""

import dezero
import numpy as np
from collections import deque
import json

text1 = None
def gets(texts,t2i):
    out = []
    for i in texts:
        out.append(t2i[i])
    return out

def load_text_data(file):
    with open(file,encoding="utf-8") as f:
        text1 = f.read()
        text1 = "<pad> <BOS> "+text1
    text1 = text1.replace("."," .")
    text1 = text1.replace("\n"," <BOS> ")
    text1 = text1.replace(","," ,")
    text1 = text1.replace("?"," ?")
    text1 = text1.replace(";"," . ")
    return tuple(text1.split(" "))
def make_dict(text):
    t2i = {}
    i2t = {}
    c = 0
    for i in set(text):
        t2i[c] = i
        i2t[i] = c
        c += 1
    return i2t,t2i

def split_sentence(text):
    idx = 0
    out = deque()
    bi = 0
    while 1:
        try:
            idx = text.index("<BOS>",idx+1)
            out.append(text[bi:idx])
            bi = idx
        except:
            break
    out = tuple(out)
    maxlen = 0
    for i in out:
        maxlen = max(maxlen,len(i))
    return out[2:],maxlen

def add_pad(tp,maxlen,t2i):
    out = deque()
    for i in tp:
        out.append(tuple(gets(i,t2i) + [t2i["<pad>"]]*(maxlen-len(i))))
    return tuple(out)

def save_data(tensor,folder,batchs):
    ite = int(len(tensor)/batchs)
    for i in range(ite+1):
        out1 = tensor[:i:ite]
        progress = "#"*(int(i/ite*10)) +"_"*(10-int(i/ite*10))
        np.save("{}/data_{}".format(folder,i),out1)
        print("\r[Converted:[{}]{:.1f}%]".format(progress,int(i/ite*100)),end='')

def save_dict(dc,file):
    with open(file, 'w') as f:
        json.dump(dc, f, indent=2)

def load_dict(file):
    with open(file) as f:
        dic = json.load(f)
    return dic
text1 = load_text_data("train.en.txt")
