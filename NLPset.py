# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:42:33 2023

@author: python314
"""

import dezero
from dezero import Parameter, Variable
from dezero.models import Model
from dezero.functions import Function
from dezero.layers import Layer
import dezero.functions as F
import dezero.layers as L
import math
import numpy as np

# =============================================================================
# Gelu/mask
# =============================================================================


def gelu(xs):
    out = (xs + 0.047715 * xs**3)*math.sqrt(2/math.pi)
    out = 0.5 * xs * (1+F.tanh(out))
    return out

class Mask(Function):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        a= x.copy()
        shape = x.shape
        self.mask = ~np.tri(shape[1],shape[2],dtype="bool")
        a[:,self.mask] = -float("inf")
        return a
    def backward(self,grad):
        dx = grad.data.copy()
        dx[:,self.mask] = 0
        return dx

def mask(x):
    return Mask()(x)

def none(x):
    return x
# =============================================================================
# GLU(Now developing?)
# =============================================================================

class GLU(Layer):
    def __init__(self, out_size):
        self.af1 = L.Linear(out_size)
        self.af2 = L.Linear(out_size)

    def forward(self, x):
        out = self.af1(x)*F.sigmoid(self.af2(x))
        return out

# =============================================================================
# Dense/MatMul/Feed Forward Network(FFN)
# =============================================================================

class Dense(L.Linear):
    def forward(self, x):
        sh = x.shape
        x = x.reshape(-1, sh[-1])
        y = super().forward(x)
        return y.reshape(*sh[:-1], -1)

class Matmul2(Function):
    def forward(self, q, k):
        return q @ k
    def backward(self, gy):
        q, k = self.inputs
        if k.ndim == 4:
            gq = matmul2(gy, k.transpose(0,1,3,2))
        elif k.ndim == 2:
            gq = matmul2(gy, k.T)
        gk = matmul2(q.transpose(0,1,3,2), gy)
        return gq, gk

def matmul2(q, k):
    return Matmul2()(q, k)


class Multi_Head_Attention(Layer):
    def __init__(self, depth, head, rate, mask=none):
        super().__init__()
        self.depth = depth
        self.head = head
        self.WQ = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WK = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WV = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WO = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.rate = rate
        self.mask = mask
    def split(self, x):
        N, T, H = x.shape
        return x.reshape(N, T, H//self.head, self.head).transpose(0, 3, 1, 2)

    def concat(self, x):
        N, h, T, H = x.shape
        return x.transpose(0, 2, 3, 1).reshape(N, T, -1)

    def forward(self, i, m):
        i = self.split(i)
        m = self.split(m)
        q = matmul2(i, self.WQ)
        k = matmul2(m, self.WK)
        v = matmul2(m, self.WV)
        logit = self.mask(matmul2(q, k.transpose(0, 1, 3, 2))/np.sqrt(self.depth//self.head))
        a = F.softmax(logit,axis=2)
        a = F.dropout(a,self.rate)
        out = matmul2(a,v)
        out = q + matmul2(out,self.WO)
        out = self.concat(out)
        return out

class Self_Attention(Multi_Head_Attention):
    def __init__(self, depth, head, rate):
        super().__init__(depth,head,rate)
    def forward(self,i):
        return super().forward(i,i)

class FFN(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.l1 = Dense(hidden_size)
        self.l2 = Dense(in_size)
    def forward(self,x):
        x = self.l2(F.relu(self.l1(x)))
        return x

# =============================================================================
# LayerNorm/PositionalEncoding/wrapper(from https://qiita.com/halhorn/items/c91497522be27bde17ce)
# =============================================================================

def Norm(x):
    N,T,D = x.shape
    x1 = x[:]
    #(N,T,D) -> (T*D,N)
    x = x.transpose(1,2,0).reshape(-1,N)
    mean = x.sum(axis=0,keepdims=True)/T*D
    var = ((x-mean).sum(axis=0,keepdims=True)/T*D)**0.5
    out = (x1-mean)/(var+1e-5)
    return out

class LayerNorm(Layer):
    def __init__(self,hidden_size=None):
        super().__init__()
        self.var = Parameter(None)
        self.ma = Parameter(None)
        if not hidden_size == None:
            self.var = Parameter(np.ones((hidden_size))).reshape(-1,1)
            self.ma = Parameter(np.zeros((hidden_size))).reshape(-1,1)
    def _init_params(self,x):
        self.var.data = np.ones((x,1))
        self.ma.data = np.zeros((x,1))
    def forward(self,x):
        x1 = x
        self.shape = x.shape
        n = self.shape[1]*self.shape[2]
        if self.var.data == None:
            self._init_params(n)
        #(N,T,H)->(N*T,H)
        x = x.reshape(self.shape[0],-1).T
        mean = x.sum(axis=0,keepdims=True)/n
        var1 = (((x - mean)**2).sum(axis=0,keepdims=True)/n)**0.5
        out = (x-mean)/(var1+1e-6)
        out = out*self.var+self.ma
        return out.T.reshape(*x1.shape)

class Positional_Encoding1(Layer):
    def __init__(self,input_shape=None,max_len=10000):
        super().__init__()
        if input_shape != None:
            self.map = Parameter(input_shape)
        else:
            self.map = Parameter(None)
    def _init_map(self,x):
        N,T,H = x.shape
        self.map.data = np.random.randn(1,H)
    def forward(self,x):
        if self.map.data == None:
            self._init_map(x)
        return x + self.map

class Positional_Encoding2(Layer):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        N,T,H = x.shape
        pos = np.arange(0,H,1)//2*2
        pos1 = (np.arange(0,H,1)%2) * np.pi/2
        time = np.arange(0,T,1).reshape(-1,1)
        u = (10000**(pos/H)).reshape(1,H).repeat(T,axis=0)
        pe = np.sin(time/u+pos1)
        self.pe = pe
        return x+pe
        
class Wrapper(Layer):
    def __init__(self,layer,dropout_rate=0.1):
        super().__init__()
        self.rate = dropout_rate
        self.norm = LayerNorm()
        self.layer = layer
    def forward(self,x,m=None):
        ou = self.norm(x)
        if m==None:
            ou = self.layer(ou)
        else:
            ou = self.layers(ou,m)
        ou = F.dropout(ou,self.rate)
        return ou

# =============================================================================
# Transformer Encoder/Transformer Decoder/Embedding With Positional Encoding
# =============================================================================

class Embedding_with_pos(Layer):
    def __init__(self,
                 word_num,
                 pos_en=Positional_Encoding2(),
                 depth=1024):
        super().__init__()
        self.layers = []
        self.layers += [L.EmbedID(word_num,depth)]
        self.layers += [pos_en]
    def forward(self,x):
        for l in self.layers:
            x = l(x)
        return x

class Transformer_Encoder(Layer):
    def __init__(self,
                 depth,
                 hidden=512,
                 hopping=6,
                 head=8,
                 rate=0.1):
        super().__init__()
        self.layers = []
        #self.layers = [lambda x:x.transpose(1,2,0)]
        for i in range(hopping):
            self.layers += [Wrapper(Self_Attention(depth, head, rate),rate)]
            self.layers += [Wrapper(FFN(depth,hidden),rate)]
    def forward(self,input_data):
        x = input_data
        for i in self.layers:
            x = i(x)
        return x

class Transformer_Decoder(Layer):
    def __init__(self,
                 depth,
                 hidden=512,
                 hopping=6,
                 head=8,
                 rate=0.1):
        super().__init__()
        self.layers = []
        for i in range(hopping):
            self.layers += [(Wrapper(Self_Attention(depth, head, rate),rate),
                             Wrapper(Multi_Head_Attention(depth, head, rate),rate),
                             Wrapper(FFN(depth,hidden),rate))]
    def forward(self,x,s):
        for i in self.layers:
            l1,l2,l3 = i
            x = l1(x)
            x = l2(x,s)
            x = l3(x)
        return x

class Transformer(Layer):
    pass