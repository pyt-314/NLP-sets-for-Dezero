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
# Gelu
# =============================================================================


def gelu(xs):
    out = (xs + 0.047715 * xs**3)*math.sqrt(2/math.pi)
    out = 0.5 * xs * (1+F.tanh(out))
    return out

# =============================================================================
# GLU
# =============================================================================


class GLU(Layer):
    def __init__(self, out_size):
        self.af1 = L.Linear(out_size)
        self.af2 = L.Linear(out_size)

    def forward(self, x):
        out = self.af1(x)*F.sigmoid(self.af2(x))
        return out

# =============================================================================
# SimpleAttention/Dense/MatMul/Feed Forward Network(FFN)
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

class SimpleAttention(Layer):
    def __init__(self, depth, rate):
        super().__init__()
        self.depth = depth
        self.linearQ = Dense(depth)
        self.linearK = Dense(depth)
        self.linearV = Dense(depth)
        self.linearO = Dense(depth)
        self.dr = rate

    def forward(self, q, kv):
        Q = self.linearQ(q)
        K = self.linearK(kv)
        V = self.linearV(kv)
        attw = F.softmax(matmul2(Q, K.transpose(0, 2, 1)))
        attw = F.dropout(attw, self.dr)
        out = matmul2(attw, V)
        out = self.linearO(out)
        return V+out


class Multi_Head_Attention(Layer):
    def __init__(self, depth, head, rate):
        super().__init__()
        self.depth = depth
        self.head = head
        self.WQ = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WK = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WV = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.WO = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)))
        self.rate = rate

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
        logit = matmul2(q, k.transpose(0, 1, 3, 2))/np.sqrt(self.depth//self.head)
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
        self.l1 = F.relu(L.Linear(hidden_size))
        self.l2 = L.linear(in_size)

# =============================================================================
# Test
# =============================================================================

a = Variable(np.random.rand(10, 24, 16))
d = Variable(np.random.rand(10, 27, 16))
lay = Self_Attention(16,4, 0.01)#(Multi_Head_Attention)
b = lay(a)
b.backward()
c = a.grad.data
