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
from dezero import cuda
import numpy as np
import matplotlib.pyplot as plt
from dezero.optimizers import Optimizer


# =============================================================================
# Gelu/mask/KLdiv/JSdiv/concat
# =============================================================================
 
pad = False

def gelu(xs):
    out = (xs + 0.047715 * xs**3)*math.sqrt(2/math.pi)
    out = 0.5 * xs * (1+F.tanh(out))
    return out

class Mask(Function):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        xp = cuda.get_array_module(x)
        global pad
        a= x.copy()
        N,H,T,D = x.shape
        self.mask = xp.tri(N,T,dtype="bool").reshape(N,1,-1).repeat(H,axis=1)
        a[self.mask,:] = 0
        return a
    def backward(self,grad):
        dx = grad.data.copy()
        dx[self.mask,:] = 0
        return dx

def mask(x):
    return Mask()(x)*(pad)

def none(x):
    return x*(pad)

def KLdiv(p,q):
    return F.sum(p*F.log(p/q))

def JSdiv(p,q):
    m = (p+q)/2
    out = (KLdiv(p,m)+KLdiv(q,m))/2
    return out

class Concat(Function):
    def __init__(self,axis=1):
        self.axis=axis
    def forward(self,x1,x2):
        xp = cuda.get_array_module(x1)
        self.len = x2.shape[self.axis]
        out = xp.concatenate((x1, x2),axis=self.axis)
        self.len2 = out.shape[self.axis]
        return out
    def backward(self,dy):
        dx1 = dy[:,:self.len2-self.len]#,:
        dx2 = dy[:,self.len2-self.len:]#
        return dx1,dx2

def concat(x1,x2,axis=1):
    return Concat(axis)(x1,x2)

    
# =============================================================================
# GLU(Now developing?)
# =============================================================================

class GLU(Layer):
    def __init__(self, out_size):
        self.af1 = Dense(out_size)
        self.af2 = Dense(out_size)

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

def T(x):
    a = list(range(x.ndim))
    a[-2:] = a[-2:][::-1]
    return x.transpose(*tuple(a))


class Matmul2(Function):
    def forward(self, q, k):
        self.shapeq = q.shape
        self.shapek = k.shape
        return q @ k
    def backward(self, gy):
        q, k = self.inputs
        gq = matmul2(gy, T(k))
        gk = matmul2(T(q),gy)
        return F.sum_to(gq,self.shapeq), F.sum_to(gk, self.shapek)

def matmul2(q, k):
    return Matmul2()(q, k)


class Multi_Head_Attention(Layer):
    def __init__(self, depth, head, rate, mask=none):
        super().__init__()
        self.depth = depth
        self.head = head
        self.WQ = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="Q")
        self.WK = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="K")
        self.WV = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="V")
        self.WO = Parameter(np.random.randn(depth//head, depth//head)*np.sqrt(1/(depth//head)),name="O")
        self.rate = rate
        self.mask = mask
    def split(self, x):
        N, T, H = x.shape
        return x.reshape(N, T, H//self.head, self.head).transpose(0, 3, 1, 2)

    def concat(self, x):
        N, h, T, H = x.shape
        return x.transpose(0, 2, 3, 1).reshape(N, T, -1)

    def forward(self, i, m):
        q = matmul2(i, self.WQ)
        k = matmul2(m, self.WK)
        v = matmul2(m, self.WV)
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        logit = matmul2(q, k.transpose(0, 1, 3, 2))/np.sqrt(self.depth//self.head)
        a = self.mask(F.softmax(logit,axis=2))
        a = F.dropout(a,self.rate)
        out = matmul2(a,v)
        out = matmul2(out,self.WO)
        out = self.concat(out)
        return F.dropout(out,self.rate)

class Self_Attention(Multi_Head_Attention):
    def __init__(self, depth, head, rate,mask=none):
        super().__init__(depth,head,rate,none)
    def forward(self,i,m):
        return super().forward(i,i)

class FFN(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.l1 = Dense(hidden_size)
        self.l2 = Dense(in_size)
    def forward(self,x):
        x = self.l2(gelu(self.l1(x)))
        return x

Retention_Parallel = True

class Retention(Layer):
    def __init__(self,depth:int,head:int,rate:float):
        super().__init__()
        self.depth = depth
        self.head = head
        self.rate = rate
        self.gamma = 1-2**(-5-np.arange(0,head,dtype=float))
        self.WQ = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="Q")
        self.WK = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="K")
        self.WV = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="V")
        self.WG = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="G")
        self.WO = Parameter(np.random.randn(depth, depth)*np.sqrt(1/(depth)),name="O")
        self.GN = GruopNorm()
        self.swish = Swish()
        self.s = 0
    def split(self, x):
        N, T, H = x.shape
        return x.reshape(N, T, H//self.head, self.head).transpose(0, 3, 1, 2)
    def concat(self, x):
        N, h, T, H = x.shape
        return x.transpose(0, 2, 3, 1).reshape(N, T, -1)
    def _init_s(self):
        self.s = 0
    def forward(self,x,s=None):
        xp = cuda.get_array_module(x)
        if Retention_Parallel:
            Q = self.split(matmul2(x,self.WQ))
            K = self.split(matmul2(x,self.WK))
            V = self.split(matmul2(x,self.WV))
            S = matmul2(Q,T(K))/xp.sqrt(Q.shape[-1])
            M = xp.arange(0,S.shape[2])
            M,N = xp.meshgrid(M,M)
            Map = N-M+1
            D = self.gamma[:,None,None]**(Map-1)
            D *= Map>=1
            D /= F.sum(D,axis=2,keepdims=True)
            D = D[None,:]
            S *= D
            out = matmul2(S, V)
        else:
            Q = self.split(matmul2(x,self.WQ))
            K = self.split(matmul2(x,self.WK))
            V = self.split(matmul2(x,self.WV))
            S = matmul2(T(K),V)
            self.s = self.s*self.gamma[None,:,None,None] + S
            out = matmul2(Q,self.s)
        out = self.concat(self.GN(out))
        out = matmul2((self.swish(matmul2(x,self.WG))*out),self.WO)
        return out

# =============================================================================
# LayerNorm/PositionalEncoding/wrapper(from https://qiita.com/halhorn/items/c91497522be27bde17ce)
# =============================================================================

def Norm(x):
    N,T,D = x.shape
    x1 = x[:]
    #(N,T,D) -> (T*D,N)
    x = x.transpose(1,2,0).reshape(-1,N)
    mean = x.sum(axis=0,keepdims=True)/T*D
    var = (((x-mean)**2).sum(axis=0,keepdims=True)/T*D)**0.5
    out = (x1-mean)/(var+1e-5)
    return out

class LayerNorm(Layer):
    def __init__(self,hidden_size=None):
        super().__init__()
        xp = dezero.cuda.cupy if dezero.cuda.gpu_enable else np
        self.var = Parameter(None)
        self.ma = Parameter(None)
        if not hidden_size == None:
            self.var = Parameter(xp.ones((hidden_size))).reshape(-1,1)
            self.ma = Parameter(xp.zeros((hidden_size))).reshape(-1,1)
    def _init_params(self,x):
        xp = dezero.cuda.cupy if dezero.cuda.gpu_enable else np
        self.var.data = xp.ones((x,1))
        self.ma.data = xp.zeros((x,1))
    def forward(self,x):
        xp = cuda.get_array_module(x)
        x1 = x
        self.shape = x.shape
        n = self.shape[2]
        if self.var.data is None:
            self._init_params(n)
        #(N,T,H)->(N*T,H)
        x = x.reshape(self.shape[2],-1).T
        mean = x.sum(axis=0,keepdims=True)/n
        var1 = (((x - mean)**2).sum(axis=0,keepdims=True)/n)**0.5
        out = (x-mean)/(var1+1e-6)
        out = out*self.var.T+self.ma.T
        return out.T.reshape(*x1.shape)

class GruopNorm(L.BatchNorm):
    def __init__(self,hidden_size=None):
        super().__init__()
    def forward(self,x):
        x = x.transpose(1,0,2,3)
        x_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        x = super().forward(x).reshape(x_shape)
        x = x.reshape(*x_shape).transpose(1,0,2,3)
        return x

class Swish(Layer):
    def __init__(self):
        super().__init__()
        self.beta = 1
    def forward(self,x):
        return x*F.sigmoid(self.beta*x)

class Positional_Encoding1(Layer):
    def __init__(self,input_shape=None,max_len=10000):
        super().__init__()
        if input_shape != None:
            self.map = Parameter(input_shape)
        else:
            self.map = Parameter(None)
    def _init_map(self,x):
        xp = dezero.cuda.cupy if dezero.cuda.gpu_enable else np
        N,T,D = x.shape
        self.map.data = xp.random.randn(1,D)
    def forward(self,x):
        if self.map.data is None:
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
        
class PostNorm_Wrapper(Layer):
    def __init__(self,layer,dropout_rate=0.1):
        super().__init__()
        self.rate = dropout_rate
        self.norm = LayerNorm()
        self.layer = layer
    def forward(self,x,m=Variable(None)):
        ou = self.norm(x)
        if m.data is None:
            ou1 = self.layer(ou)
        else:
            ou1 = self.layer(ou,m)
        ou1 = F.dropout(ou1,self.rate)
        return ou1+ou

class PreNorm_Wrapper(Layer):
    def __init__(self,layer,dropout_rate=0.1):
        super().__init__()
        self.rate = dropout_rate
        self.norm = LayerNorm()
        self.layer = layer
    def forward(self,x,m=Variable(None)):
        ou = self.norm(x)
        if m.data is None:
            ou1 = self.layer(ou)
        else:
            ou1 = self.layer(ou,m)
        ou1 = F.dropout(ou1,self.rate)
        return ou1+x

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
            self.layers += [PreNorm_Wrapper(Self_Attention(depth, head, rate),rate)]
            self.layers += [PreNorm_Wrapper(FFN(depth,hidden),rate)]
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
            self.layers += [(PreNorm_Wrapper(Self_Attention(depth, head, rate,mask=mask),rate),
                             PreNorm_Wrapper(Multi_Head_Attention(depth, head, rate,mask=mask),rate),
                             PreNorm_Wrapper(FFN(depth,hidden),rate))]
    def forward(self,x,s):
        for i in self.layers:
            l1,l2,l3 = i
            x = l1(x)
            x = l2(x,s)
            x = l3(x)
        return x

class Transformer(Layer):
    def __init__(self,
                 depth,
                 hidden=512,
                 hopping=6,
                 head=8,
                 rate=0.25):
        super().__init__()
        self.Decoder = Transformer_Decoder(depth,hidden,hopping,head,rate)
        self.Encoder = Transformer_Encoder(depth,hidden,hopping,head,rate)
    def forward(self,X_En,X_De):
        sorce = self.Encoder(X_De)
        out = self.Decoder(X_En,sorce)
        return out

class GPT_Decoder(Layer):
    def __init__(self,depth:int,
                 hidden:int=2048,
                 hopping:int=128,
                 head:int=8,
                 rate:float=0.25):
        super().__init__()
        self.layers = []
        for i in range(hopping):
            self.layers += [PreNorm_Wrapper(Self_Attention(depth, head, rate,mask=mask),rate),
                             PreNorm_Wrapper(FFN(depth,hidden),rate)]
        self.layers = dezero.models.Sequential(*tuple(self.layers))
    def forward(self,x):
        return self.layers(x)

class RetNet(Layer):
    def __init__(self,
                 depth:int,
                 hidden:int=2048,
                 hopping:int=128,
                 head:int=8,
                 rate:float=0.25):
        super().__init__()
        self.layers = []
        for i in range(hopping):
            self.layers += [PreNorm_Wrapper(Retention(depth, head, rate),rate),
                             PreNorm_Wrapper(FFN(depth,hidden),rate)]
        self.layers = dezero.models.Sequential(*tuple(self.layers))
    def forward(self,x):
        return self.layers(x)
    
# =============================================================================
# AdaBelief
# =============================================================================

class AdaBelief(Optimizer):
    def __init__(self,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        self.ms = {}
        self.vs = {}
    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * ((grad - m)**2 - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)

