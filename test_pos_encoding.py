# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:43:49 2023

@author: luigi
"""

import dezero
import NLPset as pe#Positional_Encoding2
import numpy as np
import matplotlib.pyplot as plt

depth= 512#from tensorflow

a = np.random.randn(1,50,depth)

lay = pe.Positional_Encoding2()
map1 = lay(a)

plt.pcolormesh(lay.pe, cmap='plasma')#
plt.xlabel('Depth')
plt.xlim((0, depth))
plt.ylabel('Position')
plt.colorbar()
plt.show()