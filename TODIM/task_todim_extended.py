# -*- coding: utf-8 -*-
"""
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

An example of how to use the TODIM class.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from TODIM import TODIM
import numpy as np
import matplotlib.pyplot as plt

# Defining new g functions
def g1_TODIM_llamazares (self, x):
    return np.sqrt(x)

def g2_TODIM_llamazares (self,x):
    return 1/(self.getTheta()*np.sqrt(x))

# LLamazares' TODIM
A = TODIM ('decisionMatrix.txt')
A.setG(g1_TODIM_llamazares, g2_TODIM_llamazares)
valores = A.getRCloseness(output=True)


# Lourenzutti's TODIM
B = TODIM ('decisionMatrix.txt')
valoresB = B.getRCloseness(output=True)


import pandas as pd
nome_rotulos = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']

rotulos = pd.DataFrame (nome_rotulos, columns=['rotulo'])
todimA = pd.DataFrame (valores, columns=['Llamazares'])
todimB = pd.DataFrame (valoresB, columns=['Lourenzutti'])
dados = pd.concat([rotulos, todimA, todimB], axis=1)

rankA = dados.sort_values(by=['Llamazares'], ascending=False) 
rankB = dados.sort_values(by=['Lourenzutti'], ascending=False) 

print(rankA)
print(rankB)
print(np.array([rankA.index.values==rankB.index.values]).T)

ax = dados.plot.bar(x='rotulo', rot=0)
plt.show()
# If you don't wanna use the file .txt, you can set the values 
# as lists or numpy arrays
