# -*- coding: utf-8 -*-
"""
Created based in TODIM code.

TODIM code author: Andre Pacheco
Email: pacheco.comp@gmail.com

An example of how to use the FTODIM class.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from FTODIM import FTODIM

import numpy as np


A = FTODIM ('decisionFuzzyMatrix.txt', 'decisionMatrix.txt')

print (A.getRCloseness(verbose=True))
A.plotBars(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'])


# If you don't wanna use the file .txt, you can set the values 
# as lists or numpy arrays
