#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Fuzzy TODIM code
Author: Leandro Lima

This class implements a Fuzzy TODIM [5], based in a generalized TODIM [4] and a simplified TODIM [1,2] algorithm.
In order to use it, you need to inform the decision matrix, criteria's weights and thetas's value.
You can set these parameters in two external .txt files or just call the constructors passing the variables as parameters.

Additionally, one can set their own f, g, distance and comparison functions. Default funtions are those used in [3,5] for F-TODIM.

In the file task_todim.py there is an example showing how to use this class.

For more information about F-TODIM:
    [1] L.F.A.M. Gomes, M.M.P.P. Lima TODIM: Basics and application to multicriteria ranking of projects with environmental impacts
        Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127
    
    [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-TODIM: An intuitionistic fuzzy TODIM to multi-criteria decision
        making. Knowledge-Based Systems 53, (2013), pp. 142-146.

    [3] Lourenzutti, R. and Khroling, R. A study of TODIM in a intuitionistic fuzzy and random environment,
        Expert Systems with Applications, Expert Systems with Applications 40, (2013), pp. 6459-6468

    [4] Llamazares, Bonifacio. An analysis of the generalized TODIM method. European Journal of Operational Research 269, (2018), pp. 1041–1049.

    [5] R. A. Krohling e T. T. M. de Souza, “Combining prospect theory and fuzzy numbers to multi-criteria decision making”, Expert Systems with Applications, vol. 39, nº 13, p. 11487–11493, out. 2012.
 
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from FuzzyNumber import FuzzyNumber

import sys
sys.path.append('../TODIM')
from TODIM import TODIM

class FTODIM(TODIM):
    # FTODIM(matrixD_filename, wegihts_theta_filename)
    # FTODIM(matrixD, weights, theta)
    def __init__ (self, *args):
        nargs = len(args)        
        if nargs == 0 or nargs == 1:
            print ('ERROR: There is no parameter in the construction function')
            raise ValueError
        elif nargs == 2:
            # The .txt file need to be the 1st parameter
            Matrix_filename = args[0]
            WT_filename = args[1]
            # Try to read Weights and Theta's file
            try:
                data_param = np.loadtxt(WT_filename, dtype=float)
                weights = data_param[0,:]
                theta = data_param[1,0]
            except IOError:
                print ('ERROR: there is a problem with %s file. Please, check it again' % (WT_filename ) )
                raise IOError

            # Try to read Matrix's file
            try:
                data_matrix = pd.read_csv(Matrix_filename, sep=";")
                data_matrix_np = data_matrix.applymap(lambda x: FuzzyNumber( np.array( x.split(','), dtype=float ) )).to_numpy()
            except IOError:
                print ('ERROR: there is a problem with %s file. Please, check it again' % (Matrix_filename) )
                raise IOError
        elif nargs == 3:
            matrixD = args[0]
            weights = args[1]
            theta = args[2]
            # Pandas "a1, a2, a3" or "a1, a2, a3, a4" values
            data_matrix_np = matrixD.applymap(lambda x: FuzzyNumber( np.array( x.split(','), dtype=float ) )).to_numpy()

            # Create TODIM object
            super().__init__( data_matrix_np, weights, theta )
            
            # Set fuzzy's specifics to TODIM
            super().setDistance(FuzzyNumber.distanceHamming)
            super().setComparison(FuzzyNumber.cmp)



################################## END CLASS ####################################################

